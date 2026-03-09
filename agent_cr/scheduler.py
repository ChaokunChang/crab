from __future__ import annotations

from collections import deque
from dataclasses import replace
from datetime import datetime
from threading import Lock

from .config import SchedulerConfig
from .contracts import CRPolicy, SandboxInspector, SchedulerStateStore, TelemetrySink
from .ids import JobId, SandboxId
from .models import CheckpointJob, SandboxSnapshot, ScheduleDecision, utc_now
from .telemetry import NoopTelemetrySink


class InMemorySchedulerStateStore(SchedulerStateStore):
    def __init__(self) -> None:
        self._lock = Lock()
        self._last_checkpoint: dict[SandboxId, datetime] = {}
        self._queue: deque[CheckpointJob] = deque()

    def set_last_checkpoint(self, sandbox_id: SandboxId, checkpoint_time: datetime) -> None:
        with self._lock:
            self._last_checkpoint[sandbox_id] = checkpoint_time

    def get_last_checkpoint(self, sandbox_id: SandboxId) -> datetime | None:
        with self._lock:
            return self._last_checkpoint.get(sandbox_id)

    def enqueue_checkpoint_job(self, job: CheckpointJob) -> None:
        with self._lock:
            self._queue.append(job)

    def pop_checkpoint_job(self) -> CheckpointJob | None:
        with self._lock:
            if not self._queue:
                return None
            return self._queue.popleft()

    def pending_jobs(self):
        with self._lock:
            return tuple(self._queue)


class CRScheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        policy: CRPolicy,
        inspector: SandboxInspector,
        state_store: SchedulerStateStore | None = None,
        telemetry: TelemetrySink | None = None,
    ):
        self._config = config
        self._policy = policy
        self._inspector = inspector
        self._state = state_store or InMemorySchedulerStateStore()
        self._telemetry = telemetry or NoopTelemetrySink()

    def evaluate(self, sandbox: SandboxSnapshot) -> ScheduleDecision:
        hydrated = sandbox
        if hydrated.last_checkpoint_at is None:
            last = self._state.get_last_checkpoint(hydrated.sandbox_id)
            if last is not None:
                hydrated = replace(hydrated, last_checkpoint_at=last)
        decision = self._policy.evaluate(hydrated)
        self._telemetry.emit_event(
            "scheduler.evaluate",
            {
                "sandbox_id": str(hydrated.sandbox_id),
                "policy": self._policy.name,
                "should_checkpoint": decision.should_checkpoint,
                "reason": decision.reason,
            },
        )
        return decision

    def submit_checkpoint(
        self,
        sandbox_id: SandboxId,
        *,
        reason: str = "manual",
        metadata: dict[str, object] | None = None,
    ) -> JobId:
        pending_count = len(tuple(self._state.pending_jobs()))
        if pending_count >= self._config.max_pending_jobs:
            raise RuntimeError("scheduler pending queue is full")
        job_id = JobId.new()
        job = CheckpointJob(
            job_id=job_id,
            sandbox_id=sandbox_id,
            requested_at=utc_now(),
            reason=reason,
            metadata=dict(metadata or {}),
        )
        self._state.enqueue_checkpoint_job(job)
        self._telemetry.emit_event(
            "scheduler.submit_checkpoint",
            {
                "sandbox_id": str(sandbox_id),
                "job_id": str(job_id),
                "reason": reason,
            },
        )
        return job_id

    def pop_next_checkpoint_job(self) -> CheckpointJob | None:
        return self._state.pop_checkpoint_job()

    def poll_and_schedule(self, sandbox_id: SandboxId) -> CheckpointJob | None:
        snapshot = self._inspector.inspect(sandbox_id)
        decision = self.evaluate(snapshot)
        if not decision.should_checkpoint:
            return None
        job_id = self.submit_checkpoint(
            sandbox_id=sandbox_id,
            reason=decision.reason,
            metadata={"policy": decision.policy_name, **decision.metadata},
        )
        job = self.pop_next_checkpoint_job()
        if job is None or job.job_id != job_id:
            return None
        return job

    def mark_checkpoint_complete(self, sandbox_id: SandboxId, at: datetime | None = None) -> None:
        ts = at or utc_now()
        self._state.set_last_checkpoint(sandbox_id, ts)
        self._telemetry.emit_event(
            "scheduler.checkpoint_complete",
            {"sandbox_id": str(sandbox_id), "at": ts.isoformat()},
        )

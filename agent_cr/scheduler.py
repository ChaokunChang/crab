from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
from threading import Lock

from .config import SchedulerConfig
from .contracts import CRPolicy, SandboxInspector, SandboxManager, SchedulerStateStore, TelemetrySink
from .ids import SandboxId
from .models import SandboxSnapshot, SchedulerCheckpointDecision, utc_now
from .telemetry import NoopTelemetrySink

logger = logging.getLogger(__name__)


class InMemorySchedulerStateStore(SchedulerStateStore):
    def __init__(self) -> None:
        self._lock = Lock()
        self._last_checkpoint: dict[SandboxId, datetime] = {}

    def set_last_checkpoint(self, sandbox_id: SandboxId, checkpoint_time: datetime) -> None:
        with self._lock:
            self._last_checkpoint[sandbox_id] = checkpoint_time
        logger.debug(
            "Recorded last checkpoint time for sandbox %s at %s",
            sandbox_id,
            checkpoint_time.isoformat(),
        )

    def get_last_checkpoint(self, sandbox_id: SandboxId) -> datetime | None:
        with self._lock:
            return self._last_checkpoint.get(sandbox_id)


class CRScheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        policy: CRPolicy,
        inspector: SandboxInspector,
        sandbox_manager: SandboxManager,
        state_store: SchedulerStateStore | None = None,
        telemetry: TelemetrySink | None = None,
    ):
        self._config = config
        self._policy = policy
        self._inspector = inspector
        self._sandbox_manager = sandbox_manager
        self._state = state_store or InMemorySchedulerStateStore()
        self._telemetry = telemetry or NoopTelemetrySink()

    def evaluate(self, sandbox: SandboxSnapshot) -> SchedulerCheckpointDecision:
        hydrated = sandbox
        if hydrated.last_checkpoint_at is None:
            last = self._state.get_last_checkpoint(hydrated.sandbox_id)
            if last is not None:
                hydrated = replace(hydrated, last_checkpoint_at=last)
                logger.debug(
                    "Hydrated snapshot for sandbox %s with last checkpoint time %s",
                    hydrated.sandbox_id,
                    last.isoformat(),
                )
        policy_decision = self._policy.evaluate(hydrated)
        has_scope = hydrated.process_changed or hydrated.filesystem_changed
        should_checkpoint = policy_decision.should_checkpoint and has_scope
        decision = SchedulerCheckpointDecision(
            should_checkpoint=should_checkpoint,
            checkpoint_process=should_checkpoint and hydrated.process_changed,
            checkpoint_filesystem=should_checkpoint and hydrated.filesystem_changed,
            reason=policy_decision.reason if should_checkpoint or not policy_decision.should_checkpoint else "no_change_signal",
            policy_name=policy_decision.policy_name,
            next_earliest_checkpoint_at=policy_decision.next_earliest_checkpoint_at,
            metadata=dict(policy_decision.metadata),
        )
        log_fn = logger.info if decision.should_checkpoint else logger.debug
        log_fn(
            "Scheduler evaluated sandbox %s with policy=%s should_checkpoint=%s reason=%s process=%s filesystem=%s",
            hydrated.sandbox_id,
            self._policy.name,
            decision.should_checkpoint,
            decision.reason,
            decision.checkpoint_process,
            decision.checkpoint_filesystem,
        )
        self._telemetry.emit_event(
            "scheduler.evaluate",
            {
                "sandbox_id": str(hydrated.sandbox_id),
                "policy": self._policy.name,
                "should_checkpoint": decision.should_checkpoint,
                "reason": decision.reason,
                "checkpoint_process": decision.checkpoint_process,
                "checkpoint_filesystem": decision.checkpoint_filesystem,
            },
        )
        return decision

    def query_checkpoint(self, sandbox_id: SandboxId) -> SchedulerCheckpointDecision:
        logger.debug("Querying scheduler for sandbox %s", sandbox_id)
        self._sandbox_manager.pause(sandbox_id)
        try:
            snapshot = self._inspector.inspect(sandbox_id)
            decision = self.evaluate(snapshot)
            if not decision.should_checkpoint:
                self._sandbox_manager.resume(sandbox_id)
                logger.debug(
                    "Scheduler declined checkpoint for sandbox %s reason=%s",
                    sandbox_id,
                    decision.reason,
                )
            else:
                logger.info(
                    "Scheduler selected checkpoint for sandbox %s reason=%s process=%s filesystem=%s",
                    sandbox_id,
                    decision.reason,
                    decision.checkpoint_process,
                    decision.checkpoint_filesystem,
                )
            return decision
        except Exception:
            try:
                self._sandbox_manager.resume(sandbox_id)
            except Exception:
                logger.exception("Failed to resume sandbox %s after scheduler exception", sandbox_id)
            raise

    def mark_checkpoint_complete(self, sandbox_id: SandboxId, at: datetime | None = None) -> None:
        ts = at or utc_now()
        self._state.set_last_checkpoint(sandbox_id, ts)
        logger.info("Marked checkpoint complete for sandbox %s at %s", sandbox_id, ts.isoformat())
        self._telemetry.emit_event(
            "scheduler.checkpoint_complete",
            {"sandbox_id": str(sandbox_id), "at": ts.isoformat()},
        )

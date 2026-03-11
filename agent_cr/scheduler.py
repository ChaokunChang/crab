from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
from threading import Lock

from .config import SchedulerConfig
from .contracts import SandboxInspector, SandboxManager, SchedulerStateStore, TelemetrySink
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


class CheckpointingPolicy:
    """
    Default checkpointing policy: checkpoint when sandbox is running and either
    - force interval elapsed since last checkpoint, or
    - change signal is present and minimum interval elapsed.
    """

    def __init__(self, config: SchedulerConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "default-checkpointing"

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        changed = snapshot.process_changed or snapshot.filesystem_changed
        checkpoint_process = changed
        checkpoint_filesystem = snapshot.filesystem_changed

        if not snapshot.is_running:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                reason="sandbox_not_running",
                policy_name=self.name,
            )

        request_in_flight = bool(snapshot.metadata.get("llm_request_in_flight", False))
        if self._config.require_change_signal and not changed:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                reason="no_change_signal",
                policy_name=self.name,
            )

        if snapshot.last_checkpoint_at is None:
            if self._config.require_llm_request_for_checkpoint and not request_in_flight:
                return SchedulerCheckpointDecision(
                    should_checkpoint=False,
                    checkpoint_process=False,
                    checkpoint_filesystem=False,
                    reason="llm_request_required",
                    policy_name=self.name,
                )
            reason = "no_previous_checkpoint"
            if self._config.prefer_checkpoint_during_llm_request and request_in_flight:
                reason = "llm_request_window_available"
            return SchedulerCheckpointDecision(
                should_checkpoint=True,
                checkpoint_process=checkpoint_process,
                checkpoint_filesystem=checkpoint_filesystem,
                reason=reason,
                policy_name=self.name,
                metadata={"llm_request_in_flight": request_in_flight},
            )

        elapsed = (snapshot.observed_at - snapshot.last_checkpoint_at).total_seconds()
        min_interval = self._config.min_checkpoint_interval_seconds
        force_after = self._config.force_checkpoint_after_seconds

        if force_after > 0 and elapsed >= force_after:
            return SchedulerCheckpointDecision(
                should_checkpoint=True,
                checkpoint_process=checkpoint_process,
                checkpoint_filesystem=checkpoint_filesystem,
                reason="force_interval_elapsed",
                policy_name=self.name,
                metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
            )

        if elapsed < min_interval:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                reason="minimum_interval_not_elapsed",
                policy_name=self.name,
                metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
            )

        if self._config.require_llm_request_for_checkpoint and not request_in_flight:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                reason="llm_request_required",
                policy_name=self.name,
                metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
            )

        reason = "change_signal_and_interval_elapsed"
        if self._config.prefer_checkpoint_during_llm_request and request_in_flight:
            reason = "llm_request_window_available"
        return SchedulerCheckpointDecision(
            should_checkpoint=True,
            checkpoint_process=checkpoint_process,
            checkpoint_filesystem=checkpoint_filesystem,
            reason=reason,
            policy_name=self.name,
            metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
        )


class CRScheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        inspector: SandboxInspector,
        sandbox_manager: SandboxManager,
        state_store: SchedulerStateStore | None = None,
        telemetry: TelemetrySink | None = None,
    ):
        self._config = config
        self._policy = CheckpointingPolicy(config)
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
        decision = self._policy.evaluate(hydrated)
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

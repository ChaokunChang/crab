from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
from threading import Lock
from typing import Protocol

from .config import SchedulerConfig
from .contracts import Runtime, SandboxInspector, SchedulerStateStore, TelemetrySink
from .ids import SandboxId
from .models import SandboxSnapshot, SchedulerCheckpointDecision, utc_now
from .telemetry import NoopTelemetrySink, start_operation

logger = logging.getLogger(__name__)

_PREEMPTION_NOTICE_KEY = "preemption_notice"
_PREEMPTION_GRACE_SECONDS_KEY = "preemption_grace_remaining_seconds"
_TREE_SEARCH_STEP_KEY = "tree_search_step"


class SchedulerPolicy(Protocol):
    @property
    def name(self) -> str:
        ...

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        ...


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
        checkpoint_process = snapshot.process_changed
        checkpoint_filesystem = changed

        if not snapshot.is_running:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="sandbox_not_running",
                policy_name=self.name,
            )

        request_in_flight = bool(snapshot.metadata.get("llm_request_in_flight", False))

        # Optionally force a full baseline checkpoint before applying the normal change-driven policy.
        if (
            self._config.checkpoint_full_baseline_on_first_checkpoint
            and snapshot.last_checkpoint_at is None
        ):
            reason = "no_previous_checkpoint"
            return SchedulerCheckpointDecision(
                should_checkpoint=True,
                checkpoint_process=True,
                checkpoint_filesystem=True,
                leave_running=False,
                reason=reason,
                policy_name=self.name,
                metadata={"llm_request_in_flight": request_in_flight},
            )

        if self._config.require_change_signal and not changed:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="no_change_signal",
                policy_name=self.name,
            )

        if snapshot.last_checkpoint_at is None:
            if self._config.require_llm_request_for_checkpoint and not request_in_flight:
                return SchedulerCheckpointDecision(
                    should_checkpoint=False,
                    checkpoint_process=False,
                    checkpoint_filesystem=False,
                    leave_running=False,
                    reason="llm_request_required",
                    policy_name=self.name,
                    metadata={"llm_request_in_flight": request_in_flight},
                )

            reason = "change_signal_and_interval_elapsed"
            if self._config.prefer_checkpoint_during_llm_request and request_in_flight:
                reason = "llm_request_window_available"
            return SchedulerCheckpointDecision(
                should_checkpoint=True,
                checkpoint_process=checkpoint_process,
                checkpoint_filesystem=checkpoint_filesystem,
                leave_running=False,
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
                leave_running=False,
                reason="force_interval_elapsed",
                policy_name=self.name,
                metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
            )

        if elapsed < min_interval:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="minimum_interval_not_elapsed",
                policy_name=self.name,
                metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
            )

        if self._config.require_llm_request_for_checkpoint and not request_in_flight:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
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
            leave_running=False,
            reason=reason,
            policy_name=self.name,
            metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
        )


class FaultToleranceCheckpointingPolicy(CheckpointingPolicy):
    @property
    def name(self) -> str:
        return "fault-tolerance"

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        decision = super().evaluate(snapshot)
        if not decision.should_checkpoint:
            return replace(decision, policy_name=self.name)
        return replace(decision, leave_running=True, policy_name=self.name)


class SpotPreemptionCheckpointingPolicy:
    def __init__(self, config: SchedulerConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "spot-preemption"

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        if not snapshot.is_running:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="sandbox_not_running",
                policy_name=self.name,
            )

        notice = bool(snapshot.metadata.get(_PREEMPTION_NOTICE_KEY, False))
        if not notice:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="awaiting_preemption_notice",
                policy_name=self.name,
            )

        remaining = float(snapshot.metadata.get(_PREEMPTION_GRACE_SECONDS_KEY, 0.0))
        if remaining <= 0:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="preemption_budget_expired",
                policy_name=self.name,
                metadata={_PREEMPTION_GRACE_SECONDS_KEY: remaining},
            )

        return SchedulerCheckpointDecision(
            should_checkpoint=True,
            checkpoint_process=True,
            checkpoint_filesystem=True,
            leave_running=False,
            reason="preemption_notice_received",
            policy_name=self.name,
            metadata={_PREEMPTION_GRACE_SECONDS_KEY: remaining},
        )


class TreeSearchCheckpointingPolicy:
    @property
    def name(self) -> str:
        return "tree-search"

    def evaluate(self, snapshot: SandboxSnapshot) -> SchedulerCheckpointDecision:
        if not snapshot.is_running:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="sandbox_not_running",
                policy_name=self.name,
            )
        step = snapshot.metadata.get(_TREE_SEARCH_STEP_KEY)
        if step is None:
            return SchedulerCheckpointDecision(
                should_checkpoint=False,
                checkpoint_process=False,
                checkpoint_filesystem=False,
                leave_running=False,
                reason="tree_search_step_missing",
                policy_name=self.name,
            )
        return SchedulerCheckpointDecision(
            should_checkpoint=True,
            checkpoint_process=True,
            checkpoint_filesystem=True,
            leave_running=True,
            reason="tree_search_step",
            policy_name=self.name,
            metadata={_TREE_SEARCH_STEP_KEY: step},
        )


class CRScheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        inspector: SandboxInspector,
        runtime: Runtime,
        state_store: SchedulerStateStore | None = None,
        telemetry: TelemetrySink | None = None,
        policy: SchedulerPolicy | None = None,
    ):
        self._config = config
        self._policy = policy or CheckpointingPolicy(config)
        self._inspector = inspector
        self._runtime = runtime
        self._state = state_store or InMemorySchedulerStateStore()
        self._telemetry = telemetry or NoopTelemetrySink()

    def evaluate(self, sandbox: SandboxSnapshot) -> SchedulerCheckpointDecision:
        operation = start_operation(
            self._telemetry,
            "scheduler.evaluate",
            {
                "component": "scheduler",
                "sandbox_id": str(sandbox.sandbox_id),
                "policy": self._policy.name,
            },
        )
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
            "Scheduler evaluated sandbox %s with policy=%s should_checkpoint=%s reason=%s "
            "observed_process_changed=%s observed_filesystem_changed=%s "
            "checkpoint_process=%s checkpoint_filesystem=%s leave_running=%s",
            hydrated.sandbox_id,
            self._policy.name,
            decision.should_checkpoint,
            decision.reason,
            hydrated.process_changed,
            hydrated.filesystem_changed,
            decision.checkpoint_process,
            decision.checkpoint_filesystem,
            decision.leave_running,
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
                "leave_running": decision.leave_running,
            },
        )
        operation.finish(
            status="succeeded",
            attributes={
                "sandbox_id": str(hydrated.sandbox_id),
                "should_checkpoint": decision.should_checkpoint,
                "reason": decision.reason,
                "checkpoint_process": decision.checkpoint_process,
                "checkpoint_filesystem": decision.checkpoint_filesystem,
                "leave_running": decision.leave_running,
            },
        )
        return decision

    def query_checkpoint(self, sandbox_id: SandboxId) -> SchedulerCheckpointDecision:
        logger.debug("Querying scheduler for sandbox %s", sandbox_id)
        if self._config.inspect_without_pause:
            return self._query_checkpoint_with_live_inspection(sandbox_id)
        return self._query_checkpoint_with_pause(sandbox_id)

    def _query_checkpoint_with_live_inspection(self, sandbox_id: SandboxId) -> SchedulerCheckpointDecision:
        snapshot = self._inspector.inspect(sandbox_id)
        decision = self.evaluate(snapshot)
        if not decision.should_checkpoint:
            logger.debug(
                "Scheduler declined checkpoint for sandbox %s reason=%s without pausing",
                sandbox_id,
                decision.reason,
            )
            return decision
        if decision.leave_running:
            logger.info(
                "Scheduler selected live-inspection checkpoint for sandbox %s reason=%s observed_process_changed=%s "
                "observed_filesystem_changed=%s checkpoint_process=%s checkpoint_filesystem=%s leave_running=%s",
                sandbox_id,
                decision.reason,
                snapshot.process_changed,
                snapshot.filesystem_changed,
                decision.checkpoint_process,
                decision.checkpoint_filesystem,
                decision.leave_running,
            )
            return decision
        try:
            self._runtime.pause(sandbox_id)
        except Exception:
            fallback_snapshot = self._inspector.inspect(sandbox_id)
            if not fallback_snapshot.is_running:
                decision = self.evaluate(fallback_snapshot)
                logger.info(
                    "Skipping pause for sandbox %s because it is already not running; reason=%s",
                    sandbox_id,
                    decision.reason,
                )
                return decision
            raise
        try:
            paused_snapshot = self._inspect_after_pause(sandbox_id)
            paused_decision = self.evaluate(paused_snapshot)
            if not paused_decision.should_checkpoint:
                if paused_snapshot.is_running:
                    self._runtime.resume(sandbox_id)
                else:
                    self._runtime.sync_runtime_state(sandbox_id, is_running=False)
                logger.debug(
                    "Scheduler declined checkpoint for sandbox %s after live inspection reason=%s",
                    sandbox_id,
                    paused_decision.reason,
                )
            else:
                logger.info(
                    "Scheduler selected live-inspection checkpoint for sandbox %s reason=%s observed_process_changed=%s "
                    "observed_filesystem_changed=%s checkpoint_process=%s checkpoint_filesystem=%s leave_running=%s",
                    sandbox_id,
                    paused_decision.reason,
                    paused_snapshot.process_changed,
                    paused_snapshot.filesystem_changed,
                    paused_decision.checkpoint_process,
                    paused_decision.checkpoint_filesystem,
                    paused_decision.leave_running,
                )
            return paused_decision
        except Exception:
            try:
                snapshot = self._inspector.inspect(sandbox_id)
            except Exception:
                snapshot = None
            try:
                if snapshot is not None and not snapshot.is_running:
                    self._runtime.sync_runtime_state(sandbox_id, is_running=False)
                else:
                    self._runtime.resume(sandbox_id)
            except Exception:
                logger.exception("Failed to resume sandbox %s after scheduler exception", sandbox_id)
            raise

    def _query_checkpoint_with_pause(self, sandbox_id: SandboxId) -> SchedulerCheckpointDecision:
        try:
            self._runtime.pause(sandbox_id)
        except Exception:
            snapshot = self._inspector.inspect(sandbox_id)
            if not snapshot.is_running:
                decision = self.evaluate(snapshot)
                logger.info(
                    "Skipping pause for sandbox %s because it is already not running; reason=%s",
                    sandbox_id,
                    decision.reason,
                )
                return decision
            raise
        try:
            snapshot = self._inspect_after_pause(sandbox_id)
            decision = self.evaluate(snapshot)
            if not decision.should_checkpoint:
                if snapshot.is_running:
                    self._runtime.resume(sandbox_id)
                else:
                    self._runtime.sync_runtime_state(sandbox_id, is_running=False)
                logger.debug(
                    "Scheduler declined checkpoint for sandbox %s reason=%s",
                    sandbox_id,
                    decision.reason,
                )
            else:
                logger.info(
                    "Scheduler selected checkpoint for sandbox %s reason=%s observed_process_changed=%s "
                    "observed_filesystem_changed=%s checkpoint_process=%s checkpoint_filesystem=%s leave_running=%s",
                    sandbox_id,
                    decision.reason,
                    snapshot.process_changed,
                    snapshot.filesystem_changed,
                    decision.checkpoint_process,
                    decision.checkpoint_filesystem,
                    decision.leave_running,
                )
            return decision
        except Exception:
            try:
                snapshot = self._inspector.inspect(sandbox_id)
            except Exception:
                snapshot = None
            try:
                if snapshot is not None and not snapshot.is_running:
                    self._runtime.sync_runtime_state(sandbox_id, is_running=False)
                else:
                    self._runtime.resume(sandbox_id)
            except Exception:
                logger.exception("Failed to resume sandbox %s after scheduler exception", sandbox_id)
            raise

    def _inspect_after_pause(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        snapshot = self._inspector.inspect(sandbox_id)
        if not snapshot.is_running:
            try:
                runtime_state = self._runtime.inspect_runtime(sandbox_id)
            except Exception:
                runtime_state = None
            if runtime_state is not None and runtime_state.is_running:
                logger.debug(
                    "Using runtime state to keep sandbox %s marked running after successful pause; "
                    "inspector snapshot was stale",
                    sandbox_id,
                )
                snapshot = replace(snapshot, is_running=True)
        return snapshot

    def mark_checkpoint_complete(self, sandbox_id: SandboxId, at: datetime | None = None) -> None:
        ts = at or utc_now()
        self._state.set_last_checkpoint(sandbox_id, ts)
        logger.info("Marked checkpoint complete for sandbox %s at %s", sandbox_id, ts.isoformat())
        self._telemetry.emit_event(
            "scheduler.checkpoint_complete",
            {"sandbox_id": str(sandbox_id), "at": ts.isoformat()},
        )

from __future__ import annotations

from datetime import timedelta

from ..config import PolicyConfig
from ..contracts import CRPolicy
from ..models import ScheduleDecision, SandboxSnapshot


class DefaultHeuristicPolicy(CRPolicy):
    """
    Default policy: checkpoint when sandbox is running and either
    - force interval elapsed since last checkpoint, or
    - change signal is present and minimum interval elapsed.
    """

    def __init__(self, config: PolicyConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "default-heuristic"

    def evaluate(self, snapshot: SandboxSnapshot) -> ScheduleDecision:
        if not snapshot.is_running:
            return ScheduleDecision(
                should_checkpoint=False,
                reason="sandbox_not_running",
                policy_name=self.name,
            )

        changed = snapshot.process_changed or snapshot.filesystem_changed
        request_in_flight = bool(snapshot.metadata.get("llm_request_in_flight", False))
        if self._config.require_change_signal and not changed:
            return ScheduleDecision(
                should_checkpoint=False,
                reason="no_change_signal",
                policy_name=self.name,
            )

        if snapshot.last_checkpoint_at is None:
            if self._config.require_llm_request_for_checkpoint and not request_in_flight:
                return ScheduleDecision(
                    should_checkpoint=False,
                    reason="llm_request_required",
                    policy_name=self.name,
                )
            reason = "no_previous_checkpoint"
            if self._config.prefer_checkpoint_during_llm_request and request_in_flight:
                reason = "llm_request_window_available"
            return ScheduleDecision(
                should_checkpoint=True,
                reason=reason,
                policy_name=self.name,
                metadata={"llm_request_in_flight": request_in_flight},
            )

        elapsed = (snapshot.observed_at - snapshot.last_checkpoint_at).total_seconds()
        min_interval = self._config.min_checkpoint_interval_seconds
        force_after = self._config.force_checkpoint_after_seconds

        if force_after > 0 and elapsed >= force_after:
            return ScheduleDecision(
                should_checkpoint=True,
                reason="force_interval_elapsed",
                policy_name=self.name,
                metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
            )

        if elapsed < min_interval:
            return ScheduleDecision(
                should_checkpoint=False,
                reason="minimum_interval_not_elapsed",
                policy_name=self.name,
                next_earliest_checkpoint_at=snapshot.last_checkpoint_at
                + timedelta(seconds=min_interval),
                metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
            )

        if self._config.require_llm_request_for_checkpoint and not request_in_flight:
            return ScheduleDecision(
                should_checkpoint=False,
                reason="llm_request_required",
                policy_name=self.name,
                metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
            )

        reason = "change_signal_and_interval_elapsed"
        if self._config.prefer_checkpoint_during_llm_request and request_in_flight:
            reason = "llm_request_window_available"
        return ScheduleDecision(
            should_checkpoint=True,
            reason=reason,
            policy_name=self.name,
            metadata={"elapsed_seconds": elapsed, "llm_request_in_flight": request_in_flight},
        )

from __future__ import annotations

from integrations.agents.base import BaseAgent


SIMULATED_WARMUP_STEPS = 3


class SimulatedAgent(BaseAgent):
    agent_type = "simulated"
    requires_manual_task_launch = False

    def wait_for_task_ready(self) -> None:
        self._set_status(self.poll_status())

    def perform_task(self) -> None:
        self.wait_for_progress(minimum_actions=SIMULATED_WARMUP_STEPS)
        self.wait_for_sandbox_exit()

    def poll_status(self) -> dict[str, object]:
        return self.wait_for_http_json(self.sandbox.status_url)

    def wait_for_progress(self, *, minimum_actions: int) -> dict[str, object]:
        self.wait_for_condition(
            lambda: self._enough_progress(self.poll_status(), minimum_actions=minimum_actions),
            timeout_s=45.0,
        )
        payload = self.poll_status()
        self._record_activity(payload)
        return payload

    def wait_for_action_delta(self, *, delta: int) -> dict[str, object]:
        baseline = self._total_actions(self.sandbox.last_status)
        self.wait_for_condition(
            lambda: self._total_actions(self.poll_status()) >= baseline + delta,
            timeout_s=45.0,
        )
        payload = self.poll_status()
        self._record_activity(payload)
        return payload

    @staticmethod
    def _total_actions(payload: dict[str, object]) -> int:
        return int(payload.get("total_actions", 0))

    @classmethod
    def _enough_progress(cls, payload: dict[str, object], *, minimum_actions: int) -> bool:
        return (
            cls._total_actions(payload) >= minimum_actions
            and int(payload["filesystem_actions"]) >= 1
            and int(payload["process_actions"]) >= 1
            and int(payload["network_actions"]) >= 1
            and int(payload["stateful_actions"]) >= 1
        )

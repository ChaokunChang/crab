from __future__ import annotations

import unittest

from integrations.llm_services.simulated_for_iflow.service import ScriptedLLMState, default_script_steps, handle_request


class SimulatedForIFlowServiceTests(unittest.TestCase):
    def test_scripted_service_returns_iflow_compatible_tool_call(self) -> None:
        response = handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-iflow"},
            payload={},
            state=ScriptedLLMState(default_script_steps(idle_delay_ms=0)),
        )

        message = response["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "run_shell_command")
        self.assertIn("command", message["tool_calls"][0]["function"]["arguments"])

    def test_scripted_service_eventually_returns_final_stop_message(self) -> None:
        state = ScriptedLLMState(default_script_steps(idle_delay_ms=0))
        last_response = {}
        for _ in range(4):
            last_response = handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-iflow"},
                payload={},
                state=state,
            )

        self.assertEqual(last_response["choices"][0]["finish_reason"], "stop")
        self.assertIn("Summarize", last_response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    unittest.main()

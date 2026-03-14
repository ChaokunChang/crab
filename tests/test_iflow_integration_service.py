from __future__ import annotations

import unittest

from agents.iflow_integration.service import ManualLLMState


class ManualLLMStateTests(unittest.TestCase):
    def test_enqueue_run_shell_command_returns_tool_call(self) -> None:
        state = ManualLLMState(default_sandbox_id="sbx-manual")
        state.enqueue_run_shell_command(command='sh -lc "echo hi >/tmp/hi.txt"')

        response = state.next_response(
            path="/v1/chat/completions",
            headers={},
            payload={},
        )

        message = response["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "run_shell_command")
        self.assertIn("/tmp/hi.txt", message["tool_calls"][0]["function"]["arguments"])

    def test_enqueue_final_response_returns_stop_message(self) -> None:
        state = ManualLLMState(default_sandbox_id="sbx-manual")
        state.enqueue_final_response(content="All done.")

        response = state.next_response(
            path="/v1/chat/completions",
            headers={},
            payload={},
        )

        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["choices"][0]["message"]["content"], "All done.")


if __name__ == "__main__":
    unittest.main()

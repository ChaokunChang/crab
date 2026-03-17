from __future__ import annotations

import unittest

from integrations.llm_services.manual.service import ManualLLMState


class ManualLLMStateTests(unittest.TestCase):
    def test_enqueue_run_shell_command_returns_tool_call_for_explicit_sandbox(self) -> None:
        state = ManualLLMState()
        state.enqueue_run_shell_command(command='sh -lc "echo hi >/tmp/hi.txt"', sandbox_id="sbx-manual")

        response = state.next_response(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-manual"},
            payload={},
        )

        message = response["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "run_shell_command")
        self.assertIn("/tmp/hi.txt", message["tool_calls"][0]["function"]["arguments"])

    def test_enqueue_final_response_returns_stop_message_for_explicit_sandbox(self) -> None:
        state = ManualLLMState()
        state.enqueue_final_response(content="All done.", sandbox_id="sbx-manual")

        response = state.next_response(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-manual"},
            payload={},
        )

        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["choices"][0]["message"]["content"], "All done.")

    def test_enqueue_requires_explicit_sandbox_id(self) -> None:
        state = ManualLLMState()

        with self.assertRaisesRegex(ValueError, "sandbox_id is required"):
            state.enqueue_run_shell_command(command='sh -lc "echo hi"', sandbox_id="")

    def test_next_response_requires_request_sandbox_identity(self) -> None:
        state = ManualLLMState()

        with self.assertRaisesRegex(ValueError, "missing sandbox identity"):
            state.next_response(
                path="/v1/chat/completions",
                headers={},
                payload={},
            )


if __name__ == "__main__":
    unittest.main()

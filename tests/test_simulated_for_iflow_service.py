from __future__ import annotations

import json
import unittest

from integrations.llm_services.simulated_for_iflow.service import SimulatedLLMState, handle_request


class SimulatedForIFlowServiceTests(unittest.TestCase):
    def test_first_request_returns_run_shell_command_for_transient_process(self) -> None:
        response = handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-iflow"},
            payload={},
            state=SimulatedLLMState(response_delay_ms=0, max_tool_calls_before_finish=3),
        )

        message = response["choices"][0]["message"]
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "run_shell_command")
        arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
        self.assertIn("transient-process", arguments["command"])

    def test_service_returns_finish_then_restarts_cycle(self) -> None:
        state = SimulatedLLMState(response_delay_ms=0, max_tool_calls_before_finish=2)
        headers = {"X-Agent-Sandbox-Id": "sbx-iflow"}

        first_response = handle_request(path="/v1/chat/completions", headers=headers, payload={}, state=state)
        second_response = handle_request(path="/v1/chat/completions", headers=headers, payload={}, state=state)
        finish_response = handle_request(path="/v1/chat/completions", headers=headers, payload={}, state=state)
        restarted_response = handle_request(path="/v1/chat/completions", headers=headers, payload={}, state=state)

        self.assertEqual(first_response["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(second_response["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(finish_response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(finish_response["choices"][0]["message"]["content"], "Done.")
        self.assertEqual(restarted_response["choices"][0]["finish_reason"], "tool_calls")
        restarted_arguments = json.loads(restarted_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
        self.assertIn("transient-process", restarted_arguments["command"])

    def test_different_senders_maintain_independent_counters(self) -> None:
        state = SimulatedLLMState(response_delay_ms=0, max_tool_calls_before_finish=1)

        first_sender_tool = handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-iflow-a"},
            payload={},
            state=state,
        )
        second_sender_tool = handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-iflow-b"},
            payload={},
            state=state,
        )
        first_sender_finish = handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-iflow-a"},
            payload={},
            state=state,
        )

        self.assertEqual(first_sender_tool["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(second_sender_tool["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(first_sender_finish["choices"][0]["finish_reason"], "stop")

    def test_advertised_run_shell_command_rotates_across_all_matching_variants(self) -> None:
        state = SimulatedLLMState(response_delay_ms=0, max_tool_calls_before_finish=9)
        payload = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "run_shell_command",
                        "description": "x",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ]
        }

        responses = [
            handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-iflow"},
                payload=payload,
                state=state,
            )
            for _ in range(9)
        ]

        commands = [
            json.loads(response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
            for response in responses
        ]

        self.assertEqual(len(set(commands)), 9)
        self.assertTrue(any("cat <<'EOF'" in command for command in commands))
        self.assertTrue(any("http.server 8123" in command for command in commands))
        self.assertTrue(any("find /work -maxdepth 2 -mindepth 1" in command for command in commands))
        self.assertTrue(any("env | sort | sed -n '1,12p'" in command for command in commands))
        self.assertTrue(any("counter=/work/iflow-probe/counter.txt" in command for command in commands))
        self.assertTrue(any("journal-entry %s" in command for command in commands))
        self.assertTrue(any("xargs -r sha256sum" in command for command in commands))
        self.assertTrue(any("heavy-blob.bin" in command for command in commands))

    def test_request_and_response_events_record_phase(self) -> None:
        state = SimulatedLLMState(response_delay_ms=0, max_tool_calls_before_finish=3)

        handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-iflow"},
            payload={},
            state=state,
        )

        snapshot = state.snapshot()
        self.assertEqual(snapshot["events"][0]["phase"], "transient_process")
        self.assertEqual(snapshot["events"][1]["phase"], "transient_process")


if __name__ == "__main__":
    unittest.main()

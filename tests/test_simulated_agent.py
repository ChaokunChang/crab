from __future__ import annotations

import unittest

from simulated_agent.agent_cli import parse_anthropic_tool_calls, parse_openai_tool_calls
from simulated_agent.service import (
    SimulatedLLMState,
    build_anthropic_response,
    build_openai_response,
    handle_request,
)


class SimulatedAgentTests(unittest.TestCase):
    def test_openai_response_contains_tool_calls(self) -> None:
        payload = build_openai_response("append_journal", 3)
        tool_calls = parse_openai_tool_calls(payload)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "append_journal")
        self.assertEqual(tool_calls[0]["input"]["filename"], "journal.log")

    def test_anthropic_response_contains_tool_use_block(self) -> None:
        payload = build_anthropic_response("spawn_probe", 4)
        tool_calls = parse_anthropic_tool_calls(payload)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "spawn_probe")
        self.assertEqual(tool_calls[0]["input"]["message"], "probe 4")

    def test_service_serves_both_provider_endpoints(self) -> None:
        state = SimulatedLLMState()
        openai_payload = handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-1"},
            payload={
                "model": "simulated-openai",
                "messages": [{"role": "user", "content": "continue"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "read_workdir",
                            "description": "x",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            state=state,
        )
        self.assertEqual(
            openai_payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read_workdir",
        )

        anthropic_payload = handle_request(
            path="/v1/messages",
            headers={"X-Agent-Sandbox-Id": "sbx-2"},
            payload={
                "model": "simulated-anthropic",
                "messages": [{"role": "user", "content": "continue"}],
                "tools": [
                    {
                        "name": "fetch_proxy_health",
                        "description": "x",
                        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
                    }
                ],
            },
            state=state,
        )
        self.assertEqual(anthropic_payload["content"][0]["name"], "fetch_proxy_health")


if __name__ == "__main__":
    unittest.main()

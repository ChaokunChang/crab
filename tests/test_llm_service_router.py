from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from integrations.llm_services.router import BenchmarkLLMRouter


class BenchmarkLLMRouterTests(unittest.TestCase):
    def test_router_dispatches_to_simulated_service_for_registered_sandbox(self) -> None:
        router = BenchmarkLLMRouter()
        router.register_sandbox(sandbox_id="sbx-sim", llm_service_type="simulated")

        response = router.handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-sim"},
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
        )

        self.assertEqual(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "read_workdir",
        )

    def test_router_dispatches_to_iflow_simulated_service_for_registered_sandbox(self) -> None:
        router = BenchmarkLLMRouter()
        router.register_sandbox(sandbox_id="sbx-iflow", llm_service_type="simulated_for_iflow")

        response = router.handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-iflow"},
            payload={
                "messages": [{"role": "user", "content": "continue"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "run_shell_command",
                            "description": "x",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
        )

        self.assertEqual(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "run_shell_command",
        )

    def test_router_control_request_targets_registered_manual_service(self) -> None:
        router = BenchmarkLLMRouter()
        router.register_sandbox(sandbox_id="sbx-manual", llm_service_type="manual")

        router.handle_control_request(
            path="/control/run_shell_command",
            payload={"sandbox_id": "sbx-manual", "command": 'sh -lc "echo hi"'},
        )
        response = router.handle_request(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-manual"},
            payload={},
        )

        self.assertEqual(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "run_shell_command",
        )

    def test_router_dispatches_to_iflow_trace_replay_and_deduplicates_duplicate_tool_requests_after_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "request",
                                "data": {
                                    "model": "trace-model",
                                    "messages": [{"role": "user", "content": "first"}],
                                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "tool_calls",
                                            "message": {
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [
                                                    {
                                                        "id": "call-first",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "run_shell_command",
                                                            "arguments": json.dumps(
                                                                {"command": 'sh -lc "echo first >/tmp/first.txt"'},
                                                                sort_keys=True,
                                                            ),
                                                        },
                                                    }
                                                ],
                                            },
                                        }
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "request",
                                "data": {
                                    "model": "trace-model",
                                    "messages": [{"role": "user", "content": "second"}],
                                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "tool_calls",
                                            "message": {
                                                "role": "assistant",
                                                "content": None,
                                                "tool_calls": [
                                                    {
                                                        "id": "call-second",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "run_shell_command",
                                                            "arguments": json.dumps(
                                                                {"command": 'sh -lc "echo second >/tmp/second.txt"'},
                                                                sort_keys=True,
                                                            ),
                                                        },
                                                    }
                                                ],
                                            },
                                        }
                                    ]
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            router = BenchmarkLLMRouter()
            router.register_sandbox(
                sandbox_id="sbx-replay",
                llm_service_type="iflow_trace_replay",
                llm_service_config={"trace_path": str(trace_path)},
            )

            first = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            checkpoint_metadata = router.checkpoint_metadata("sbx-replay")
            second = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            router.restore_from_checkpoint_metadata("sbx-replay", checkpoint_metadata)
            replayed_second = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            router.reset_sandbox("sbx-replay")
            replayed_first = router.handle_request(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx-replay"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        first_command = json.loads(first["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        second_command = json.loads(second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        replayed_command = json.loads(replayed_second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        replayed_first_command = json.loads(
            replayed_first["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["command"]
        self.assertEqual(first_command, 'sh -lc "echo first >/tmp/first.txt"')
        self.assertEqual(second_command, 'sh -lc "echo second >/tmp/second.txt"')
        self.assertIn("/dev/null", replayed_command)
        self.assertEqual(replayed_first_command, 'sh -lc "echo first >/tmp/first.txt"')


if __name__ == "__main__":
    unittest.main()

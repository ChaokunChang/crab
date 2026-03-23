from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.generate_termnius_iflow_replay_dataset import generate_dataset
from integrations.llm_services.iflow_trace_replay.service import (
    TraceReplayLLMState,
    _lookup_stats,
    _normalized_lookup_messages,
    parse_replay_trace,
)


class IFlowTraceReplayTests(unittest.TestCase):
    @staticmethod
    def _request_response_lines(*contents: str) -> list[str]:
        lines: list[str] = []
        for content in contents:
            lines.append(
                json.dumps(
                    {
                        "type": "request",
                        "data": {
                            "model": "trace-model",
                            "messages": [{"role": "user", "content": content}],
                            "tools": [],
                        },
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "type": "response",
                        "data": {"choices": [{"message": {"content": content}}]},
                    }
                )
            )
        return lines

    @staticmethod
    def _tool_request_response_lines(*items: tuple[str, str]) -> list[str]:
        lines: list[str] = []
        for content, command in items:
            lines.append(
                json.dumps(
                    {
                        "type": "request",
                        "data": {
                            "model": "trace-model",
                            "messages": [{"role": "user", "content": content}],
                            "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                        },
                    }
                )
            )
            lines.append(
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
                                                "id": f"call-{content}",
                                                "type": "function",
                                                "function": {
                                                    "name": "run_shell_command",
                                                    "arguments": json.dumps({"command": command}, sort_keys=True),
                                                },
                                            }
                                        ],
                                    },
                                }
                            ]
                        },
                    }
                )
            )
        return lines

    def test_parse_replay_trace_skips_noise_and_tracks_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    [
                        "Proxy error: Upstream request timed out",
                        json.dumps({"type": "proxy_exception", "data": {"message": "timeout"}}),
                        json.dumps(
                            {
                                "type": "response",
                                "data": json.dumps({"choices": [{"message": {"content": "first"}}]}),
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {"choices": [{"message": {"content": "second"}}]},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            parsed = parse_replay_trace(trace_path)

        self.assertEqual(len(parsed.responses), 2)
        self.assertEqual(parsed.responses[0]["choices"][0]["message"]["content"], "first")
        self.assertEqual(parsed.responses[1]["choices"][0]["message"]["content"], "second")
        self.assertEqual(list(parsed.malformed_lines), [1])

    def test_trace_replay_state_duplicate_tool_request_returns_dummy_tool_call(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._tool_request_response_lines(
                        ("first", 'sh -lc "echo first >/tmp/first.txt"'),
                        ("second", 'sh -lc "echo second >/tmp/second.txt"'),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first_index, first = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            second_index, second = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            replayed_index, replayed = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            state.reset()
            reset_index, reset_response = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        self.assertEqual(first_index, 1)
        self.assertEqual(second_index, 2)
        self.assertEqual(replayed_index, 2)
        self.assertEqual(reset_index, 1)
        first_command = json.loads(first["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        second_command = json.loads(second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        replayed_command = json.loads(replayed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        reset_command = json.loads(reset_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"]
        self.assertEqual(first_command, 'sh -lc "echo first >/tmp/first.txt"')
        self.assertEqual(second_command, 'sh -lc "echo second >/tmp/second.txt"')
        self.assertIn("/dev/null", replayed_command)
        self.assertEqual(reset_command, 'sh -lc "echo first >/tmp/first.txt"')

    def test_trace_replay_state_snapshot_reports_consumed_response_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(self._request_response_lines("first", "second")),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first_index, first = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "first"}], "tools": []},
            )
            snapshot = state.snapshot()

        self.assertEqual(first_index, 1)
        self.assertEqual(first["choices"][0]["message"]["content"], "first")
        self.assertEqual(snapshot["consumed_response_count"], 1)
        self.assertEqual(snapshot["trace_cursor"], 1)

    def test_trace_replay_state_matches_escape_only_formatting_differences(self) -> None:
        trace_content = 'If potentially_problematic_new_string is console.log("Hello\nWorld"), fix it.'
        live_content = 'If potentially_problematic_new_string is console.log(\\"Hello\\\\nWorld\\"), fix it.'

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(self._request_response_lines(trace_content)),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            strict_hash, _ = _lookup_stats(
                "/v1/chat/completions",
                {"model": "trace-model", "messages": [{"role": "user", "content": live_content}]},
            )
            trace_hash, _ = _lookup_stats(
                "/v1/chat/completions",
                {"model": "trace-model", "messages": [{"role": "user", "content": trace_content}]},
            )
            self.assertNotEqual(strict_hash, trace_hash)

            replayed_index, replayed = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": live_content}]},
            )

        self.assertEqual(replayed_index, 1)
        self.assertEqual(replayed["choices"][0]["message"]["content"], trace_content)

    def test_trace_replay_state_unmatched_request_falls_back_to_cursor_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(self._request_response_lines("first", "second")),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0})

            first_index, first = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "unmatched"}], "tools": []},
            )
            second_index, second = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "second"}], "tools": []},
            )
            snapshot = state.snapshot()

        self.assertEqual(first_index, 1)
        self.assertEqual(first["choices"][0]["message"]["content"], "first")
        self.assertEqual(second_index, 2)
        self.assertEqual(second["choices"][0]["message"]["content"], "second")
        self.assertEqual(snapshot["consumed_response_count"], 2)
        self.assertEqual(snapshot["trace_cursor"], 2)

    def test_trace_replay_state_restore_after_cursor_fallback_preserves_prefix_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(self._request_response_lines("first", "second", "third")),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0})

            state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "first"}], "tools": []},
            )
            state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "unmatched"}], "tools": []},
            )

            state.restore(consumed_response_count=2)
            restored_index, restored = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "third"}], "tools": []},
            )

        self.assertEqual(restored_index, 3)
        self.assertEqual(restored["choices"][0]["message"]["content"], "third")

    def test_trace_replay_state_restore_rewinds_duplicate_tracking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._tool_request_response_lines(
                        ("first", 'sh -lc "echo first >/tmp/first.txt"'),
                        ("second", 'sh -lc "echo second >/tmp/second.txt"'),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

            state.restore(consumed_response_count=1)
            restored_index, restored = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        restored_command = json.loads(
            restored["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["command"]
        self.assertEqual(restored_index, 2)
        self.assertEqual(restored_command, 'sh -lc "echo second >/tmp/second.txt"')

    def test_trace_replay_state_restore_preserves_dummy_for_consumed_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._tool_request_response_lines(
                        ("first", 'sh -lc "echo first >/tmp/first.txt"'),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

            state.restore(consumed_response_count=1)
            replayed_index, replayed = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        replayed_command = json.loads(
            replayed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["command"]
        self.assertEqual(replayed_index, 1)
        self.assertIn("/dev/null", replayed_command)

    def test_trace_replay_state_duplicate_later_tool_request_returns_dummy_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._tool_request_response_lines(
                        ("first", "ls -la /app/process_data.sh"),
                        ("second", "chmod +x /app/process_data.sh"),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            _, second = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            replayed_index, replayed = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "second"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        self.assertEqual(replayed_index, 2)
        self.assertNotEqual(
            second["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            replayed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
        )
        self.assertIn(
            "/dev/null",
            json.loads(replayed["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"],
        )

    def test_trace_replay_state_duplicate_tool_request_does_not_advance_progress_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._tool_request_response_lines(
                        ("first", 'sh -lc "echo first >/tmp/first.txt"'),
                        ("second", 'sh -lc "echo second >/tmp/second.txt"'),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0})

            state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            duplicate_index, duplicate_response = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            snapshot = state.snapshot()

        self.assertEqual(duplicate_index, 1)
        duplicate_command = json.loads(
            duplicate_response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
        )["command"]
        self.assertIn("/dev/null", duplicate_command)
        self.assertEqual(snapshot["consumed_response_count"], 1)
        self.assertEqual(snapshot["duplicate_response_count"], 1)
        self.assertEqual(snapshot["trace_cursor"], 1)

    def test_trace_replay_state_duplicate_tool_request_keeps_followup_request_matchable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            original_arguments = json.dumps({"command": 'sh -lc "echo first >/tmp/first.txt"'}, sort_keys=True)
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
                                                "content": "Let me do the work.",
                                                "tool_calls": [
                                                    {
                                                        "id": "call-first",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "run_shell_command",
                                                            "arguments": original_arguments,
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
                                    "messages": [
                                        {"role": "user", "content": "first"},
                                        {
                                            "role": "assistant",
                                            "content": "Let me do the work.",
                                            "tool_calls": [
                                                {
                                                    "id": "call-first",
                                                    "type": "function",
                                                    "function": {
                                                        "name": "run_shell_command",
                                                        "arguments": original_arguments,
                                                    },
                                                }
                                            ],
                                        },
                                        {"role": "tool", "tool_call_id": "call-first", "content": "original tool output"},
                                    ],
                                    "tools": [],
                                },
                            }
                        ),
                        json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "done"}}]}}),
                    ]
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0})

            _, original = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            duplicate_index, duplicate = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "first"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            duplicate_arguments = duplicate["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            followup_index, followup = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [
                        {"role": "user", "content": "first"},
                        {
                            "role": "assistant",
                            "content": duplicate["choices"][0]["message"]["content"],
                            "tool_calls": [
                                {
                                    "id": duplicate["choices"][0]["message"]["tool_calls"][0]["id"],
                                    "type": "function",
                                    "function": {
                                        "name": duplicate["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
                                        "arguments": duplicate_arguments,
                                    },
                                }
                            ],
                        },
                        {"role": "tool", "tool_call_id": "call-first", "content": "dummy tool output"},
                    ],
                    "tools": [],
                },
            )

        self.assertEqual(duplicate_index, 1)
        self.assertEqual(followup_index, 2)
        self.assertEqual(original["choices"][0]["message"]["content"], duplicate["choices"][0]["message"]["content"])
        self.assertEqual(
            original["choices"][0]["message"]["tool_calls"][0]["id"],
            duplicate["choices"][0]["message"]["tool_calls"][0]["id"],
        )
        self.assertIn("/dev/null", json.loads(duplicate_arguments)["command"])
        self.assertEqual(followup["choices"][0]["message"]["content"], "done")

    def test_trace_replay_state_duplicate_non_shell_tool_request_returns_dummy_arguments(self) -> None:
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
                                    "messages": [{"role": "user", "content": "read first"}],
                                    "tools": [
                                        {
                                            "type": "function",
                                            "function": {
                                                "name": "read_file",
                                                "parameters": {
                                                    "type": "object",
                                                    "properties": {"absolute_path": {"type": "string"}},
                                                    "required": ["absolute_path"],
                                                },
                                            },
                                        }
                                    ],
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
                                                "content": "Let me read the file.",
                                                "tool_calls": [
                                                    {
                                                        "id": "call-read",
                                                        "type": "function",
                                                        "function": {
                                                            "name": "read_file",
                                                            "arguments": json.dumps(
                                                                {"absolute_path": "/app/input.txt"},
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
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0})

            _, original = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "read first"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"absolute_path": {"type": "string"}},
                                    "required": ["absolute_path"],
                                },
                            },
                        }
                    ],
                },
            )
            duplicate_index, duplicate = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "read first"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "parameters": {
                                    "type": "object",
                                    "properties": {"absolute_path": {"type": "string"}},
                                    "required": ["absolute_path"],
                                },
                            },
                        }
                    ],
                },
            )

        self.assertEqual(duplicate_index, 1)
        self.assertEqual(
            json.loads(duplicate["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]),
            {"absolute_path": "/dev/null"},
        )

    def test_trace_replay_state_duplicate_read_only_shell_request_returns_dummy_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._tool_request_response_lines(
                        ("check pip", "pip3 --version"),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0})

            _, original = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "check pip"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            duplicate_index, duplicate = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "check pip"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        self.assertEqual(duplicate_index, 1)
        self.assertIn(
            "/dev/null",
            json.loads(duplicate["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"],
        )

    def test_trace_replay_state_duplicate_system_setup_shell_request_preserves_original_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            original_command = "apt-get install -y python3-pip && pip3 install --break-system-packages pyarrow"
            trace_path.write_text(
                "\n".join(
                    self._tool_request_response_lines(
                        (
                            "install deps",
                            original_command,
                        ),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0})

            _, original = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "install deps"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )
            duplicate_index, duplicate = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [{"role": "user", "content": "install deps"}],
                    "tools": [{"type": "function", "function": {"name": "run_shell_command"}}],
                },
            )

        self.assertEqual(duplicate_index, 1)
        self.assertEqual(original, duplicate)
        self.assertEqual(
            json.loads(duplicate["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])["command"],
            original_command,
        )

    def test_trace_replay_state_applies_default_response_delay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(self._request_response_lines("first")) + "\n",
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            with patch("integrations.llm_services.iflow_trace_replay.service.time.sleep") as sleep:
                _, response = state.next_response(
                    path="/v1/chat/completions",
                    headers={"X-Agent-Sandbox-Id": "sbx"},
                    payload={"model": "trace-model", "messages": [{"role": "user", "content": "first"}], "tools": []},
                )

        sleep.assert_called_once_with(0.25)
        self.assertEqual(response["choices"][0]["message"]["content"], "first")

    def test_trace_replay_state_normalizes_tool_result_contents(self) -> None:
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
                                    "messages": [
                                        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "clock", "arguments": "{}"}}]},
                                        {"role": "tool", "tool_call_id": "call-1", "content": "2026-01-01T00:00:00Z"},
                                    ],
                                    "tools": [],
                                },
                            }
                        ),
                        json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "done"}}]}}),
                    ]
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            _, response = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [
                        {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "clock", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "call-1", "content": "2027-02-03T04:05:06Z"},
                    ],
                    "tools": [],
                },
            )

        self.assertEqual(response["choices"][0]["message"]["content"], "done")

    def test_trace_replay_state_normalizes_volatile_text_in_messages(self) -> None:
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
                                    "messages": [
                                        {
                                            "role": "user",
                                            "content": (
                                                "Today's date is Tuesday, February 24, 2026. "
                                                "See http://127.0.0.1:43123/v1 and request "
                                                "123e4567-e89b-12d3-a456-426614174000 from /tmp/run-123/log.txt."
                                            ),
                                        }
                                    ],
                                    "tools": [],
                                },
                            }
                        ),
                        json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "done"}}]}}),
                    ]
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            _, response = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                "Today's date is Thursday, March 19, 2026. "
                                "See http://10.250.0.1:35871/v1 and request "
                                "923e4567-e89b-12d3-a456-426614174999 from /tmp/agent_cr_scenario_bench_qc7nbuio/log.txt."
                            ),
                        }
                    ],
                    "tools": [],
                },
            )

        self.assertEqual(response["choices"][0]["message"]["content"], "done")

    def test_trace_replay_state_normalizes_iflow_bootstrap_messages(self) -> None:
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
                                    "messages": [
                                        {"role": "system", "content": "You are iFlow CLI, an interactive CLI agent with old instructions"},
                                        {"role": "user", "content": [{"type": "text", "text": "This is the iFlow CLI. We are setting up the context for our chat.\nToday's date is Tuesday, February 24, 2026."}]},
                                        {"role": "assistant", "content": "Got it. Thanks for the context!"},
                                        {"role": "user", "content": "<system-reminder>old reminder</system-reminder>Solve the task"},
                                    ],
                                    "tools": [{"type": "function", "function": {"name": "read_file", "description": "old", "parameters": {"type": "object"}}}],
                                },
                            }
                        ),
                        json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "done"}}]}}),
                    ]
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            _, response = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [
                        {"role": "system", "content": "You are iFlow CLI, an interactive CLI agent with new instructions"},
                        {"role": "user", "content": [{"type": "text", "text": "This is the iFlow CLI. We are setting up the context for our chat.\nToday's date is Thursday, March 19, 2026."}]},
                        {"role": "assistant", "content": "Thanks for the context. Ready when you are."},
                        {"role": "user", "content": "<system-reminder>new reminder</system-reminder>Solve the task"},
                    ],
                    "tools": [{"type": "function", "function": {"name": "read_file", "description": "new", "parameters": {"type": "array"}}}],
                },
            )

        self.assertEqual(response["choices"][0]["message"]["content"], "done")

    def test_trace_replay_state_ignores_tool_schema_drift(self) -> None:
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
                                    "messages": [
                                        {"role": "system", "content": "You are iFlow CLI, an interactive CLI agent with old instructions"},
                                        {"role": "user", "content": [{"type": "text", "text": "This is the iFlow CLI. We are setting up the context for our chat.\nToday's date is Tuesday, February 24, 2026."}]},
                                        {"role": "assistant", "content": "Got it. Thanks for the context!"},
                                        {"role": "user", "content": "<system-reminder>old reminder</system-reminder>Solve the task"},
                                    ],
                                    "tools": [
                                        {"type": "function", "function": {"name": "read_file"}},
                                        {"type": "function", "function": {"name": "write_file"}},
                                        {"type": "function", "function": {"name": "fetch_html"}},
                                    ],
                                },
                            }
                        ),
                        json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "done"}}]}}),
                    ]
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            _, response = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "new-model",
                    "messages": [
                        {"role": "system", "content": "You are iFlow CLI, an interactive CLI agent with new instructions"},
                        {"role": "user", "content": [{"type": "text", "text": "This is the iFlow CLI. We are setting up the context for our chat.\nToday's date is Thursday, March 19, 2026."}]},
                        {"role": "assistant", "content": "Thanks for the context. Ready when you are."},
                        {"role": "user", "content": "<system-reminder>new reminder</system-reminder>Solve the task"},
                    ],
                    "tool_choice": "required",
                    "tools": [
                        {"type": "function", "function": {"name": "read_file"}},
                    ],
                },
            )

        self.assertEqual(response["choices"][0]["message"]["content"], "done")

    def test_trace_replay_state_rejects_unmatched_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text("\n".join(self._request_response_lines("first")), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first_index, first = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "unmatched"}], "tools": []},
            )

            self.assertEqual(first_index, 1)
            self.assertEqual(first["choices"][0]["message"]["content"], "first")
            with self.assertRaisesRegex(ValueError, "no replay response found"):
                state.next_response(
                    path="/v1/chat/completions",
                    headers={"X-Agent-Sandbox-Id": "sbx"},
                    payload={"model": "trace-model", "messages": [{"role": "user", "content": "second"}], "tools": []},
                )

    def test_parse_replay_trace_includes_sidecar_evaluator_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "tool_calls",
                                            "message": {"content": "first", "tool_calls": [{"id": "1"}]},
                                        }
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "stop",
                                            "message": {
                                                "content": "```json\n{\"reasoning\":\"meta\",\"confidence\":0.5}\n```"
                                            },
                                        }
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "stop",
                                            "message": {"content": "real-final"},
                                        }
                                    ]
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            parsed = parse_replay_trace(trace_path)

        self.assertEqual(len(parsed.responses), 3)
        self.assertEqual(parsed.responses[0]["choices"][0]["message"]["content"], "first")
        self.assertEqual(parsed.responses[1]["choices"][0]["message"]["content"], "```json\n{\"reasoning\":\"meta\",\"confidence\":0.5}\n```")
        self.assertEqual(parsed.responses[2]["choices"][0]["message"]["content"], "real-final")

    def test_parse_replay_trace_includes_unfenced_sidecar_evaluator_responses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "tool_calls",
                                            "message": {"content": "first", "tool_calls": [{"id": "1"}]},
                                        }
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "stop",
                                            "message": {
                                                "content": json.dumps(
                                                    {"reasoning": "meta", "confidence": 0.5}
                                                )
                                            },
                                        }
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "stop",
                                            "message": {"content": "real-final"},
                                        }
                                    ]
                                },
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            parsed = parse_replay_trace(trace_path)

        self.assertEqual(len(parsed.responses), 3)
        self.assertEqual(parsed.responses[0]["choices"][0]["message"]["content"], "first")
        self.assertEqual(parsed.responses[1]["choices"][0]["message"]["content"], json.dumps({"reasoning": "meta", "confidence": 0.5}))
        self.assertEqual(parsed.responses[2]["choices"][0]["message"]["content"], "real-final")

    def test_parse_replay_trace_includes_string_escaping_correction_responses(self) -> None:
        for key in ("corrected_string_escaping", "corrected_new_string_escaping"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                trace_path = Path(tmp) / "trace.log"
                trace_path.write_text(
                    "\n".join(
                        [
                            json.dumps(
                                {
                                    "type": "response",
                                    "data": {
                                        "choices": [
                                            {
                                                "finish_reason": "tool_calls",
                                                "message": {"content": "first", "tool_calls": [{"id": "1"}]},
                                            }
                                        ]
                                    },
                                }
                            ),
                            json.dumps(
                                {
                                    "type": "response",
                                    "data": {
                                        "choices": [
                                            {
                                                "finish_reason": "stop",
                                                "message": {"content": f"```json\n{json.dumps({key: 'print(1)'})}\n```"},
                                            }
                                        ]
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
                                                "message": {"content": "second", "tool_calls": [{"id": "2"}]},
                                            }
                                        ]
                                    },
                                }
                            ),
                        ]
                    ),
                    encoding="utf-8",
                )

                parsed = parse_replay_trace(trace_path)

            self.assertEqual(len(parsed.responses), 3)
            self.assertEqual(parsed.responses[0]["choices"][0]["message"]["content"], "first")
            self.assertEqual(parsed.responses[1]["choices"][0]["message"]["content"], f"```json\n{json.dumps({key: 'print(1)'})}\n```")
            self.assertEqual(parsed.responses[2]["choices"][0]["message"]["content"], "second")

    def test_trace_replay_state_counts_stop_responses_as_replay_progress(self) -> None:
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
                                    "messages": [
                                        {"role": "system", "content": "You are iFlow CLI, an interactive CLI agent with old instructions"},
                                        {
                                            "role": "user",
                                            "content": [
                                                {
                                                    "type": "text",
                                                    "text": "Context: An LLM has just generated potentially_problematic_string and the text might have been improperly escaped.",
                                                }
                                            ],
                                        },
                                    ],
                                    "tools": [],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "stop",
                                            "message": {
                                                "content": "```json\n{\"corrected_string_escaping\": \"print(1)\"}\n```"
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
                                    "messages": [{"role": "user", "content": "real"}],
                                    "tools": [],
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {"choices": [{"message": {"content": "real-final"}}]},
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path), "response_delay_ms": 0})

            first_index, first = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={
                    "model": "trace-model",
                    "messages": [
                        {"role": "system", "content": "You are iFlow CLI, an interactive CLI agent with new instructions"},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Context: An LLM has just generated potentially_problematic_string and the text might have been improperly escaped.",
                                }
                            ],
                        },
                    ],
                    "tools": [],
                },
            )
            second_index, second = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "real"}], "tools": []},
            )
            snapshot = state.snapshot()

        self.assertEqual(first_index, 1)
        self.assertEqual(first["choices"][0]["message"]["content"], "```json\n{\"corrected_string_escaping\": \"print(1)\"}\n```")
        self.assertEqual(second_index, 2)
        self.assertEqual(second["choices"][0]["message"]["content"], "real-final")
        self.assertEqual(snapshot["total_responses"], 2)
        self.assertEqual(snapshot["consumed_response_count"], 2)

    def test_generate_dataset_builds_expected_replay_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            output_path = root / "datasets" / "termnius_iflow_replay.jsonl"
            task_root = tasks_root / "hello-world"
            trace_root = traces_root / "hello-world" / "hello-world.1-of-1.2026-02-24__20-20-40"
            (task_root / "tests").mkdir(parents=True, exist_ok=True)
            (trace_root / "agent-logs").mkdir(parents=True, exist_ok=True)
            (task_root / "task.yaml").write_text(
                "\n".join(
                    [
                        "instruction: |-",
                        "  Create /app/hello.txt",
                        "max_agent_timeout_sec: 900.0",
                        "max_test_timeout_sec: 180.0",
                        "run_tests_in_same_shell: false",
                    ]
                ),
                encoding="utf-8",
            )
            (task_root / "docker-compose.yaml").write_text(
                "\n".join(
                    [
                        "services:",
                        "  client:",
                        "    image: example:latest",
                    ]
                ),
                encoding="utf-8",
            )
            (task_root / "run-tests.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (trace_root / "agent-logs" / "proxy_server_trajectory.log").write_text(
                json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "done"}}]}}) + "\n",
                encoding="utf-8",
            )

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["task_id"], "hello-world")
        self.assertEqual(rows[0]["agent_type"], "iflow")
        self.assertEqual(rows[0]["llm_service_type"], "iflow_trace_replay")
        self.assertEqual(rows[0]["service_name"], "client")
        self.assertEqual(rows[0]["trace_response_count"], 1)
        self.assertEqual(rows[0]["trace_malformed_line_count"], 0)
        self.assertEqual(rows[0]["task_description"]["prompt"], "Create /app/hello.txt")

    def test_generate_dataset_preserves_raw_response_count_for_sidecar_stops(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            output_path = root / "datasets" / "termnius_iflow_replay.jsonl"
            task_root = tasks_root / "hello-world"
            trace_root = traces_root / "hello-world" / "hello-world.1-of-1.2026-02-24__20-20-40"
            (task_root / "tests").mkdir(parents=True, exist_ok=True)
            (trace_root / "agent-logs").mkdir(parents=True, exist_ok=True)
            (task_root / "task.yaml").write_text(
                "\n".join(
                    [
                        "instruction: |-",
                        "  Create /app/hello.txt",
                    ]
                ),
                encoding="utf-8",
            )
            (task_root / "docker-compose.yaml").write_text(
                "\n".join(
                    [
                        "services:",
                        "  client:",
                        "    image: example:latest",
                    ]
                ),
                encoding="utf-8",
            )
            (task_root / "run-tests.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (trace_root / "agent-logs" / "proxy_server_trajectory.log").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "tool_calls",
                                            "message": {"content": "first", "tool_calls": [{"id": "1"}]},
                                        }
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "stop",
                                            "message": {
                                                "content": "```json\n{\"reasoning\":\"meta\",\"confidence\":0.5}\n```"
                                            },
                                        }
                                    ]
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "type": "response",
                                "data": {
                                    "choices": [
                                        {
                                            "finish_reason": "stop",
                                            "message": {"content": "real-final"},
                                        }
                                    ]
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
            )

        self.assertEqual(rows[0]["trace_response_count"], 3)

    def test_generate_dataset_counts_malformed_response_lines_in_raw_total(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_root = root / "tasks"
            traces_root = root / "traces"
            output_path = root / "datasets" / "termnius_iflow_replay.jsonl"
            task_root = tasks_root / "hello-world"
            trace_root = traces_root / "hello-world" / "hello-world.1-of-1.2026-02-24__20-20-40"
            (task_root / "tests").mkdir(parents=True, exist_ok=True)
            (trace_root / "agent-logs").mkdir(parents=True, exist_ok=True)
            (task_root / "task.yaml").write_text("instruction: hello\n", encoding="utf-8")
            (task_root / "docker-compose.yaml").write_text(
                "\n".join(
                    [
                        "services:",
                        "  client:",
                        "    image: example:latest",
                    ]
                ),
                encoding="utf-8",
            )
            (task_root / "run-tests.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            (trace_root / "agent-logs" / "proxy_server_trajectory.log").write_text(
                "\n".join(
                    [
                        '{"timestamp": 1, "type": "response", "data": {"choices": [{"message": {"content": "broken"}}]}} trailing',
                        json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "done"}}]}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            rows = generate_dataset(
                tasks_root=tasks_root,
                traces_root=traces_root,
                output_path=output_path,
            )

        self.assertEqual(rows[0]["trace_response_count"], 2)
        self.assertEqual(rows[0]["trace_malformed_line_count"], 1)


class TestHashCollisionRealTraces(unittest.TestCase):
    RESULTS_DIR = Path("/root/workspace/agent-cr/results/2026-02-24__20-20-40_passed")
    ALLOWED_PARSE_ERROR_TRACE_NAMES = {
        "vulnerable-secret.1-of-1.2026-02-24__20-20-40",
    }

    def _find_trace_files(self) -> list[Path]:
        return sorted(self.RESULTS_DIR.glob("**/proxy_server_trajectory.log"))

    def test_no_hash_collisions_in_real_traces(self) -> None:
        trace_files = self._find_trace_files()
        self.assertGreater(len(trace_files), 0, "No trace files found")

        parse_errors: list[tuple[Path, str]] = []
        collisions: list[dict] = []
        total_exchanges = 0

        for trace_path in trace_files:
            try:
                parsed = parse_replay_trace(trace_path)
            except ValueError as exc:
                # parse_replay_trace already detects different-response collisions
                parse_errors.append((trace_path, str(exc)))
                continue

            # Check for same-key, different-request collisions not caught by parse
            seen: dict[str, tuple[int, tuple[object, ...]]] = {}
            for exchange in parsed.exchanges:
                total_exchanges += 1
                key = exchange.lookup_key
                normalized_request = _normalized_lookup_messages(exchange.request)
                if key in seen:
                    prev_idx, prev_request = seen[key]
                    if normalized_request != prev_request:
                        collisions.append({
                            "trace": trace_path.parent.parent.name,
                            "hash_prefix": key[:16],
                            "exchange_a_index": prev_idx,
                            "exchange_b_index": exchange.response_index,
                        })
                else:
                    seen[key] = (exchange.response_index, normalized_request)

        # Print diagnostic summary regardless of outcome
        print(
            f"\nTraces checked: {len(trace_files)}, "
            f"successful: {len(trace_files) - len(parse_errors)}, "
            f"total exchanges: {total_exchanges}"
        )
        for path, err in parse_errors:
            print(f"  PARSE ERROR {path.parent.parent.name}: {err}")
        for c in collisions:
            print(
                f"  COLLISION {c['trace']} hash={c['hash_prefix']} "
                f"exchanges #{c['exchange_a_index']} vs #{c['exchange_b_index']}"
            )

        unexpected_parse_errors = [
            (path, err)
            for path, err in parse_errors
            if path.parent.parent.name not in self.ALLOWED_PARSE_ERROR_TRACE_NAMES
        ]
        self.assertEqual(
            unexpected_parse_errors,
            [],
            f"{len(unexpected_parse_errors)} unexpected traces failed to parse:\n"
            + "\n".join(f"  {p.parent.parent.name}: {e}" for p, e in unexpected_parse_errors),
        )
        self.assertEqual(
            collisions,
            [],
            f"{len(collisions)} hash collisions found (same key, different requests)",
        )


class ResponseDelayPolicyTests(unittest.TestCase):
    @staticmethod
    def _timestamped_request_response_lines(
        *items: tuple[str, float, float],
    ) -> list[str]:
        """Each item is (content, request_timestamp, response_timestamp)."""
        lines: list[str] = []
        for content, req_ts, resp_ts in items:
            lines.append(
                json.dumps(
                    {
                        "timestamp": req_ts,
                        "type": "request",
                        "data": {
                            "model": "trace-model",
                            "messages": [{"role": "user", "content": content}],
                            "tools": [],
                        },
                    }
                )
            )
            lines.append(
                json.dumps(
                    {
                        "timestamp": resp_ts,
                        "type": "response",
                        "data": {"choices": [{"message": {"content": content}}]},
                    }
                )
            )
        return lines

    def test_parse_replay_trace_extracts_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._timestamped_request_response_lines(
                        ("first", 1000.0, 1002.5),
                        ("second", 1003.0, 1005.0),
                    )
                ),
                encoding="utf-8",
            )
            parsed = parse_replay_trace(trace_path)

        self.assertEqual(len(parsed.exchanges), 2)
        self.assertAlmostEqual(parsed.exchanges[0].request_timestamp, 1000.0)
        self.assertAlmostEqual(parsed.exchanges[0].response_timestamp, 1002.5)
        self.assertAlmostEqual(parsed.exchanges[0].trace_delay_ms, 2500.0)
        self.assertAlmostEqual(parsed.exchanges[1].request_timestamp, 1003.0)
        self.assertAlmostEqual(parsed.exchanges[1].response_timestamp, 1005.0)
        self.assertAlmostEqual(parsed.exchanges[1].trace_delay_ms, 2000.0)

    def test_trace_delay_ms_none_when_timestamps_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            lines = [
                json.dumps(
                    {
                        "type": "request",
                        "data": {
                            "model": "trace-model",
                            "messages": [{"role": "user", "content": "no-ts"}],
                            "tools": [],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response",
                        "data": {"choices": [{"message": {"content": "no-ts"}}]},
                    }
                ),
            ]
            trace_path.write_text("\n".join(lines), encoding="utf-8")
            parsed = parse_replay_trace(trace_path)

        self.assertEqual(len(parsed.exchanges), 1)
        self.assertIsNone(parsed.exchanges[0].request_timestamp)
        self.assertIsNone(parsed.exchanges[0].response_timestamp)
        self.assertIsNone(parsed.exchanges[0].trace_delay_ms)

    def test_fixed_policy_uses_constant_delay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._timestamped_request_response_lines(
                        ("hello", 1000.0, 1003.0),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_ms": 0,
                    "response_delay_policy": "fixed",
                }
            )
            snapshot = state.snapshot()

        self.assertEqual(snapshot["response_delay_policy"], "fixed")
        self.assertEqual(snapshot["response_delay_ms"], 0)
        self.assertAlmostEqual(snapshot["response_delay_scaling_factor"], 1.0)

    def test_trace_replay_policy_uses_trace_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._timestamped_request_response_lines(
                        ("hello", 1000.0, 1002.0),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_policy": "trace_replay",
                    "response_delay_scaling_factor": 1.0,
                    "response_delay_ms": 0,
                }
            )
            exchange = state._trace.exchanges[0]
            delay = state._compute_delay_ms(exchange)

        self.assertAlmostEqual(delay, 2000.0)

    def test_trace_replay_policy_applies_trace_delay_in_next_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._timestamped_request_response_lines(
                        ("hello", 1000.0, 1002.0),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_policy": "trace_replay",
                    "response_delay_scaling_factor": 0.5,
                    "response_delay_ms": 0,
                }
            )

            with patch("integrations.llm_services.iflow_trace_replay.service.time.sleep") as sleep:
                _, response = state.next_response(
                    path="/v1/chat/completions",
                    headers={"X-Agent-Sandbox-Id": "sbx"},
                    payload={"model": "trace-model", "messages": [{"role": "user", "content": "hello"}], "tools": []},
                )

        sleep.assert_called_once_with(1.0)
        self.assertEqual(response["choices"][0]["message"]["content"], "hello")

    def test_trace_replay_policy_applies_scaling_factor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._timestamped_request_response_lines(
                        ("hello", 1000.0, 1002.0),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_policy": "trace_replay",
                    "response_delay_scaling_factor": 0.5,
                    "response_delay_ms": 0,
                }
            )
            exchange = state._trace.exchanges[0]
            delay = state._compute_delay_ms(exchange)

        self.assertAlmostEqual(delay, 1000.0)

    def test_trace_replay_policy_falls_back_to_fixed_when_no_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            lines = [
                json.dumps(
                    {
                        "type": "request",
                        "data": {
                            "model": "trace-model",
                            "messages": [{"role": "user", "content": "no-ts"}],
                            "tools": [],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response",
                        "data": {"choices": [{"message": {"content": "no-ts"}}]},
                    }
                ),
            ]
            trace_path.write_text("\n".join(lines), encoding="utf-8")
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_policy": "trace_replay",
                    "response_delay_ms": 100,
                }
            )
            exchange = state._trace.exchanges[0]
            delay = state._compute_delay_ms(exchange)

        self.assertAlmostEqual(delay, 100.0)

    def test_invalid_policy_raises_value_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._timestamped_request_response_lines(
                        ("hello", 1000.0, 1002.0),
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError) as ctx:
                TraceReplayLLMState(
                    llm_service_config={
                        "trace_path": str(trace_path),
                        "response_delay_policy": "invalid_policy",
                    }
                )
            self.assertIn("response_delay_policy", str(ctx.exception))

    def test_default_policy_is_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._timestamped_request_response_lines(
                        ("hello", 1000.0, 1002.0),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_ms": 0,
                }
            )

        self.assertEqual(state._response_delay_policy, "fixed")

    def test_snapshot_includes_policy_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                "\n".join(
                    self._timestamped_request_response_lines(
                        ("hello", 1000.0, 1002.0),
                    )
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_policy": "trace_replay",
                    "response_delay_scaling_factor": 2.0,
                    "response_delay_ms": 0,
                }
            )
            snapshot = state.snapshot()

        self.assertEqual(snapshot["response_delay_policy"], "trace_replay")
        self.assertAlmostEqual(snapshot["response_delay_scaling_factor"], 2.0)


if __name__ == "__main__":
    unittest.main()

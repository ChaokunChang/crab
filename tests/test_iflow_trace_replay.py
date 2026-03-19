from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.generate_termnius_iflow_replay_dataset import generate_dataset
from integrations.llm_services.iflow_trace_replay.service import TraceReplayLLMState, parse_replay_trace


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

    def test_trace_replay_state_checkpoint_restore_and_reset(self) -> None:
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
            checkpoint = state.checkpoint_metadata()
            second_index, second = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "second"}], "tools": []},
            )
            state.restore_from_checkpoint_metadata(checkpoint)
            replayed_index, replayed = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "second"}], "tools": []},
            )
            state.reset()
            reset_index, reset_response = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "first"}], "tools": []},
            )

        self.assertEqual(first_index, 1)
        self.assertEqual(second_index, 2)
        self.assertEqual(replayed_index, 3)
        self.assertEqual(reset_index, 1)
        self.assertEqual(first["choices"][0]["message"]["content"], "first")
        self.assertEqual(second["choices"][0]["message"]["content"], "second")
        self.assertEqual(replayed["choices"][0]["message"]["content"], "second")
        self.assertEqual(reset_response["choices"][0]["message"]["content"], "first")

    def test_trace_replay_state_restore_metadata_is_a_noop(self) -> None:
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
            checkpoint = state.checkpoint_metadata()
            state.restore_from_checkpoint_metadata({**checkpoint, "captures_inflight_llm": True})
            replayed_index, replayed = state.next_response(
                path="/v1/chat/completions",
                headers={"X-Agent-Sandbox-Id": "sbx"},
                payload={"model": "trace-model", "messages": [{"role": "user", "content": "first"}], "tools": []},
            )

        self.assertEqual(first_index, 1)
        self.assertEqual(first["choices"][0]["message"]["content"], "first")
        self.assertEqual(replayed_index, 2)
        self.assertEqual(replayed["choices"][0]["message"]["content"], "first")

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
        self.assertEqual(snapshot["matched_response_count"], 2)

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


if __name__ == "__main__":
    unittest.main()

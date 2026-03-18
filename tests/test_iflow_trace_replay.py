from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.generate_termnius_iflow_replay_dataset import generate_dataset
from integrations.llm_services.iflow_trace_replay.service import TraceReplayLLMState, parse_replay_trace


class IFlowTraceReplayTests(unittest.TestCase):
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
                "\n".join(
                    [
                        json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "first"}}]}}),
                        json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "second"}}]}}),
                    ]
                ),
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first_index, first = state.next_response(headers={"X-Agent-Sandbox-Id": "sbx"}, payload={})
            checkpoint = state.checkpoint_metadata()
            second_index, second = state.next_response(headers={"X-Agent-Sandbox-Id": "sbx"}, payload={})
            state.restore_from_checkpoint_metadata(checkpoint)
            replayed_index, replayed = state.next_response(headers={"X-Agent-Sandbox-Id": "sbx"}, payload={})
            state.reset()
            reset_index, reset_response = state.next_response(headers={"X-Agent-Sandbox-Id": "sbx"}, payload={})

        self.assertEqual(first_index, 0)
        self.assertEqual(second_index, 1)
        self.assertEqual(replayed_index, 1)
        self.assertEqual(reset_index, 0)
        self.assertEqual(first["choices"][0]["message"]["content"], "first")
        self.assertEqual(second["choices"][0]["message"]["content"], "second")
        self.assertEqual(replayed["choices"][0]["message"]["content"], "second")
        self.assertEqual(reset_response["choices"][0]["message"]["content"], "first")

    def test_trace_replay_state_applies_default_response_delay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.log"
            trace_path.write_text(
                json.dumps({"type": "response", "data": {"choices": [{"message": {"content": "first"}}]}}) + "\n",
                encoding="utf-8",
            )
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            with patch("integrations.llm_services.iflow_trace_replay.service.time.sleep") as sleep:
                _, response = state.next_response(headers={"X-Agent-Sandbox-Id": "sbx"}, payload={})

        sleep.assert_called_once_with(0.25)
        self.assertEqual(response["choices"][0]["message"]["content"], "first")

    def test_parse_replay_trace_skips_sidecar_evaluator_responses(self) -> None:
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

        self.assertEqual(len(parsed.responses), 2)
        self.assertEqual(parsed.responses[0]["choices"][0]["message"]["content"], "first")
        self.assertEqual(parsed.responses[1]["choices"][0]["message"]["content"], "real-final")

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


if __name__ == "__main__":
    unittest.main()

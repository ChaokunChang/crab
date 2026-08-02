from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations.llm_services.claude_code_trace_replay.service import (
    TraceReplayLLMState,
    _normalize_runtime_specific_bash_command,
    is_strictly_replayable_trace,
    parse_replay_trace,
    strict_replayability_errors,
)


def _base_trace_payload() -> dict[str, object]:
    return {
        "schema_version": "ATIF-v1.2",
        "session_id": "session-1",
        "agent": {
            "name": "claude-code",
            "version": "2.1.34",
            "model_name": "claude-opus-4-6",
            "extra": {},
        },
        "steps": [],
        "final_metrics": {"total_steps": 0, "extra": {}},
    }


def _write_results_trace(
    root: Path,
    *,
    payload: dict[str, object],
    trial_id: str = "trial-1",
    task_dir_name: str = "task__fixture",
) -> tuple[Path, Path]:
    agent_root = root / f"{trial_id}-results" / task_dir_name / "agent"
    agent_root.mkdir(parents=True, exist_ok=True)
    trace_path = agent_root / "trajectory.json"
    trace_path.write_text(json.dumps(payload), encoding="utf-8")
    return trace_path, agent_root


def _write_subagent_session(
    agent_root: Path,
    *,
    prompt: str,
    agent_id: str = "a1",
    session_id: str = "session-1",
) -> Path:
    subagent_path = agent_root / "sessions" / "projects" / "-app" / session_id / "subagents" / f"agent-{agent_id}.jsonl"
    subagent_path.parent.mkdir(parents=True, exist_ok=True)
    entries = [
        {
            "parentUuid": None,
            "isSidechain": True,
            "userType": "external",
            "cwd": "/app",
            "sessionId": session_id,
            "version": "2.1.34",
            "gitBranch": "HEAD",
            "agentId": agent_id,
            "type": "user",
            "message": {"role": "user", "content": prompt},
            "uuid": "user-1",
            "timestamp": "2026-02-07T00:00:00Z",
        },
        {
            "parentUuid": "p-1",
            "isSidechain": True,
            "userType": "external",
            "cwd": "/app",
            "sessionId": session_id,
            "version": "2.1.34",
            "gitBranch": "HEAD",
            "agentId": agent_id,
            "type": "assistant",
            "message": {
                "model": "claude-haiku-4-5-20251001",
                "id": "msg_side_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Inspecting the workspace"}],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {},
            },
            "uuid": "assistant-1",
            "timestamp": "2026-02-07T00:00:01Z",
        },
        {
            "parentUuid": "p-2",
            "isSidechain": True,
            "userType": "external",
            "cwd": "/app",
            "sessionId": session_id,
            "version": "2.1.34",
            "gitBranch": "HEAD",
            "agentId": agent_id,
            "type": "assistant",
            "message": {
                "model": "claude-haiku-4-5-20251001",
                "id": "msg_side_1",
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_sub_1",
                        "name": "Write",
                        "input": {"file_path": "/app/attack.py", "content": "print('hi')\n"},
                    }
                ],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {},
            },
            "uuid": "assistant-2",
            "timestamp": "2026-02-07T00:00:02Z",
        },
        {
            "parentUuid": "p-3",
            "isSidechain": True,
            "userType": "external",
            "cwd": "/app",
            "sessionId": session_id,
            "version": "2.1.34",
            "gitBranch": "HEAD",
            "agentId": agent_id,
            "type": "user",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_sub_1", "content": "ok", "is_error": False}],
            },
            "uuid": "user-2",
            "timestamp": "2026-02-07T00:00:03Z",
        },
        {
            "parentUuid": "p-4",
            "isSidechain": True,
            "userType": "external",
            "cwd": "/app",
            "sessionId": session_id,
            "version": "2.1.34",
            "gitBranch": "HEAD",
            "agentId": agent_id,
            "type": "assistant",
            "message": {
                "model": "claude-haiku-4-5-20251001",
                "id": "msg_side_2",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Finished helper work"}],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {},
            },
            "uuid": "assistant-3",
            "timestamp": "2026-02-07T00:00:04Z",
        },
    ]
    subagent_path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    return subagent_path


class ClaudeCodeTraceReplayTests(unittest.TestCase):
    def test_restore_duplicate_catchup_replays_original_tool_without_advancing_cursor(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Inspect files",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Read toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": "Read",
                        "arguments": {"file_path": "/app/input.txt"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "hello"}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Install dependencies",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_2",
                "timestamp": "2026-02-07T00:00:04Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_2",
                        "function_name": "Bash",
                        "arguments": {"command": "pip install tokenizers"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_2", "content": "ok"}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Write the answer",
                "timestamp": "2026-02-07T00:00:05Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Write toolu_3",
                "timestamp": "2026-02-07T00:00:06Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_3",
                        "function_name": "Write",
                        "arguments": {"file_path": "/app/answer.txt", "content": "79586\n"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_3", "content": "ok"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            state.restore(consumed_response_count=2)

            duplicate_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"messages": [{"role": "assistant", "content": "duplicate"}]},
            )
            duplicate_blocks = duplicate_response["content"]
            self.assertEqual(duplicate_blocks[0]["type"], "text")
            self.assertEqual(duplicate_blocks[0]["text"], "Install dependencies")
            self.assertEqual(duplicate_blocks[1]["type"], "tool_use")
            self.assertEqual(duplicate_blocks[1]["name"], "Bash")
            self.assertIn("pip install tokenizers", duplicate_blocks[1]["input"]["command"])
            self.assertEqual(state.snapshot()["trace_cursor"], 2)
            self.assertEqual(state.snapshot()["duplicate_response_count"], 1)

            actual_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {"role": "assistant", "content": "first"},
                        {"role": "assistant", "content": "second"},
                    ]
                },
            )
            actual_blocks = actual_response["content"]
            self.assertEqual(actual_blocks[0]["type"], "text")
            self.assertEqual(actual_blocks[0]["text"], "Write the answer")
            self.assertEqual(actual_blocks[1]["type"], "tool_use")
            self.assertEqual(actual_blocks[1]["name"], "Bash")
            self.assertIn("/app/answer.txt", actual_blocks[1]["input"]["command"])
            self.assertEqual(state.snapshot()["trace_cursor"], 3)

    def test_restore_duplicate_catchup_edit_is_idempotent_when_change_already_applied(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Patch the file",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Edit toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": "Edit",
                        "arguments": {
                            "file_path": "/app/file.txt",
                            "old_string": "old-value",
                            "new_string": "new-value",
                            "replace_all": False,
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "ok"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            target_path = Path(tmp) / "file.txt"
            target_path.write_text("before new-value after", encoding="utf-8")
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            state.restore(consumed_response_count=1)

            duplicate_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"messages": []},
            )
            duplicate_blocks = duplicate_response["content"]
            self.assertEqual(duplicate_blocks[1]["type"], "tool_use")
            self.assertEqual(duplicate_blocks[1]["name"], "Bash")
            subprocess.run(
                ["bash", "-lc", duplicate_blocks[1]["input"]["command"].replace("/app/file.txt", str(target_path))],
                check=True,
                cwd=tmp,
            )

            self.assertEqual(target_path.read_text(encoding="utf-8"), "before new-value after")

    def test_restore_duplicate_catchup_preserves_read_target_via_bash_shim(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Inspect files",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Read toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": "Read",
                        "arguments": {"file_path": "/app/input.txt"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "hello"}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Done",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            state.restore(consumed_response_count=1)

            duplicate_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"messages": []},
            )
            duplicate_blocks = duplicate_response["content"]
            self.assertEqual(duplicate_blocks[0]["type"], "text")
            self.assertEqual(duplicate_blocks[0]["text"], "Inspect files")
            self.assertEqual(duplicate_blocks[1]["type"], "tool_use")
            self.assertEqual(duplicate_blocks[1]["name"], "Bash")
            self.assertIn("/app/input.txt", duplicate_blocks[1]["input"]["command"])

    def test_restore_preserves_sticky_filename_aliases_across_requests(self) -> None:
        payload = _base_trace_payload()
        recorded_listing = (
            "total 8\n"
            "-rw-r--r-- 1 root root 123 Jan  1 00:00 old-a.jpg\n"
            "-rw-r--r-- 1 root root 456 Jan  1 00:00 old-b.pdf\n"
        )
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Inspect files",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_ls",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_ls",
                        "function_name": "Bash",
                        "arguments": {"command": "ls -la /app/documents"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_ls", "content": recorded_listing}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Move the renamed files",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_mv",
                        "function_name": "Bash",
                        "arguments": {
                            "command": (
                                "mv /app/documents/old-a.jpg /app/invoices/ && "
                                "mv /app/documents/old-b.pdf /app/invoices/"
                            )
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_mv", "content": "ok"}]},
                "extra": {},
            },
        ]

        live_listing = (
            "total 8\n"
            "-rw-r--r-- 1 root root 123 Jan  1 00:00 live-a.jpg\n"
            "-rw-r--r-- 1 root root 456 Jan  1 00:00 live-b.pdf\n"
        )

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_ls",
                                    "content": live_listing,
                                }
                            ],
                        }
                    ]
                },
            )
            self.assertIn("Inspect files", first_response["content"][0]["text"])

            state.restore(consumed_response_count=1)

            second_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {"role": "assistant", "content": "prior assistant turn"},
                    ]
                },
            )

            bash_block = next(block for block in second_response["content"] if block.get("type") == "tool_use")
            self.assertEqual(bash_block["type"], "tool_use")
            self.assertEqual(bash_block["name"], "Bash")
            self.assertIn("live-a.jpg", bash_block["input"]["command"])
            self.assertIn("live-b.pdf", bash_block["input"]["command"])
            self.assertNotIn("old-a.jpg", bash_block["input"]["command"])
            self.assertNotIn("old-b.pdf", bash_block["input"]["command"])

    def test_normalize_runtime_specific_bash_command_wraps_apt_commands(self) -> None:
        command = "apt-get update -qq && apt-get install -y build-essential 2>&1 | tail -5"

        normalized = _normalize_runtime_specific_bash_command(command)

        self.assertIn("crab_retry_apt_command", normalized)
        self.assertIn("wait_for_apt_lock()", normalized)
        self.assertIn(command, normalized)

    def test_normalize_runtime_specific_bash_command_ensures_usr_local_bin_for_cp(self) -> None:
        command = "cp /app/povray-2.2/source/povray /usr/local/bin/povray && chmod +x /usr/local/bin/povray"

        normalized = _normalize_runtime_specific_bash_command(command)

        self.assertTrue(normalized.startswith("install -d -m 755 /usr/local/bin && "))
        self.assertIn(command, normalized)

    def test_parse_replay_trace_merges_atif_steps_and_drops_placeholder_only_text(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Plan first",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed TodoWrite toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {"tool_call_id": "toolu_1", "function_name": "TodoWrite", "arguments": {"todos": []}}
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "ok"}]},
                "extra": {"tool_result_metadata": {"tool_use_result": {"oldTodos": [], "newTodos": []}}},
            },
            {
                "source": "agent",
                "message": "Need to read file",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Read toolu_2",
                "timestamp": "2026-02-07T00:00:04Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_2",
                        "function_name": "Read",
                        "arguments": {"file_path": "/app/foo.txt", "offset": 10, "limit": 5},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_2", "content": "file contents"}]},
                "extra": {
                    "tool_result_metadata": {
                        "tool_use_result": {
                            "type": "text",
                            "file": {
                                "filePath": "/app/foo.txt",
                                "content": "hello",
                                "startLine": 10,
                                "numLines": 5,
                            },
                        }
                    }
                },
            },
            {
                "source": "agent",
                "message": "$4c",
                "timestamp": "2026-02-07T00:00:05Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "All done",
                "timestamp": "2026-02-07T00:00:08Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")

            parsed = parse_replay_trace(trace_path)

        self.assertEqual(parsed.model_name, "claude-opus-4-6")
        self.assertEqual(len(parsed.agent_steps), 2)
        self.assertEqual(parsed.agent_step_delays_ms, (None, 4000.0))
        first_turn = parsed.agent_steps[0]
        self.assertEqual(first_turn["text_segments"], ["Plan first", "Need to read file"])
        self.assertEqual(len(first_turn["tools"]), 1)
        self.assertEqual(first_turn["tools"][0]["function_name"], "Read")
        self.assertEqual(parsed.agent_steps[1]["text_segments"], ["All done"])

    def test_parse_replay_trace_keeps_consecutive_tool_only_steps_separate(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Plan first",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {"tool_call_id": "toolu_1", "function_name": "Bash", "arguments": {"command": "echo one"}}
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "one"}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_2",
                "timestamp": "2026-02-07T00:00:05Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {"tool_call_id": "toolu_2", "function_name": "Bash", "arguments": {"command": "echo two"}}
                ],
                "observation": {"results": [{"source_call_id": "toolu_2", "content": "two"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")

            parsed = parse_replay_trace(trace_path)

        self.assertEqual(len(parsed.agent_steps), 2)
        self.assertEqual(parsed.agent_step_delays_ms, (None, 3000.0))
        self.assertEqual(parsed.agent_steps[0]["text_segments"], ["Plan first"])
        self.assertEqual([tool["function_name"] for tool in parsed.agent_steps[0]["tools"]], ["Bash"])
        self.assertEqual(parsed.agent_steps[1]["text_segments"], [])
        self.assertEqual([tool["function_name"] for tool in parsed.agent_steps[1]["tools"]], ["Bash"])

    def test_handle_request_preserves_atif_tool_arguments(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Let's patch the file",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Edit toolu_3",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_3",
                        "function_name": "Edit",
                        "arguments": {
                            "file_path": "/app/file.txt",
                            "old_string": "$aa",
                            "new_string": "$ab",
                            "replace_all": False,
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_3", "content": "$39"}]},
                "extra": {
                    "tool_result_metadata": {
                        "tool_use_result": {
                            "filePath": "/app/file.txt",
                            "oldString": "old-value",
                            "newString": "new-value",
                        }
                    }
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            target_path = Path(tmp) / "file.txt"
            target_path.write_text("before old-value after", encoding="utf-8")
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            response = state.handle_request(path="/v1/messages", headers={}, payload={})
            blocks = response["content"]
            self.assertEqual(blocks[0]["type"], "text")
            self.assertEqual(blocks[1]["type"], "tool_use")
            self.assertEqual(blocks[1]["name"], "Bash")
            subprocess.run(
                ["bash", "-lc", blocks[1]["input"]["command"].replace("/app/file.txt", str(target_path))],
                check=True,
                cwd=tmp,
            )

            self.assertEqual(target_path.read_text(encoding="utf-8"), "before new-value after")
            self.assertFalse(blocks[1]["input"]["run_in_background"])

    def test_handle_request_normalizes_mail_service_commands_for_runtime(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Start Postfix",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_postfix",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_postfix",
                        "function_name": "Bash",
                        "arguments": {"command": "postfix start 2>&1 || postfix reload 2>&1"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_postfix", "content": "ok"}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Start Mailman",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_mailman",
                "timestamp": "2026-02-07T00:00:04Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_mailman",
                        "function_name": "Bash",
                        "arguments": {"command": "mailman --run-as-root start 2>&1"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_mailman", "content": "ok"}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Reload Postfix after generating maps",
                "timestamp": "2026-02-07T00:00:05Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_reload",
                "timestamp": "2026-02-07T00:00:06Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_reload",
                        "function_name": "Bash",
                        "arguments": {
                            "command": (
                                "postmap /var/lib/mailman3/data/postfix_lmtp && "
                                "postmap /var/lib/mailman3/data/postfix_domains && postfix reload 2>&1"
                            )
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_reload", "content": "ok"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            response1 = state.handle_request(path="/v1/messages", headers={}, payload={})
            postfix_command = response1["content"][1]["input"]["command"]
            self.assertIn("service postfix start", postfix_command)
            self.assertIn("service postfix restart", postfix_command)

            response2 = state.handle_request(path="/v1/messages", headers={}, payload={})
            mailman_command = response2["content"][1]["input"]["command"]
            self.assertIn("/usr/lib/mailman3/bin/mailman -C /etc/mailman3/mailman.cfg --run-as-root start", mailman_command)

            response3 = state.handle_request(path="/v1/messages", headers={}, payload={})
            reload_command = response3["content"][1]["input"]["command"]
            self.assertIn("service postfix reload", reload_command)
            self.assertIn("service postfix start", reload_command)

    def test_handle_request_strips_mailman_relay_domains_from_postfix_main_cf_edits(self) -> None:
        old_main_cf = "\n".join(
            [
                "myhostname = localhost.local",
                "alias_maps = hash:/etc/aliases",
                "alias_database = hash:/etc/aliases",
                "mydestination = $myhostname, localhost, localhost.localdomain, , localhost",
                "relayhost = ",
                "mynetworks = 127.0.0.0/8 [::ffff:127.0.0.0]/104 [::1]/128",
                "mailbox_size_limit = 0",
                "recipient_delimiter = +",
                "inet_interfaces = all",
                "inet_protocols = all",
            ]
        )
        new_main_cf = "\n".join(
            [
                "myhostname = local.edu",
                "mydomain = local.edu",
                "alias_maps = hash:/etc/aliases",
                "alias_database = hash:/etc/aliases",
                "mydestination = $myhostname, localhost, localhost.localdomain, local.edu",
                "relayhost =",
                "mynetworks = 127.0.0.0/8 [::ffff:127.0.0.0]/104 [::1]/128",
                "mailbox_size_limit = 0",
                "recipient_delimiter = +",
                "inet_interfaces = all",
                "inet_protocols = all",
                "",
                "# Mailman3 integration",
                "transport_maps = hash:/var/lib/mailman3/data/postfix_lmtp",
                "local_recipient_maps = proxy:unix:passwd.byname $alias_maps hash:/var/lib/mailman3/data/postfix_lmtp",
                "relay_domains = hash:/var/lib/mailman3/data/postfix_domains",
            ]
        )
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Patch postfix main.cf",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Edit toolu_postfix_edit",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_postfix_edit",
                        "function_name": "Edit",
                        "arguments": {
                            "file_path": "/etc/postfix/main.cf",
                            "old_string": old_main_cf,
                            "new_string": new_main_cf,
                            "replace_all": False,
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_postfix_edit", "content": "ok"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            target_path = Path(tmp) / "main.cf"
            target_path.write_text(old_main_cf, encoding="utf-8")
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            response = state.handle_request(path="/v1/messages", headers={}, payload={})
            command = response["content"][1]["input"]["command"].replace("/etc/postfix/main.cf", str(target_path))
            subprocess.run(["bash", "-lc", command], check=True, cwd=tmp)

            updated = target_path.read_text(encoding="utf-8")
            self.assertIn("transport_maps = hash:/var/lib/mailman3/data/postfix_lmtp", updated)
            self.assertIn(
                "local_recipient_maps = proxy:unix:passwd.byname $alias_maps hash:/var/lib/mailman3/data/postfix_lmtp",
                updated,
            )
            self.assertNotIn("relay_domains = hash:/var/lib/mailman3/data/postfix_domains", updated)

    def test_handle_request_remaps_filenames_from_live_directory_listing(self) -> None:
        recorded_listing = "\n".join(
            [
                "total 8",
                "drwxr-xr-x 2 root root   4096 Oct 28 22:06 .",
                "drwxr-xr-x 3 root root   4096 Oct 28 22:06 ..",
                "-rw-r--r-- 1 root root 770021 Oct 28 22:06 old-invoice.jpg",
                "-rw-r--r-- 1 root root   2046 Oct 28 22:06 old-invoice.pdf",
            ]
        )
        live_listing = "\n".join(
            [
                "total 8",
                "drwxr-xr-x 2 root root   4096 Mar 28 16:19 .",
                "drwxr-xr-x 3 root root   4096 Mar 28 16:19 ..",
                "-rw-r--r-- 1 root root 770021 Mar 28 16:19 live-alpha.jpg",
                "-rw-r--r-- 1 root root   2046 Mar 28 16:19 live-beta.pdf",
            ]
        )
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "List the documents first",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_ls",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_ls",
                        "function_name": "Bash",
                        "arguments": {"command": "ls -la /app/documents/"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_ls", "content": recorded_listing}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Read the invoice body",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Read toolu_read",
                "timestamp": "2026-02-07T00:00:04Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_read",
                        "function_name": "Read",
                        "arguments": {"file_path": "/app/documents/old-invoice.jpg"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_read", "content": "contents"}]},
                "extra": {
                    "tool_result_metadata": {
                        "tool_use_result": {
                            "type": "text",
                            "file": {
                                "filePath": "/app/documents/old-invoice.jpg",
                                "content": "contents",
                                "startLine": 1,
                                "numLines": 1,
                            },
                        }
                    }
                },
            },
            {
                "source": "agent",
                "message": "Move files into place",
                "timestamp": "2026-02-07T00:00:05Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_move",
                "timestamp": "2026-02-07T00:00:06Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_move",
                        "function_name": "Bash",
                        "arguments": {
                            "command": (
                                "mv /app/documents/old-invoice.jpg /app/invoices/ && "
                                "mv /app/documents/old-invoice.pdf /app/invoices/"
                            )
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_move", "content": "ok"}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Write the CSV summary",
                "timestamp": "2026-02-07T00:00:07Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Write toolu_write",
                "timestamp": "2026-02-07T00:00:08Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_write",
                        "function_name": "Write",
                        "arguments": {
                            "file_path": "/app/invoices/summary.csv",
                            "content": (
                                "filename,total_amount,vat_amount\n"
                                "old-invoice.jpg,100,0\n"
                                "old-invoice.pdf,200,0\n"
                            ),
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_write", "content": "ok"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            summary_path = Path(tmp) / "summary.csv"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first = state.handle_request(path="/v1/messages", headers={}, payload={})
            second = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {"role": "assistant", "content": first["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_ls",
                                    "content": live_listing,
                                    "is_error": False,
                                }
                            ],
                        },
                    ]
                },
            )
            third = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {"role": "assistant", "content": first["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_ls",
                                    "content": live_listing,
                                    "is_error": False,
                                }
                            ],
                        },
                        {"role": "assistant", "content": second["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_read",
                                    "content": "contents",
                                    "is_error": False,
                                }
                            ],
                        },
                    ]
                },
            )
            fourth = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {"role": "assistant", "content": first["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_ls",
                                    "content": live_listing,
                                    "is_error": False,
                                }
                            ],
                        },
                        {"role": "assistant", "content": second["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_read",
                                    "content": "contents",
                                    "is_error": False,
                                }
                            ],
                        },
                        {"role": "assistant", "content": third["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_move",
                                    "content": "ok",
                                    "is_error": False,
                                }
                            ],
                        },
                    ]
                },
            )

            self.assertIn("/app/documents/live-alpha.jpg", second["content"][1]["input"]["command"])
            self.assertNotIn("old-invoice.jpg", second["content"][1]["input"]["command"])
            self.assertIn("/app/documents/live-alpha.jpg", third["content"][1]["input"]["command"])
            self.assertIn("/app/documents/live-beta.pdf", third["content"][1]["input"]["command"])

            write_command = fourth["content"][1]["input"]["command"].replace("/app/invoices/summary.csv", str(summary_path))
            subprocess.run(["bash", "-lc", write_command], check=True, cwd=tmp)
            self.assertEqual(
                summary_path.read_text(encoding="utf-8"),
                "filename,total_amount,vat_amount\nlive-alpha.jpg,100,0\nlive-beta.pdf,200,0\n",
            )

    def test_count_tokens_request_does_not_consume_first_replay_turn(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Read the file first",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Read toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": "Read",
                        "arguments": {"file_path": "/app/file.txt"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "hello"}]},
                "extra": {
                    "tool_result_metadata": {
                        "tool_use_result": {
                            "type": "text",
                            "file": {"filePath": "/app/file.txt", "content": "hello", "startLine": 1, "numLines": 1},
                        }
                    }
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            token_response = state.handle_request(
                path="/v1/messages/count_tokens",
                headers={},
                payload={"messages": [{"role": "user", "content": "task"}], "tools": [{"name": "Read"}]},
            )
            snapshot_after_token = state.snapshot()
            replay_response = state.handle_request(path="/v1/messages", headers={}, payload={})

        self.assertGreaterEqual(token_response["input_tokens"], 1)
        self.assertEqual(snapshot_after_token["trace_cursor"], 0)
        self.assertEqual(snapshot_after_token["main_loop_request_count"], 0)
        self.assertEqual(snapshot_after_token["helper_request_count"], 0)
        self.assertEqual(snapshot_after_token["count_tokens_request_count"], 1)
        self.assertEqual(state.snapshot()["trace_cursor"], 1)
        self.assertEqual(replay_response["content"][0]["type"], "text")
        self.assertEqual(replay_response["content"][0]["text"], "Read the file first")
        self.assertEqual(replay_response["content"][1]["type"], "tool_use")
        self.assertEqual(replay_response["content"][1]["name"], "Bash")

    def test_helper_model_request_does_not_consume_replay_turn(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Read the file first",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Read toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": "Read",
                        "arguments": {"file_path": "/app/file.txt"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "hello"}]},
                "extra": {
                    "tool_result_metadata": {
                        "tool_use_result": {
                            "type": "text",
                            "file": {"filePath": "/app/file.txt", "content": "hello", "startLine": 1, "numLines": 1},
                        }
                    }
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            helper_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "helper"}]},
            )
            snapshot_after_helper = state.snapshot()
            replay_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"model": "claude-opus-4-6", "messages": [{"role": "user", "content": "task"}]},
            )

        self.assertEqual(helper_response["content"], [{"type": "text", "text": ""}])
        self.assertEqual(snapshot_after_helper["trace_cursor"], 0)
        self.assertEqual(snapshot_after_helper["main_loop_request_count"], 0)
        self.assertEqual(snapshot_after_helper["helper_request_count"], 1)
        self.assertEqual(snapshot_after_helper["count_tokens_request_count"], 0)
        self.assertEqual(state.snapshot()["trace_cursor"], 1)
        self.assertEqual(replay_response["content"][0]["type"], "text")
        self.assertEqual(replay_response["content"][0]["text"], "Read the file first")

    def test_auxiliary_requests_do_not_advance_main_loop_trace_cursor(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "First",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": "Bash",
                        "arguments": {"command": "echo first", "description": "first"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "first"}]},
                "extra": {"tool_result_metadata": {"tool_use_result": {"stdout": "first"}}},
            },
            {
                "source": "agent",
                "message": "Second",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_2",
                "timestamp": "2026-02-07T00:00:04Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_2",
                        "function_name": "Bash",
                        "arguments": {"command": "echo second", "description": "second"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_2", "content": "second"}]},
                "extra": {"tool_result_metadata": {"tool_use_result": {"stdout": "second"}}},
            },
            {
                "source": "agent",
                "message": "Third",
                "timestamp": "2026-02-07T00:00:05Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_3",
                "timestamp": "2026-02-07T00:00:06Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_3",
                        "function_name": "Bash",
                        "arguments": {"command": "echo third", "description": "third"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_3", "content": "third"}]},
                "extra": {"tool_result_metadata": {"tool_use_result": {"stdout": "third"}}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first = state.handle_request(path="/v1/messages", headers={}, payload={"model": "claude-opus-4-6"})
            snapshot_after_first = state.snapshot()
            helper_one = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "helper-1"}]},
            )
            helper_two = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "helper-2"}]},
            )
            snapshot_after_helpers = state.snapshot()
            second = state.handle_request(path="/v1/messages", headers={}, payload={"model": "claude-opus-4-6"})
            token_response = state.handle_request(
                path="/v1/messages/count_tokens",
                headers={},
                payload={"messages": [{"role": "user", "content": "task"}]},
            )
            snapshot_after_count_tokens = state.snapshot()
            third = state.handle_request(path="/v1/messages", headers={}, payload={"model": "claude-opus-4-6"})

        self.assertEqual(first["content"][0]["text"], "First")
        self.assertEqual(helper_one["content"], [{"type": "text", "text": ""}])
        self.assertEqual(helper_two["content"], [{"type": "text", "text": ""}])
        self.assertGreaterEqual(token_response["input_tokens"], 1)
        self.assertEqual(second["content"][0]["text"], "Second")
        self.assertEqual(third["content"][0]["text"], "Third")
        self.assertEqual(snapshot_after_first["trace_cursor"], 1)
        self.assertEqual(snapshot_after_helpers["trace_cursor"], 1)
        self.assertEqual(snapshot_after_count_tokens["trace_cursor"], 2)
        self.assertEqual(state.snapshot()["trace_cursor"], 3)
        self.assertEqual(state.snapshot()["main_loop_request_count"], 3)
        self.assertEqual(state.snapshot()["helper_request_count"], 2)
        self.assertEqual(state.snapshot()["count_tokens_request_count"], 1)

    def test_restore_uses_main_loop_turns_even_after_auxiliary_requests(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "First",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": "Bash",
                        "arguments": {"command": "echo first", "description": "first"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "first"}]},
                "extra": {"tool_result_metadata": {"tool_use_result": {"stdout": "first"}}},
            },
            {
                "source": "agent",
                "message": "Second",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_2",
                "timestamp": "2026-02-07T00:00:04Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_2",
                        "function_name": "Bash",
                        "arguments": {"command": "echo second", "description": "second"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_2", "content": "second"}]},
                "extra": {"tool_result_metadata": {"tool_use_result": {"stdout": "second"}}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first = state.handle_request(path="/v1/messages", headers={}, payload={"model": "claude-opus-4-6"})
            state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "helper"}]},
            )
            state.handle_request(
                path="/v1/messages/count_tokens",
                headers={},
                payload={"messages": [{"role": "user", "content": "task"}]},
            )
            state.restore(consumed_response_count=1)
            restored_snapshot = state.snapshot()
            second = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "model": "claude-opus-4-6",
                    "messages": [
                        {"role": "assistant", "content": first["content"]},
                        {"role": "user", "content": "continue"},
                    ],
                },
            )

        self.assertEqual(restored_snapshot["trace_cursor"], 1)
        self.assertEqual(restored_snapshot["main_loop_request_count"], 1)
        self.assertEqual(restored_snapshot["helper_request_count"], 1)
        self.assertEqual(restored_snapshot["count_tokens_request_count"], 1)
        self.assertEqual(second["content"][0]["text"], "Second")
        self.assertEqual(state.snapshot()["trace_cursor"], 2)

    def test_helper_model_request_replays_task_sidechain_from_results_bundle(self) -> None:
        prompt = "Create /app/attack.py"
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Delegate the helper work",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Task toolu_task_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_task_1",
                        "function_name": "Task",
                        "arguments": {
                            "description": "Build helper files",
                            "subagent_type": "general-purpose",
                            "prompt": prompt,
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_task_1", "content": "subagent finished"}]},
                "extra": {"tool_result_metadata": {"tool_use_result": {"agentId": "a1"}}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_path, agent_root = _write_results_trace(root, payload=payload)
            _write_subagent_session(agent_root, prompt=prompt, agent_id="a1")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            main_response = state.handle_request(path="/v1/messages", headers={}, payload={})
            helper_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": prompt}]},
            )
            helper_follow_up = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "model": "claude-haiku-4-5-20251001",
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": helper_response["content"]},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_sub_1",
                                    "content": "ok",
                                    "is_error": False,
                                }
                            ],
                        },
                    ],
                },
            )

        self.assertEqual(main_response["content"][1]["type"], "tool_use")
        self.assertEqual(main_response["content"][1]["name"], "Task")
        self.assertEqual(helper_response["content"][0]["type"], "text")
        self.assertEqual(helper_response["content"][0]["text"], "Inspecting the workspace")
        self.assertEqual(helper_response["content"][1]["type"], "tool_use")
        self.assertEqual(helper_response["content"][1]["name"], "Bash")
        self.assertEqual(helper_response["content"][1]["id"], "toolu_sub_1")
        self.assertIn("/app/attack.py", helper_response["content"][1]["input"]["command"])
        self.assertEqual(helper_follow_up["content"], [{"type": "text", "text": "Finished helper work"}])

    def test_helper_model_request_falls_back_when_task_sidechain_is_missing(self) -> None:
        prompt = "Analyze FEAL cipher structure"
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Executed Task toolu_task_missing",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_task_missing",
                        "function_name": "Task",
                        "arguments": {
                            "description": "Analyze the cipher",
                            "subagent_type": "Explore",
                            "prompt": prompt,
                        },
                    }
                ],
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace_path, _ = _write_results_trace(root, payload=payload)
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            helper_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": prompt}]},
            )

        self.assertEqual(helper_response["content"], [{"type": "text", "text": ""}])

    def test_handle_request_replays_task_output_and_task_stop_with_live_background_id(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Start the background job",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_bg_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_bg_1",
                        "function_name": "Bash",
                        "arguments": {"command": "sleep 10", "run_in_background": True},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "toolu_bg_1",
                            "content": (
                                "Command running in background with ID: rec123. "
                                "Output is being written to: /tmp/rec123.output\n\n"
                                "[metadata] {\"backgroundTaskId\": \"rec123\"}"
                            ),
                        }
                    ]
                },
                "extra": {"tool_result_metadata": {"tool_use_result": {"backgroundTaskId": "rec123"}}},
            },
            {
                "source": "agent",
                "message": "Executed TaskOutput toolu_poll_1",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_poll_1",
                        "function_name": "TaskOutput",
                        "arguments": {"task_id": "rec123", "block": True, "timeout": 120000},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_poll_1", "content": "<task_id>rec123</task_id>"}]},
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed TaskStop toolu_stop_1",
                "timestamp": "2026-02-07T00:00:04Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_stop_1",
                        "function_name": "TaskStop",
                        "arguments": {"task_id": "rec123"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_stop_1", "content": "{\"task_id\":\"rec123\"}"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first_response = state.handle_request(path="/v1/messages", headers={}, payload={})
            second_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_bg_1",
                                    "content": (
                                        "Command running in background with ID: live999. "
                                        "Output is being written to: /tmp/live999.output"
                                    ),
                                    "is_error": False,
                                }
                            ],
                        }
                    ]
                },
            )
            third_response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_poll_1",
                                    "content": "<task_id>live999</task_id>",
                                    "is_error": False,
                                }
                            ],
                        }
                    ]
                },
            )

        self.assertEqual(first_response["content"][1]["type"], "tool_use")
        self.assertEqual(first_response["content"][1]["id"], "toolu_bg_1")
        self.assertTrue(first_response["content"][1]["input"]["run_in_background"])
        self.assertEqual(second_response["content"][0]["type"], "tool_use")
        self.assertEqual(second_response["content"][0]["name"], "TaskOutput")
        self.assertEqual(second_response["content"][0]["input"]["task_id"], "live999")
        self.assertEqual(third_response["content"][0]["type"], "tool_use")
        self.assertEqual(third_response["content"][0]["name"], "TaskStop")
        self.assertEqual(third_response["content"][0]["input"]["task_id"], "live999")

    def test_handle_request_detaches_uncontrolled_background_bash_tasks(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Start the background job",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_bg_persist",
                        "function_name": "Bash",
                        "arguments": {"command": "python server.py", "run_in_background": True},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "toolu_bg_persist",
                            "content": (
                                "Command running in background with ID: rec-persist. "
                                "Output is being written to: /tmp/rec-persist.output\n\n"
                                "[metadata] {\"backgroundTaskId\": \"rec-persist\"}"
                            ),
                        }
                    ]
                },
                "extra": {"tool_result_metadata": {"tool_use_result": {"backgroundTaskId": "rec-persist"}}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            response = state.handle_request(path="/v1/messages", headers={}, payload={})

        self.assertEqual(response["content"][0]["type"], "tool_use")
        self.assertEqual(response["content"][0]["name"], "Bash")
        self.assertFalse(response["content"][0]["input"]["run_in_background"])
        self.assertIn("nohup setsid /bin/bash -lc", response["content"][0]["input"]["command"])
        self.assertIn("python server.py", response["content"][0]["input"]["command"])

    def test_handle_request_rewrites_git_hashes_from_live_reflog_tool_result(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Cherry-pick the lost commit",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Bash toolu_1",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_1",
                        "function_name": "Bash",
                        "arguments": {"command": "git cherry-pick 650dba4"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_1", "content": "ok"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_prev",
                                    "content": (
                                        "d7d3e4b HEAD@{0}: checkout: moving from 91d315ea to master\n"
                                        "91d315e HEAD@{1}: commit: Move to Stanford\n"
                                        "c4e38a1 HEAD@{2}: checkout: moving from master to HEAD~1"
                                    ),
                                    "is_error": False,
                                }
                            ],
                        }
                    ]
                },
            )

        self.assertEqual(response["content"][0]["type"], "text")
        self.assertEqual(response["content"][1]["type"], "tool_use")
        self.assertEqual(response["content"][1]["name"], "Bash")
        self.assertEqual(response["content"][1]["input"]["command"], "git cherry-pick 91d315e")

    def test_handle_request_rewrites_conflict_marker_hashes_for_edit_steps(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Resolve the conflict",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Edit toolu_2",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_2",
                        "function_name": "Edit",
                        "arguments": {
                            "file_path": "/app/conflict.txt",
                            "old_string": "<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>> 650dba4 (Move to Stanford)",
                            "new_string": "new",
                            "replace_all": False,
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_2", "content": "ok"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            target_path = Path(tmp) / "conflict.txt"
            target_path.write_text(
                "<<<<<<< HEAD\nold\n=======\nnew\n>>>>>>> 91d315e (Move to Stanford)",
                encoding="utf-8",
            )
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            response = state.handle_request(
                path="/v1/messages",
                headers={},
                payload={
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": "toolu_prev",
                                    "content": "91d315e HEAD@{1}: commit: Move to Stanford",
                                    "is_error": False,
                                }
                            ],
                        }
                    ]
                },
            )
            command = response["content"][1]["input"]["command"].replace("/app/conflict.txt", str(target_path))
            subprocess.run(["bash", "-lc", command], check=True, cwd=tmp)
            self.assertEqual(target_path.read_text(encoding="utf-8"), "new")

    def test_strict_replayable_allows_write_and_edit_metadata_fallbacks(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Prepare outputs",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed Write toolu_4",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_4",
                        "function_name": "Write",
                        "arguments": {"file_path": "/app/out.txt", "content": "$51"},
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": "toolu_4",
                            "content": (
                                "File created successfully\n\n"
                                "[metadata] {\"filePath\": \"/app/out.txt\", \"content\": \"hello\"}"
                            ),
                        }
                    ]
                },
                "extra": {
                    "tool_result_metadata": {
                        "tool_use_result": {"filePath": "/app/out.txt", "content": "hello"}
                    }
                },
            },
            {
                "source": "agent",
                "message": "Executed Edit toolu_5",
                "timestamp": "2026-02-07T00:00:03Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_5",
                        "function_name": "Edit",
                        "arguments": {
                            "file_path": "/app/out.txt",
                            "old_string": "$52",
                            "new_string": "$53",
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_5", "content": "$54"}]},
                "extra": {
                    "tool_result_metadata": {
                        "tool_use_result": {
                            "filePath": "/app/out.txt",
                            "oldString": "hello",
                            "newString": "goodbye",
                        }
                    }
                },
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertTrue(is_strictly_replayable_trace(trace_path))
            self.assertEqual(strict_replayability_errors(trace_path), ())

    def test_strict_replayable_rejects_unresolved_bash_placeholders(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Executed Bash toolu_6",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_6",
                        "function_name": "Bash",
                        "arguments": {"command": "$aa", "description": "unknown"},
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_6", "content": "stdout"}]},
                "extra": {"tool_result_metadata": {"tool_use_result": {"stdout": "stdout"}}},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            errors = strict_replayability_errors(trace_path)
            response = state.handle_request(path="/v1/messages", headers={}, payload={})

        self.assertEqual(errors, ("step 0: unresolved Bash command",))
        tool_block = response["content"][0]
        self.assertEqual(tool_block["type"], "tool_use")
        self.assertEqual(tool_block["name"], "Bash")
        self.assertIn("replay skipped unresolved Bash command", tool_block["input"]["command"])

    def test_strict_replayable_rejects_task_tool_dependency(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Executed Task toolu_7",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_7",
                        "function_name": "Task",
                        "arguments": {
                            "description": "Build helper files",
                            "subagent_type": "general-purpose",
                            "prompt": "Create /app/attack.py",
                        },
                    }
                ],
                "observation": {"results": [{"source_call_id": "toolu_7", "content": "subagent finished"}]},
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertFalse(is_strictly_replayable_trace(trace_path))
            self.assertEqual(
                strict_replayability_errors(trace_path),
                ("step 0: Task tool side effects are not strictly replayable",),
            )

    def test_strict_replayable_rejects_task_output_and_task_stop_tools(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Executed TaskOutput toolu_8",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_8",
                        "function_name": "TaskOutput",
                        "arguments": {"task_id": "abc123", "block": True},
                    }
                ],
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Executed TaskStop toolu_9",
                "timestamp": "2026-02-07T00:00:02Z",
                "model_name": "claude-opus-4-6",
                "tool_calls": [
                    {
                        "tool_call_id": "toolu_9",
                        "function_name": "TaskStop",
                        "arguments": {"task_id": "abc123"},
                    }
                ],
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertEqual(
                strict_replayability_errors(trace_path),
                (
                    "step 0: TaskOutput tool side effects are not strictly replayable",
                    "step 1: TaskStop tool side effects are not strictly replayable",
                ),
            )

    def test_snapshot_includes_delay_bounds_defaults(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Plan first",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            snapshot = state.snapshot()

        self.assertEqual(snapshot["minimal_delay"], 0.0)
        self.assertEqual(snapshot["maximal_delay"], 1_000_000_000.0)

    def test_snapshot_accepts_include_events_flag(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Plan first",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            snapshot = state.snapshot(include_events=False)

        self.assertEqual(snapshot["trace_cursor"], 0)
        self.assertEqual(snapshot["total_responses"], 1)

    def test_trace_replay_delay_is_clamped_to_bounds(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "First",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
            {
                "source": "agent",
                "message": "Second",
                "timestamp": "2026-02-07T00:00:05Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_policy": "trace_replay",
                    "response_delay_scaling_factor": 0.25,
                    "minimal_delay": 1500,
                    "maximal_delay": 3000,
                }
            )

            with patch("integrations.llm_services.claude_code_trace_replay.service.time.sleep") as sleep:
                state.handle_request(path="/v1/messages", headers={}, payload={})
                state.handle_request(path="/v1/messages", headers={}, payload={})

        sleep.assert_called_once_with(1.5)

    def test_fixed_delay_is_clamped_to_maximum_bound(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Only turn",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_policy": "fixed",
                    "response_delay_ms": 4000,
                    "maximal_delay": 2500,
                }
            )

            with patch("integrations.llm_services.claude_code_trace_replay.service.time.sleep") as sleep:
                state.handle_request(path="/v1/messages", headers={}, payload={})

        sleep.assert_called_once_with(2.5)

    def test_invalid_delay_bounds_are_rejected(self) -> None:
        payload = _base_trace_payload()
        payload["steps"] = [
            {"source": "user", "message": "task", "timestamp": "2026-02-07T00:00:00Z", "extra": {}},
            {
                "source": "agent",
                "message": "Only turn",
                "timestamp": "2026-02-07T00:00:01Z",
                "model_name": "claude-opus-4-6",
                "extra": {},
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.json"
            trace_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "maximal_delay"):
                TraceReplayLLMState(
                    llm_service_config={
                        "trace_path": str(trace_path),
                        "minimal_delay": 10,
                        "maximal_delay": 5,
                    }
                )


if __name__ == "__main__":
    unittest.main()

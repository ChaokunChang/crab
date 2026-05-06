"""Unit tests for the spec-decision collection pipeline.

Phase A only — no replay-side wiring is exercised here. We cover:

* Sidecar schema round-trip + path resolution
* Per-turn prompt reconstruction for terminus + mini_swe trajectories
* Acceptance scoring under the supported match policies
* End-to-end collection script logic with the network mocked out
"""

from __future__ import annotations

import csv as csvmod
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from integrations.llm_services.speculation.draft_client import DraftResult
from integrations.llm_services.speculation.reconstruct import (
    reconstruct_claude_code_turns,
    reconstruct_mini_swe_turns,
    reconstruct_terminus_turns,
)
from integrations.llm_services.speculation.schema import (
    AGENT_CLAUDE_CODE,
    AGENT_MINI_SWE,
    AGENT_TERMINUS,
    LEVEL_LITERAL,
    LEVEL_NORMALIZED,
    LEVEL_SEMANTIC,
    SCORE_LEVELS,
    SpeculationSidecar,
    SpeculationTurn,
    load_sidecar,
    resolve_side_by_side_csv_path,
    resolve_sidecar_path,
    write_side_by_side_csv,
    write_sidecar,
)
from integrations.llm_services.speculation.score import (
    extract_first_command,
    is_task_complete,
    score_levels,
)


def _terminus_response(command: str, task_complete: bool = False) -> str:
    payload = {
        "analysis": "a",
        "plan": "p",
        "commands": [{"keystrokes": command, "duration": 0.1}],
    }
    if task_complete:
        payload["task_complete"] = True
    return json.dumps(payload)


def _build_terminus_trajectory(commands: list[str]) -> dict:
    """Build a minimal terminus trajectory.json with one agent step per command.

    Each agent step's ``observation`` is a synthetic terminal echo so the
    reconstructor can validate that observations are threaded into the next
    turn's user message.
    """
    steps: list[dict] = [
        {
            "step_id": 1,
            "timestamp": "2026-04-30T00:00:00+00:00",
            "source": "user",
            "message": "SYSTEM_PROMPT_AND_TASK_DESCRIPTION",
        }
    ]
    for index, command in enumerate(commands):
        steps.append(
            {
                "step_id": index + 2,
                "timestamp": f"2026-04-30T00:00:{index + 1:02d}+00:00",
                "source": "agent",
                "message": f"Analysis: turn {index}\nPlan: run {command}",
                "tool_calls": [
                    {
                        "tool_call_id": f"call_{index}",
                        "function_name": "bash_command",
                        "arguments": {"keystrokes": command, "duration": 0.1},
                    }
                ],
                "observation": {
                    "results": [
                        {"content": f"OBSERVATION_AFTER_TURN_{index}"}
                    ]
                },
            }
        )
    return {
        "schema_version": "ATIF-v1.6",
        "agent": {"model_name": "test-model"},
        "steps": steps,
    }


def _build_claude_code_trajectory(steps_spec: list[dict]) -> dict:
    """Build a synthetic claude_code trajectory.

    ``steps_spec`` is a list of:
      * ``{"role": "user", "message": "..."}`` for user steps,
      * ``{"role": "agent", "thinking": "..."}`` for text-only thinking steps,
      * ``{"role": "agent", "tool": "Bash", "args": {"command": "ls"}, "obs": "..."}``
        for tool-call steps. ``thinking`` may also be present on a tool step
        if the assistant emitted text before the tool call in the same call.
    """
    steps = []
    step_id = 1
    base = 1_700_000_000.0
    for i, spec in enumerate(steps_spec):
        ts = base + i
        ts_iso = (
            "2026-04-30T18:00:00.000Z"
            if i == 0
            else f"2026-04-30T18:00:{i:02d}.000Z"
        )
        if spec["role"] == "user":
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": ts_iso,
                    "source": "user",
                    "message": spec["message"],
                    "extra": {"is_sidechain": False},
                }
            )
        elif spec["role"] == "agent" and "tool" not in spec:
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": ts_iso,
                    "source": "agent",
                    "model_name": "claude-opus-4-6",
                    "message": spec.get("thinking", ""),
                    "extra": {"is_sidechain": False},
                }
            )
        else:
            call_id = f"toolu_test_{step_id}"
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": ts_iso,
                    "source": "agent",
                    "model_name": "claude-opus-4-6",
                    "message": spec.get("thinking", f"Executed {spec['tool']} {call_id}"),
                    "tool_calls": [
                        {
                            "tool_call_id": call_id,
                            "function_name": spec["tool"],
                            "arguments": spec.get("args", {}),
                        }
                    ],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": call_id,
                                "content": spec.get("obs", ""),
                            }
                        ]
                    },
                    "extra": {"is_sidechain": False},
                }
            )
        step_id += 1
    return {
        "schema_version": "ATIF-v1.6",
        "agent": {"name": "claude-code", "model_name": "claude-opus-4-6"},
        "steps": steps,
    }


def _build_mini_swe_trajectory(commands: list[str]) -> dict:
    messages: list[dict] = [
        {"role": "system", "content": "you are mini-swe"},
        {"role": "user", "content": "<pr_description>fix it</pr_description>"},
    ]
    for index, command in enumerate(commands):
        messages.append(
            {
                "role": "assistant",
                "content": (
                    f"THOUGHT: turn {index}\n\n"
                    f"<mswea_bash_command>{command}</mswea_bash_command>"
                ),
            }
        )
        messages.append({"role": "user", "content": f"OBS_{index}"})
    return {"messages": messages}


class SchemaTests(unittest.TestCase):
    def test_sidecar_round_trip(self) -> None:
        sidecar = SpeculationSidecar(
            trace_path="/tmp/trace.json",
            agent_kind=AGENT_TERMINUS,
            draft_model={"tag": "deepseek-fast", "name": "deepseek-chat"},
            turns=[
                SpeculationTurn(
                    turn_index=0,
                    oracle_first_command="ls -la\n",
                    draft_response_content=_terminus_response("ls -la\n"),
                    draft_first_command="ls -la\n",
                    accepted={lvl: True for lvl in SCORE_LEVELS},
                    draft_latency_ms=42.5,
                    draft_prompt_tokens=100,
                    draft_completion_tokens=12,
                ),
                SpeculationTurn(
                    turn_index=1,
                    oracle_first_command="git status\n",
                    draft_response_content="",
                    draft_first_command="",
                    accepted={lvl: False for lvl in SCORE_LEVELS},
                    error="HTTP 500",
                ),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.spec.json"
            write_sidecar(path, sidecar)
            loaded = load_sidecar(path)
        self.assertEqual(loaded.agent_kind, AGENT_TERMINUS)
        self.assertEqual(set(loaded.score_levels), set(SCORE_LEVELS))
        self.assertEqual(len(loaded.turns), 2)
        self.assertEqual(loaded.turns[0].draft_first_command, "ls -la\n")
        self.assertTrue(loaded.turns[0].accepted[LEVEL_LITERAL])
        self.assertEqual(loaded.turns[1].error, "HTTP 500")
        summary = loaded.summary()
        self.assertEqual(summary["turns"], 2)
        self.assertEqual(summary["scored"], 1)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["accept_counts"][LEVEL_LITERAL], 1)
        self.assertAlmostEqual(summary["accept_rates"][LEVEL_LITERAL], 1.0)

    def test_v1_sidecar_loads_with_legacy_accepted_bool(self) -> None:
        """v1 sidecars stored ``accepted: bool``; we accept them on load."""
        legacy_payload = {
            "schema_version": "1",
            "trace_path": "/tmp/trace.json",
            "agent_kind": AGENT_TERMINUS,
            "draft_model": {"tag": "x", "name": "x"},
            "match_policy": LEVEL_NORMALIZED,
            "turns": [
                {
                    "turn_index": 0,
                    "oracle_first_command": "ls\n",
                    "draft_response_content": "{}",
                    "draft_first_command": "ls\n",
                    "accepted": True,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.spec.json"
            path.write_text(json.dumps(legacy_payload))
            loaded = load_sidecar(path)
        self.assertEqual(loaded.score_levels, [LEVEL_NORMALIZED])
        self.assertTrue(loaded.turns[0].accepted[LEVEL_NORMALIZED])

    def test_resolve_sidecar_path_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "traces" / "task_a" / "trajectory.json"
            trace.parent.mkdir(parents=True)
            trace.write_text("{}")
            p1 = resolve_sidecar_path(sidecar_root=root, draft_tag="deepseek-fast", trace_path=trace)
            p2 = resolve_sidecar_path(sidecar_root=root, draft_tag="deepseek-fast", trace_path=trace)
            self.assertEqual(p1, p2)
            # Different trace -> different path
            other = root / "traces" / "task_b" / "trajectory.json"
            other.parent.mkdir(parents=True)
            other.write_text("{}")
            p3 = resolve_sidecar_path(sidecar_root=root, draft_tag="deepseek-fast", trace_path=other)
            self.assertNotEqual(p1, p3)
            # Different tag -> different path
            p4 = resolve_sidecar_path(sidecar_root=root, draft_tag="other-tag", trace_path=trace)
            self.assertNotEqual(p1, p4)

    def test_unknown_agent_kind_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpeculationSidecar(trace_path="x", agent_kind="bogus", draft_model={})

    def test_unknown_score_level_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SpeculationSidecar(
                trace_path="x",
                agent_kind=AGENT_TERMINUS,
                draft_model={},
                score_levels=["bogus"],
            )


class ClaudeCodeReconstructTests(unittest.TestCase):
    def _reconstruct(self, steps_spec: list[dict]) -> list:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.json"
            path.write_text(json.dumps(_build_claude_code_trajectory(steps_spec)))
            return reconstruct_claude_code_turns(path)

    def test_thinking_step_is_coalesced_with_following_tool_call(self) -> None:
        turns = self._reconstruct(
            [
                {"role": "user", "message": "TASK"},
                {"role": "agent", "thinking": "I should check the dir"},
                {
                    "role": "agent",
                    "thinking": "",
                    "tool": "Bash",
                    "args": {"command": "ls -la\n"},
                    "obs": "file1\nfile2\n",
                },
                {
                    "role": "agent",
                    "tool": "Read",
                    "args": {"file_path": "/app/file1"},
                    "obs": "...content...",
                },
            ]
        )
        # Two coalesced turns (the trailing thinking-only step before the
        # first Bash gets folded in; second turn has no preceding thinking).
        self.assertEqual(len(turns), 2)
        envelope_0 = json.loads(turns[0].oracle_response_content)
        self.assertIn("check the dir", envelope_0["thinking"])
        self.assertEqual(envelope_0["tool_calls"][0]["name"], "Bash")
        self.assertEqual(envelope_0["tool_calls"][0]["arguments"]["command"], "ls -la\n")

    def test_input_messages_use_openai_tool_format(self) -> None:
        turns = self._reconstruct(
            [
                {"role": "user", "message": "TASK"},
                {
                    "role": "agent",
                    "tool": "Bash",
                    "args": {"command": "ls\n"},
                    "obs": "OUTPUT_OF_LS",
                },
                {
                    "role": "agent",
                    "tool": "Read",
                    "args": {"file_path": "/app/x"},
                    "obs": "FILE_CONTENT",
                },
            ]
        )
        # Second turn should see: system + user(TASK) + assistant(tool_call=Bash)
        # + tool(result=OUTPUT_OF_LS).
        msgs = turns[1].input_messages
        self.assertEqual(msgs[0]["role"], "system")
        self.assertEqual(msgs[1]["role"], "user")
        self.assertEqual(msgs[1]["content"], "TASK")
        self.assertEqual(msgs[2]["role"], "assistant")
        self.assertEqual(len(msgs[2]["tool_calls"]), 1)
        self.assertEqual(msgs[2]["tool_calls"][0]["function"]["name"], "Bash")
        # Arguments must be a JSON string (per OpenAI tool-call format).
        decoded_args = json.loads(msgs[2]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(decoded_args["command"], "ls\n")
        self.assertEqual(msgs[3]["role"], "tool")
        self.assertEqual(msgs[3]["content"], "OUTPUT_OF_LS")
        self.assertEqual(
            msgs[3]["tool_call_id"], msgs[2]["tool_calls"][0]["id"]
        )

    def test_trailing_thinking_becomes_final_response_turn(self) -> None:
        turns = self._reconstruct(
            [
                {"role": "user", "message": "TASK"},
                {
                    "role": "agent",
                    "tool": "Bash",
                    "args": {"command": "ls\n"},
                    "obs": "out",
                },
                {"role": "agent", "thinking": "All done."},
            ]
        )
        self.assertEqual(len(turns), 2)
        last = json.loads(turns[1].oracle_response_content)
        self.assertEqual(last["tool_calls"], [])
        self.assertIn("done", last["thinking"].lower())

    def test_oracle_first_command_renders_tool_and_primary_arg(self) -> None:
        turns = self._reconstruct(
            [
                {"role": "user", "message": "TASK"},
                {
                    "role": "agent",
                    "tool": "Bash",
                    "args": {"command": "ls\n"},
                    "obs": "x",
                },
            ]
        )
        # The score module's extractor turns the JSON envelope into a
        # comparison string.
        from integrations.llm_services.speculation.score import (
            extract_first_command,
        )

        rendered = extract_first_command(
            agent_kind=AGENT_CLAUDE_CODE,
            content=turns[0].oracle_response_content,
        )
        self.assertEqual(rendered, "Bash: ls\n")


class ClaudeCodeScoreTests(unittest.TestCase):
    def _envelope(self, *, name: str | None, args: dict | None, thinking: str = "") -> str:
        tool_calls = []
        if name is not None:
            tool_calls.append({"name": name, "arguments": args or {}})
        return json.dumps({"thinking": thinking, "tool_calls": tool_calls})

    def test_identical_bash_commands_match_at_all_levels(self) -> None:
        from integrations.llm_services.speculation.score import score_levels

        oracle = self._envelope(name="Bash", args={"command": "ls\n"})
        verdict = score_levels(
            agent_kind=AGENT_CLAUDE_CODE, oracle_content=oracle, draft_content=oracle
        )
        self.assertEqual(verdict, {lvl: True for lvl in SCORE_LEVELS})

    def test_different_tools_reject_at_all_levels(self) -> None:
        from integrations.llm_services.speculation.score import score_levels

        oracle = self._envelope(name="Bash", args={"command": "ls\n"})
        draft = self._envelope(name="Read", args={"file_path": "/a"})
        verdict = score_levels(
            agent_kind=AGENT_CLAUDE_CODE, oracle_content=oracle, draft_content=draft
        )
        self.assertFalse(verdict[LEVEL_LITERAL])
        self.assertFalse(verdict[LEVEL_NORMALIZED])
        self.assertFalse(verdict[LEVEL_SEMANTIC])

    def test_final_turn_different_thinking_text_rejects_all_levels(self) -> None:
        from integrations.llm_services.speculation.score import score_levels

        oracle = self._envelope(name=None, args=None, thinking="all done")
        draft = self._envelope(name=None, args=None, thinking="finished")
        verdict = score_levels(
            agent_kind=AGENT_CLAUDE_CODE,
            oracle_content=oracle,
            draft_content=draft,
            is_final_turn=True,
        )
        self.assertFalse(verdict[LEVEL_LITERAL])
        self.assertFalse(verdict[LEVEL_NORMALIZED])
        self.assertFalse(verdict[LEVEL_SEMANTIC])

    def test_final_turn_identical_thinking_matches_all_levels(self) -> None:
        from integrations.llm_services.speculation.score import score_levels

        oracle = self._envelope(name=None, args=None, thinking="all done")
        draft = self._envelope(name=None, args=None, thinking="all done")
        verdict = score_levels(
            agent_kind=AGENT_CLAUDE_CODE,
            oracle_content=oracle,
            draft_content=draft,
            is_final_turn=True,
        )
        self.assertTrue(verdict[LEVEL_LITERAL])
        self.assertTrue(verdict[LEVEL_NORMALIZED])
        self.assertTrue(verdict[LEVEL_SEMANTIC])

    def test_mid_traj_both_empty_tool_calls_accepts_all_levels(self) -> None:
        from integrations.llm_services.speculation.score import score_levels

        oracle = self._envelope(name=None, args=None, thinking="thinking...")
        draft = self._envelope(name=None, args=None, thinking="(empty)")
        verdict = score_levels(
            agent_kind=AGENT_CLAUDE_CODE,
            oracle_content=oracle,
            draft_content=draft,
            is_final_turn=False,
        )
        # Mid-trajectory both-empty-tool-call ⇒ wait-like ⇒ accept all.
        self.assertTrue(verdict[LEVEL_LITERAL])
        self.assertTrue(verdict[LEVEL_NORMALIZED])
        self.assertTrue(verdict[LEVEL_SEMANTIC])

    def test_todowrite_match_ignores_arguments(self) -> None:
        from integrations.llm_services.speculation.score import score_levels

        oracle = self._envelope(name="TodoWrite", args={"todos": [{"x": 1}]})
        draft = self._envelope(name="TodoWrite", args={"todos": [{"y": 2}]})
        verdict = score_levels(
            agent_kind=AGENT_CLAUDE_CODE, oracle_content=oracle, draft_content=draft
        )
        # primary_arg returns "" for TodoWrite, so both render to "TodoWrite";
        # they match across all levels.
        self.assertTrue(verdict[LEVEL_LITERAL])


class CsvOutputTests(unittest.TestCase):
    def _make_sidecar(self) -> SpeculationSidecar:
        return SpeculationSidecar(
            trace_path="/tmp/trace.json",
            agent_kind=AGENT_TERMINUS,
            draft_model={"tag": "deepseek-chat", "name": "deepseek-chat"},
            turns=[
                SpeculationTurn(
                    turn_index=0,
                    oracle_first_command="git status -sb\n",
                    draft_response_content=_terminus_response("git status\n"),
                    draft_first_command="git status\n",
                    accepted={
                        LEVEL_LITERAL: False,
                        LEVEL_NORMALIZED: False,
                        LEVEL_SEMANTIC: False,
                    },
                    oracle_latency_ms=8000.0,
                    draft_latency_ms=2000.0,
                ),
                SpeculationTurn(
                    turn_index=1,
                    oracle_first_command="ls -la\n",
                    draft_response_content=_terminus_response("ls    -la\n"),
                    draft_first_command="ls    -la\n",
                    accepted={
                        LEVEL_LITERAL: False,
                        LEVEL_NORMALIZED: True,
                        LEVEL_SEMANTIC: True,
                    },
                    oracle_latency_ms=None,  # missing for first recorded turn
                    draft_latency_ms=1500.0,
                ),
                SpeculationTurn(
                    turn_index=2,
                    oracle_first_command="",
                    draft_response_content="",
                    draft_first_command="",
                    accepted={
                        LEVEL_LITERAL: False,
                        LEVEL_NORMALIZED: False,
                        LEVEL_SEMANTIC: False,
                    },
                    error="HTTP 500",
                ),
            ],
        )

    def test_side_by_side_csv_path_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trace = root / "traces" / "task" / "trajectory.json"
            trace.parent.mkdir(parents=True)
            trace.write_text("{}")
            path = resolve_side_by_side_csv_path(
                csv_root=root,
                draft_tag="deepseek-chat",
                run_id="20260430-180000",
                trace_path=trace,
            )
            self.assertTrue(str(path).startswith(str((root).resolve())))
            self.assertEqual(path.parent.parent.name, "deepseek-chat")
            self.assertEqual(path.parent.name, "20260430-180000")
            self.assertTrue(path.name.endswith("__trajectory.csv"))

    def test_side_by_side_csv_round_trips_values(self) -> None:
        sidecar = self._make_sidecar()
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            write_side_by_side_csv(csv_path, sidecar)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csvmod.reader(handle))
        self.assertEqual(
            rows[0],
            [
                "turn",
                "oracle",
                "draft",
                LEVEL_LITERAL,
                LEVEL_NORMALIZED,
                LEVEL_SEMANTIC,
                "oracle_latency_ms",
                "draft_latency_ms",
                "speedup",
                "error",
            ],
        )
        self.assertEqual(len(rows), 4)  # header + 3 turns
        # Turn 0 row: all rejected, latencies populated, speedup = 8000/2000 = 4
        self.assertEqual(rows[1][0], "0")
        self.assertEqual(rows[1][1], "git status -sb\n")
        self.assertEqual(rows[1][2], "git status\n")
        self.assertEqual(rows[1][3:6], ["false", "false", "false"])
        self.assertEqual(rows[1][6], "8000.0")
        self.assertEqual(rows[1][7], "2000.0")
        self.assertEqual(rows[1][8], "4.0000")
        self.assertEqual(rows[1][9], "")
        # Turn 1: oracle latency missing -> oracle_latency_ms blank, speedup blank
        self.assertEqual(rows[2][3:6], ["false", "true", "true"])
        self.assertEqual(rows[2][6], "")
        self.assertEqual(rows[2][7], "1500.0")
        self.assertEqual(rows[2][8], "")
        # Turn 2: error populated, latencies missing
        self.assertEqual(rows[3][6], "")
        self.assertEqual(rows[3][7], "")
        self.assertEqual(rows[3][8], "")
        self.assertEqual(rows[3][9], "HTTP 500")

    def test_side_by_side_csv_quotes_special_chars(self) -> None:
        sidecar = SpeculationSidecar(
            trace_path="/tmp/trace.json",
            agent_kind=AGENT_TERMINUS,
            draft_model={},
            turns=[
                SpeculationTurn(
                    turn_index=0,
                    oracle_first_command='echo "a, b"\n',
                    draft_response_content="",
                    draft_first_command="echo 'a, b'\n",
                    accepted={lvl: False for lvl in SCORE_LEVELS},
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "out.csv"
            write_side_by_side_csv(csv_path, sidecar)
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csvmod.reader(handle))
        # Both embedded commas and quotes round-trip intact through the
        # csv-module's RFC 4180 escaping.
        self.assertEqual(rows[1][1], 'echo "a, b"\n')
        self.assertEqual(rows[1][2], "echo 'a, b'\n")


class ReconstructTests(unittest.TestCase):
    def test_terminus_first_turn_has_only_initial_user(self) -> None:
        trajectory = _build_terminus_trajectory(["ls -la\n", "pwd\n", "echo done\n"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.json"
            path.write_text(json.dumps(trajectory))
            turns = reconstruct_terminus_turns(path)
        self.assertEqual(len(turns), 3)
        first = turns[0]
        self.assertEqual(first.turn_index, 0)
        self.assertEqual(len(first.input_messages), 1)
        self.assertEqual(first.input_messages[0]["role"], "user")
        self.assertIn("SYSTEM_PROMPT", first.input_messages[0]["content"])

    def test_terminus_oracle_latency_diffs_step_timestamps(self) -> None:
        # Each agent step is at T+1s, T+2s, T+3s after the user step at T+0.
        trajectory = _build_terminus_trajectory(["a\n", "b\n", "c\n"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.json"
            path.write_text(json.dumps(trajectory))
            turns = reconstruct_terminus_turns(path)
        # First turn: 1s after the initial user step.
        self.assertAlmostEqual(turns[0].oracle_latency_ms, 1000.0, places=1)
        # Subsequent turns: 1s gap each.
        self.assertAlmostEqual(turns[1].oracle_latency_ms, 1000.0, places=1)
        self.assertAlmostEqual(turns[2].oracle_latency_ms, 1000.0, places=1)

    def test_mini_swe_oracle_latency_uses_extra_timestamp(self) -> None:
        messages = [
            {"role": "system", "content": "sys", "extra": {"timestamp": 1000.0}},
            {"role": "user", "content": "go", "extra": {"timestamp": 1001.0}},
            {
                "role": "assistant",
                "content": "<mswea_bash_command>ls</mswea_bash_command>",
                "extra": {"timestamp": 1003.5},
            },
            {"role": "user", "content": "out", "extra": {"timestamp": 1003.6}},
            {
                "role": "assistant",
                "content": "<mswea_bash_command>pwd</mswea_bash_command>",
                "extra": {"timestamp": 1005.0},
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.traj.json"
            path.write_text(json.dumps({"messages": messages}))
            turns = reconstruct_mini_swe_turns(path)
        self.assertEqual(len(turns), 2)
        # 1003.5 - 1001.0 = 2.5s = 2500ms
        self.assertAlmostEqual(turns[0].oracle_latency_ms, 2500.0, places=1)
        # 1005.0 - 1003.6 = 1.4s = 1400ms
        self.assertAlmostEqual(turns[1].oracle_latency_ms, 1400.0, places=1)

    def test_terminus_second_turn_includes_first_observation(self) -> None:
        trajectory = _build_terminus_trajectory(["ls -la\n", "pwd\n"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trajectory.json"
            path.write_text(json.dumps(trajectory))
            turns = reconstruct_terminus_turns(path)
        second = turns[1]
        self.assertEqual(second.turn_index, 1)
        # user, assistant, user (observation from turn 0)
        roles = [m["role"] for m in second.input_messages]
        self.assertEqual(roles, ["user", "assistant", "user"])
        self.assertIn("OBSERVATION_AFTER_TURN_0", second.input_messages[-1]["content"])
        # Assistant content is a JSON blob with the recorded keystrokes.
        decoded = json.loads(second.input_messages[1]["content"])
        self.assertEqual(decoded["commands"][0]["keystrokes"], "ls -la\n")

    def test_mini_swe_reconstruct_alternates_correctly(self) -> None:
        trajectory = _build_mini_swe_trajectory(["ls\n", "pytest -k foo\n"])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "task.traj.json"
            path.write_text(json.dumps(trajectory))
            turns = reconstruct_mini_swe_turns(path)
        self.assertEqual(len(turns), 2)
        # First turn: system + user only
        self.assertEqual([m["role"] for m in turns[0].input_messages], ["system", "user"])
        # Second turn: system, user, assistant, user
        self.assertEqual(
            [m["role"] for m in turns[1].input_messages],
            ["system", "user", "assistant", "user"],
        )
        self.assertIn("<mswea_bash_command>ls", turns[1].input_messages[2]["content"])


class ScoreTests(unittest.TestCase):
    def test_terminus_command_extraction(self) -> None:
        content = _terminus_response("git status\n")
        self.assertEqual(
            extract_first_command(agent_kind=AGENT_TERMINUS, content=content), "git status\n"
        )

    def test_terminus_command_extraction_handles_fenced_json(self) -> None:
        wrapped = "```json\n" + _terminus_response("ls\n") + "\n```"
        self.assertEqual(extract_first_command(agent_kind=AGENT_TERMINUS, content=wrapped), "ls\n")

    def test_terminus_command_extraction_returns_empty_for_garbage(self) -> None:
        self.assertEqual(
            extract_first_command(agent_kind=AGENT_TERMINUS, content="totally not json"),
            "",
        )

    def test_levels_strict_to_loose_for_identical_commands(self) -> None:
        oracle = _terminus_response("ls -la\n")
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS, oracle_content=oracle, draft_content=oracle
        )
        self.assertEqual(verdict, {lvl: True for lvl in SCORE_LEVELS})

    def test_normalized_accepts_collapsed_whitespace_but_literal_does_not(self) -> None:
        oracle = _terminus_response("ls -la\n")
        draft = _terminus_response("ls    -la\n")
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS, oracle_content=oracle, draft_content=draft
        )
        self.assertFalse(verdict[LEVEL_LITERAL])
        self.assertTrue(verdict[LEVEL_NORMALIZED])
        self.assertTrue(verdict[LEVEL_SEMANTIC])

    def test_semantic_accepts_heredoc_operator_whitespace_drift(self) -> None:
        oracle = _terminus_response("cat > f.md <<'EOF'\nbody\nEOF\n")
        draft = _terminus_response("cat > f.md << 'EOF'\nbody\nEOF\n")
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS, oracle_content=oracle, draft_content=draft
        )
        self.assertFalse(verdict[LEVEL_LITERAL])
        self.assertFalse(verdict[LEVEL_NORMALIZED])
        self.assertTrue(verdict[LEVEL_SEMANTIC])

    def test_mid_traj_wait_both_empty_accepts_all_levels(self) -> None:
        # NOT the final turn — both empty keystrokes ⇒ wait-vs-wait ⇒ accept.
        oracle = _terminus_response("", task_complete=False)
        draft = _terminus_response("", task_complete=False)
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS,
            oracle_content=oracle,
            draft_content=draft,
            is_final_turn=False,
        )
        self.assertTrue(verdict[LEVEL_LITERAL])
        self.assertTrue(verdict[LEVEL_NORMALIZED])
        self.assertTrue(verdict[LEVEL_SEMANTIC])

    def test_mid_traj_wait_only_oracle_empty_rejects(self) -> None:
        oracle = _terminus_response("", task_complete=False)
        draft = _terminus_response("ls\n", task_complete=False)
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS,
            oracle_content=oracle,
            draft_content=draft,
            is_final_turn=False,
        )
        self.assertFalse(verdict[LEVEL_LITERAL])
        self.assertFalse(verdict[LEVEL_NORMALIZED])
        self.assertFalse(verdict[LEVEL_SEMANTIC])

    def test_final_turn_text_match_passes_all_levels(self) -> None:
        # Identical analysis+plan text ⇒ literal match.
        oracle = _terminus_response("", task_complete=True)
        draft = _terminus_response("", task_complete=True)
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS,
            oracle_content=oracle,
            draft_content=draft,
            is_final_turn=True,
        )
        self.assertTrue(verdict[LEVEL_LITERAL])
        self.assertTrue(verdict[LEVEL_NORMALIZED])
        self.assertTrue(verdict[LEVEL_SEMANTIC])

    def test_final_turn_different_text_rejects_all_levels(self) -> None:
        # Different analysis text ⇒ all three reject (the user-described
        # "usually they are not same at three levels" case).
        oracle = json.dumps(
            {"analysis": "fully done", "plan": "noop", "commands": [], "task_complete": True}
        )
        draft = json.dumps(
            {"analysis": "task finished successfully", "plan": "", "commands": [], "task_complete": True}
        )
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS,
            oracle_content=oracle,
            draft_content=draft,
            is_final_turn=True,
        )
        self.assertFalse(verdict[LEVEL_LITERAL])
        self.assertFalse(verdict[LEVEL_NORMALIZED])
        self.assertFalse(verdict[LEVEL_SEMANTIC])

    def test_semantic_does_not_short_circuit_when_only_one_task_complete(self) -> None:
        # When the oracle's final-text says nothing (empty analysis/plan)
        # but the draft proposed a command, treat as standard command
        # comparison: the oracle's command is empty so literal/normalized
        # fall back to wait-comparison logic — but we are NOT final here.
        oracle = _terminus_response("", task_complete=False)
        draft = _terminus_response("ls\n", task_complete=False)
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS,
            oracle_content=oracle,
            draft_content=draft,
            is_final_turn=False,
        )
        self.assertFalse(verdict[LEVEL_SEMANTIC])

    def test_score_levels_rejects_different_commands(self) -> None:
        oracle = _terminus_response("ls\n")
        draft = _terminus_response("pwd\n")
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS, oracle_content=oracle, draft_content=draft
        )
        self.assertFalse(verdict[LEVEL_LITERAL])
        self.assertFalse(verdict[LEVEL_NORMALIZED])
        self.assertFalse(verdict[LEVEL_SEMANTIC])

    def test_score_levels_rejects_unparseable_draft(self) -> None:
        oracle = _terminus_response("ls\n")
        verdict = score_levels(
            agent_kind=AGENT_TERMINUS, oracle_content=oracle, draft_content="<no json here>"
        )
        self.assertFalse(verdict[LEVEL_LITERAL])
        self.assertFalse(verdict[LEVEL_NORMALIZED])
        self.assertFalse(verdict[LEVEL_SEMANTIC])

    def test_is_task_complete_terminus(self) -> None:
        self.assertTrue(
            is_task_complete(
                agent_kind=AGENT_TERMINUS,
                content=_terminus_response("", task_complete=True),
            )
        )
        self.assertFalse(
            is_task_complete(
                agent_kind=AGENT_TERMINUS, content=_terminus_response("ls\n")
            )
        )

    def test_is_task_complete_mini_swe(self) -> None:
        self.assertTrue(
            is_task_complete(
                agent_kind=AGENT_MINI_SWE,
                content="<mswea_bash_command>COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT</mswea_bash_command>",
            )
        )
        self.assertFalse(
            is_task_complete(agent_kind=AGENT_MINI_SWE, content="just a thought")
        )

    def test_mini_swe_command_extraction(self) -> None:
        content = "THOUGHT: x\n<mswea_bash_command>ls -la</mswea_bash_command>"
        self.assertEqual(
            extract_first_command(agent_kind=AGENT_MINI_SWE, content=content), "ls -la"
        )

    def test_mini_swe_returns_empty_when_multiple_commands(self) -> None:
        content = (
            "<mswea_bash_command>a</mswea_bash_command>"
            "<mswea_bash_command>b</mswea_bash_command>"
        )
        self.assertEqual(extract_first_command(agent_kind=AGENT_MINI_SWE, content=content), "")


class CollectionScriptTests(unittest.TestCase):
    """End-to-end test of scripts/collect_spec_decisions.main with mocked HTTP."""

    def setUp(self) -> None:
        # Import lazily so test discovery doesn't require the script's argparse
        # dependencies to be available at module load time.
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "collect_spec_decisions",
            _REPO_ROOT / "scripts" / "collect_spec_decisions.py",
        )
        assert spec is not None and spec.loader is not None
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)

    def _write_terminus_trace(self, root: Path) -> Path:
        trajectory = _build_terminus_trajectory(["ls -la\n", "pwd\n"])
        trace_path = root / "trajectory.json"
        trace_path.write_text(json.dumps(trajectory))
        return trace_path

    def test_full_run_writes_sidecar_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path = self._write_terminus_trace(tmp_path)
            sidecar_root = tmp_path / "spec_decisions"
            csv_root = tmp_path / "spec_decisions_csv"

            # First run: simulate matching first turn, mismatched second turn.
            responses = iter(
                [
                    DraftResult(
                        content=_terminus_response("ls -la\n"),
                        latency_ms=120.0,
                        prompt_tokens=100,
                        completion_tokens=10,
                    ),
                    DraftResult(
                        content=_terminus_response("git status\n"),
                        latency_ms=130.0,
                        prompt_tokens=110,
                        completion_tokens=12,
                    ),
                ]
            )

            def fake_complete(self_, messages, *, tools=None, tool_choice=None):  # noqa: ANN001 - test stub
                return next(responses)

            argv = [
                "collect_spec_decisions",
                "--trace",
                str(trace_path),
                "--agent",
                AGENT_TERMINUS,
                "--draft-tag",
                "deepseek-fast",
                "--draft-base-url",
                "http://example.invalid",
                "--draft-model",
                "deepseek-chat",
                "--sidecar-root",
                str(sidecar_root),
                "--csv-root",
                str(csv_root),
                "--run-id",
                "test-run",
                "--concurrency",
                "1",  # deterministic ordering for the responses iterator
            ]
            with patch.object(sys, "argv", argv), patch.object(
                self._module.OpenAICompatibleDraftClient, "complete", fake_complete
            ):
                exit_code = self._module.main()
            self.assertEqual(exit_code, 0)

            sidecar_path = resolve_sidecar_path(
                sidecar_root=sidecar_root,
                draft_tag="deepseek-fast",
                trace_path=trace_path,
            )
            self.assertTrue(sidecar_path.is_file())
            sidecar = load_sidecar(sidecar_path)
            self.assertEqual(sidecar.agent_kind, AGENT_TERMINUS)
            self.assertEqual(len(sidecar.turns), 2)
            by_index = {t.turn_index: t for t in sidecar.turns}
            for lvl in SCORE_LEVELS:
                self.assertTrue(by_index[0].accepted[lvl], f"turn 0 {lvl}")
                self.assertFalse(by_index[1].accepted[lvl], f"turn 1 {lvl}")
            self.assertIn("git status", by_index[1].draft_response_content)
            summary = sidecar.summary()
            self.assertEqual(summary["accept_counts"][LEVEL_LITERAL], 1)
            self.assertEqual(summary["accept_counts"][LEVEL_SEMANTIC], 1)

            csv_path = resolve_side_by_side_csv_path(
                csv_root=csv_root,
                draft_tag="deepseek-fast",
                run_id="test-run",
                trace_path=trace_path,
            )
            self.assertTrue(csv_path.is_file(), f"missing CSV: {csv_path}")
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csvmod.reader(handle))
            self.assertEqual(len(rows), 3)  # header + 2 turns
            self.assertEqual(
                rows[0],
                [
                    "turn",
                    "oracle",
                    "draft",
                    LEVEL_LITERAL,
                    LEVEL_NORMALIZED,
                    LEVEL_SEMANTIC,
                    "oracle_latency_ms",
                    "draft_latency_ms",
                    "speedup",
                    "error",
                ],
            )
            self.assertEqual(rows[1][3], "true")
            self.assertEqual(rows[2][3], "false")

            # Second run with --resume: should skip both already-covered turns.
            def boom(self_, messages, *, tools=None, tool_choice=None):  # noqa: ANN001
                raise AssertionError(
                    "client.complete must not be called when all turns are already covered"
                )

            argv_resume = list(argv) + ["--resume"]
            with patch.object(sys, "argv", argv_resume), patch.object(
                self._module.OpenAICompatibleDraftClient, "complete", boom
            ):
                exit_code = self._module.main()
            self.assertEqual(exit_code, 0)
            # Resume run should still rewrite the CSV with the same data.
            self.assertTrue(csv_path.is_file())

    def test_dataset_dispatch_resolves_trace_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trace_path = self._write_terminus_trace(tmp_path)
            dataset_path = tmp_path / "dataset.jsonl"
            row = {
                "agent_type": "terminus",
                "llm_service_type": "terminus_spec_trace_replay",
                "llm_service_config": {"trace_path": "trajectory.json"},
                "task_id": "synthetic",
            }
            dataset_path.write_text(json.dumps(row) + "\n")
            sidecar_root = tmp_path / "spec_decisions"

            def fake_complete(self_, messages, *, tools=None, tool_choice=None):  # noqa: ANN001
                return DraftResult(
                    content=_terminus_response("ls -la\n"),
                    latency_ms=10.0,
                    prompt_tokens=1,
                    completion_tokens=1,
                )

            argv = [
                "collect_spec_decisions",
                "--dataset",
                str(dataset_path),
                "--draft-tag",
                "deepseek-fast",
                "--draft-base-url",
                "http://example.invalid",
                "--draft-model",
                "deepseek-chat",
                "--sidecar-root",
                str(sidecar_root),
                "--concurrency",
                "1",
            ]
            with patch.object(sys, "argv", argv), patch.object(
                self._module.OpenAICompatibleDraftClient, "complete", fake_complete
            ):
                exit_code = self._module.main()
            self.assertEqual(exit_code, 0)
            sidecar_path = resolve_sidecar_path(
                sidecar_root=sidecar_root,
                draft_tag="deepseek-fast",
                trace_path=trace_path,
            )
            self.assertTrue(sidecar_path.is_file())


class BackfillScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "backfill_spec_oracle_latency",
            _REPO_ROOT / "scripts" / "backfill_spec_oracle_latency.py",
        )
        assert spec is not None and spec.loader is not None
        self._module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self._module)

    def test_backfill_fills_oracle_latency_and_rewrites_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Build a real terminus trajectory and a sidecar that lacks oracle
            # latency (simulating files written before this feature).
            trajectory = _build_terminus_trajectory(["a\n", "b\n", "c\n"])
            trace_path = tmp_path / "trajectory.json"
            trace_path.write_text(json.dumps(trajectory))

            sidecar = SpeculationSidecar(
                trace_path=str(trace_path),
                agent_kind=AGENT_TERMINUS,
                draft_model={"tag": "draftX", "name": "draftX"},
                turns=[
                    SpeculationTurn(
                        turn_index=i,
                        oracle_first_command=f"cmd_{i}\n",
                        draft_response_content="",
                        draft_first_command=f"cmd_{i}\n",
                        accepted={lvl: True for lvl in SCORE_LEVELS},
                        draft_latency_ms=500.0,
                        oracle_latency_ms=None,
                    )
                    for i in range(3)
                ],
            )
            sidecar_root = tmp_path / "sidecars"
            sidecar_path = resolve_sidecar_path(
                sidecar_root=sidecar_root,
                draft_tag="draftX",
                trace_path=trace_path,
            )
            write_sidecar(sidecar_path, sidecar)

            csv_root = tmp_path / "csv"
            argv = [
                "backfill_spec_oracle_latency",
                "--sidecar-root",
                str(sidecar_root),
                "--draft-tag",
                "draftX",
                "--csv-root",
                str(csv_root),
                "--run-id",
                "test-run",
            ]
            with patch.object(sys, "argv", argv):
                exit_code = self._module.main()
            self.assertEqual(exit_code, 0)

            # Sidecar now carries oracle_latency_ms (1000ms per turn).
            updated = load_sidecar(sidecar_path)
            for turn in updated.turns:
                self.assertAlmostEqual(turn.oracle_latency_ms, 1000.0, places=1)

            # CSV reflects the same data with new columns.
            csv_path = resolve_side_by_side_csv_path(
                csv_root=csv_root,
                draft_tag="draftX",
                run_id="test-run",
                trace_path=trace_path,
            )
            self.assertTrue(csv_path.is_file())
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csvmod.reader(handle))
            header = rows[0]
            self.assertIn("oracle_latency_ms", header)
            self.assertIn("draft_latency_ms", header)
            self.assertIn("speedup", header)
            ol_col = header.index("oracle_latency_ms")
            dl_col = header.index("draft_latency_ms")
            sp_col = header.index("speedup")
            for body_row in rows[1:]:
                self.assertEqual(body_row[ol_col], "1000.0")
                self.assertEqual(body_row[dl_col], "500.0")
                self.assertEqual(body_row[sp_col], "2.0000")

    def test_backfill_dry_run_does_not_modify_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            trajectory = _build_terminus_trajectory(["a\n"])
            trace_path = tmp_path / "trajectory.json"
            trace_path.write_text(json.dumps(trajectory))
            sidecar = SpeculationSidecar(
                trace_path=str(trace_path),
                agent_kind=AGENT_TERMINUS,
                draft_model={"tag": "draftX", "name": "draftX"},
                turns=[
                    SpeculationTurn(
                        turn_index=0,
                        oracle_first_command="a\n",
                        draft_response_content="",
                        draft_first_command="a\n",
                        accepted={lvl: True for lvl in SCORE_LEVELS},
                    )
                ],
            )
            sidecar_root = tmp_path / "sidecars"
            sidecar_path = resolve_sidecar_path(
                sidecar_root=sidecar_root,
                draft_tag="draftX",
                trace_path=trace_path,
            )
            write_sidecar(sidecar_path, sidecar)
            mtime_before = sidecar_path.stat().st_mtime_ns
            argv = [
                "backfill_spec_oracle_latency",
                "--sidecar-root",
                str(sidecar_root),
                "--draft-tag",
                "draftX",
                "--dry-run",
            ]
            with patch.object(sys, "argv", argv):
                exit_code = self._module.main()
            self.assertEqual(exit_code, 0)
            self.assertEqual(sidecar_path.stat().st_mtime_ns, mtime_before)
            unchanged = load_sidecar(sidecar_path)
            self.assertIsNone(unchanged.turns[0].oracle_latency_ms)


if __name__ == "__main__":
    unittest.main()

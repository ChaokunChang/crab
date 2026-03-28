from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from integrations.llm_services.mini_swe_trace_replay.service import TraceReplayLLMState, parse_replay_trace


def _trace_payload() -> dict[str, object]:
    return {
        "trajectory_format": "mini-swe-agent-1.1",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "THOUGHT: first\n<mswea_bash_command>echo first</mswea_bash_command>",
                "reasoning_content": "first reasoning",
                "extra": {"timestamp": 10.0},
            },
            {
                "role": "user",
                "content": "<returncode>0</returncode><output>first</output>",
                "extra": {"timestamp": 12.0},
            },
            {
                "role": "assistant",
                "content": "THOUGHT: second\n<mswea_bash_command>echo second</mswea_bash_command>",
                "extra": {"timestamp": 16.0},
            },
            {"role": "exit", "content": "submitted"},
        ],
    }


class MiniSWETraceReplayTests(unittest.TestCase):
    def test_parse_replay_trace_keeps_assistant_action_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.traj.json"
            trace_path.write_text(json.dumps(_trace_payload()), encoding="utf-8")

            parsed = parse_replay_trace(trace_path)

        self.assertEqual(parsed.trace_path, trace_path)
        self.assertEqual(len(parsed.assistant_messages), 2)
        self.assertIn("echo first", parsed.assistant_messages[0]["content"])
        self.assertIn("echo second", parsed.assistant_messages[1]["content"])
        self.assertIsNone(parsed.assistant_trace_delays_ms[0])
        self.assertEqual(parsed.assistant_trace_delays_ms[1], 4000.0)

    def test_state_advances_sequentially_reset_rewinds_and_restore_is_noop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.traj.json"
            trace_path.write_text(json.dumps(_trace_payload()), encoding="utf-8")
            state = TraceReplayLLMState(llm_service_config={"trace_path": str(trace_path)})

            first = state.handle_request(path="/v1/chat/completions", headers={}, payload={})
            snapshot_after_first = state.snapshot()
            state.restore(consumed_response_count=0)
            second = state.handle_request(path="/v1/chat/completions", headers={}, payload={})
            snapshot_after_second = state.snapshot()
            state.reset()
            replayed_first = state.handle_request(path="/v1/chat/completions", headers={}, payload={})

        self.assertIn("echo first", first["choices"][0]["message"]["content"])
        self.assertEqual(snapshot_after_first["trace_cursor"], 1)
        self.assertIn("echo second", second["choices"][0]["message"]["content"])
        self.assertEqual(snapshot_after_second["trace_cursor"], 2)
        self.assertTrue(snapshot_after_second["is_complete"])
        self.assertIn("echo first", replayed_first["choices"][0]["message"]["content"])

    def test_trace_replay_policy_uses_trace_timestamps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.traj.json"
            trace_path.write_text(json.dumps(_trace_payload()), encoding="utf-8")
            state = TraceReplayLLMState(
                llm_service_config={
                    "trace_path": str(trace_path),
                    "response_delay_policy": "trace_replay",
                    "response_delay_scaling_factor": 0.5,
                }
            )

            with patch("integrations.llm_services.mini_swe_trace_replay.service.time.sleep") as sleep:
                state.handle_request(path="/v1/chat/completions", headers={}, payload={})
                state.handle_request(path="/v1/chat/completions", headers={}, payload={})

        self.assertEqual(sleep.call_count, 1)
        sleep.assert_called_with(2.0)

    def test_invalid_response_delay_policy_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace_path = Path(tmp) / "trace.traj.json"
            trace_path.write_text(json.dumps(_trace_payload()), encoding="utf-8")

            with self.assertRaises(ValueError):
                TraceReplayLLMState(
                    llm_service_config={
                        "trace_path": str(trace_path),
                        "response_delay_policy": "nope",
                    }
                )


if __name__ == "__main__":
    unittest.main()

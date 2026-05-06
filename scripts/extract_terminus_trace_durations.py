"""Extract per-trace duration breakdown for the terminus_replay dataset.

The benchmark harness reports per-task wall-clock under various
`task_completion_ms` columns, but those measurements include checkpoint /
fork / verification overhead and are subject to the run-time
`max_agent_timeout_sec` cap. To compare a benchmark run against the
original trace, we need the trace's *own* end-to-end timing — and to
attribute that timing between the LLM (waiting on assistant responses) and
the tool side (running the agent's emitted commands).

This script walks each row of `results/datasets/terminus_replay.jsonl`,
opens the referenced `agent/trajectory.json`, and emits a CSV with:

* `task_id`              — task name (e.g. `adaptive-rejection-sampler`)
* `trial_id`             — trace trial id from the dataset row
* `model_label`          — model label from the dataset row
* `trace_response_count` — number of assistant turns in the trace
* `trace_reward`         — original trace reward (1.0 if it solved the task)
* `num_steps`            — total user+agent step count in the trajectory
* `num_agent_steps`      — number of agent (assistant) steps
* `trace_duration_s`     — wall-clock from first to last step timestamp
* `llm_time_s`           — per-turn wall-clock minus `tool_time_s`. Approximates
                           assistant-response latency (TTFT + decoding +
                           network) per turn.
* `tool_time_s`          — sum of every `tool_call.arguments.duration` hint
                           the agent supplied (terminus 2 protocol). Acts as
                           the tool-side floor: terminus may poll longer if
                           the terminal output is still streaming, so this
                           is a lower bound.
* `llm_fraction`         — `llm_time_s / (llm_time_s + tool_time_s)`
* `tool_fraction`        — `tool_time_s / (llm_time_s + tool_time_s)`

Timing decomposition rationale: the trajectory has exactly one `user` step
(the initial task prompt) followed by N `agent` steps; observations from
the previous turn's commands are embedded inside each agent step's
`observation` field rather than as standalone user rows. That means we
can't directly read tool-execution intervals from step timestamps. We
instead use the agent-provided per-tool `duration` hint as the tool-side
estimate and treat the residual turn-to-turn wall-clock as LLM latency.

Usage:
    python3 scripts/extract_terminus_trace_durations.py \\
        --dataset results/datasets/terminus_replay.jsonl \\
        --output  logs/terminus_trace_durations.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable


def _parse_timestamp(raw: str | None) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _resolve_trace_path(dataset_path: Path, raw_trace_path: str) -> Path:
    candidate = Path(raw_trace_path)
    if candidate.is_absolute():
        return candidate
    return (dataset_path.parent / candidate).resolve()


def _sum_tool_call_durations(step: dict[str, object]) -> float:
    total, _, _, _ = _split_tool_call_durations(step)
    return total


def _split_tool_call_durations(
    step: dict[str, object],
) -> tuple[float, float, float, bool]:
    """Return (total_dur, action_dur, wait_dur, has_action) for an agent step.

    `action_dur` sums durations for tool_calls whose `keystrokes` contain
    non-whitespace input — these are real shell commands the agent issued.
    `wait_dur` sums durations for tool_calls with empty/whitespace
    `keystrokes` — pure-wait polls (`{keystrokes: '', duration: 60}`),
    where speculation can never recover wall-time because the per-turn
    savings cap `min(draft_exec, oracle_wait)` collapses to ~0 (the
    "draft" is just a sleep). `has_action` is True iff at least one
    tool_call on this step was a real command — i.e., the step is an
    action turn rather than a wait turn.
    """
    total = 0.0
    action_dur = 0.0
    wait_dur = 0.0
    has_action = False
    tool_calls = step.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return total, action_dur, wait_dur, has_action
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        args = tc.get("arguments") or {}
        if not isinstance(args, dict):
            continue
        raw = args.get("duration")
        try:
            d = max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
        total += d
        keystrokes_raw = args.get("keystrokes")
        keystrokes = (
            keystrokes_raw if isinstance(keystrokes_raw, str) else ""
        ).strip()
        if keystrokes:
            action_dur += d
            has_action = True
        else:
            wait_dur += d
    return total, action_dur, wait_dur, has_action


def _summarize_trajectory(trajectory_path: Path) -> dict[str, float | int]:
    with trajectory_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    steps = payload.get("steps") or []
    timestamps = [_parse_timestamp(step.get("timestamp")) for step in steps]
    agent_count = sum(1 for step in steps if step.get("source") == "agent")
    if not steps or any(ts is None for ts in timestamps):
        return {
            "num_steps": len(steps),
            "num_agent_steps": agent_count,
            "trace_duration_s": 0.0,
            "llm_time_s": 0.0,
            "tool_time_s": 0.0,
        }

    duration_s = (timestamps[-1] - timestamps[0]).total_seconds()
    # Tool-time floor: every agent step's tool_calls list its expected
    # duration (in seconds) per command. terminus 2 may poll for longer
    # if output is still streaming, but never less, so summing these gives
    # a conservative lower bound on tool-side wall-clock.
    #
    # We also split that total into action / wait based on whether each
    # tool_call carried real keystrokes. Action-tool-time is what
    # speculation can actually overlap with the next turn's LLM round-
    # trip; wait-tool-time is pure-sleep polling that pays nothing per
    # accepted turn.
    tool_time_s = 0.0
    action_tool_time_s = 0.0
    wait_tool_time_s = 0.0
    action_turns = 0
    wait_turns = 0
    # Largest single-command duration across action tool_calls. Captures
    # long-running `pip install`, `make all`, `apt-get install -y …`
    # etc. that exceed the per-LLM-window timescale on which the host
    # inspector reports `process_changed`. During such a command the
    # inspector typically reports process_changed=False (the long-running
    # process is one PID with no new spawns, soft-dirty pagemap stays
    # quiet while it waits on IO, and no `(deleted)` mmap suffixes
    # appear); the scheduler then elects an fs-only checkpoint, which
    # `_SpecForkManager.ensure_fork` correctly refuses to base a fork
    # on (older process image + newer fs snapshot composes badly post-
    # `apt-get install`/`dpkg`). So traces with even a few > ~60s
    # action commands hit the fs-only-skip guard repeatedly and spec
    # never fires.
    max_action_command_duration_s = 0.0
    for step in steps:
        if step.get("source") != "agent":
            tool_time_s += _sum_tool_call_durations(step)
            continue
        total_dur, action_dur, wait_dur, has_action = _split_tool_call_durations(step)
        tool_time_s += total_dur
        action_tool_time_s += action_dur
        wait_tool_time_s += wait_dur
        if has_action:
            action_turns += 1
        elif total_dur > 0 or step.get("tool_calls"):
            # All tool_calls on this step had empty keystrokes — pure-wait turn.
            wait_turns += 1
        # Track the longest single-command duration on action turns.
        for tc in step.get("tool_calls") or []:
            if not isinstance(tc, dict):
                continue
            args = tc.get("arguments") or {}
            if not isinstance(args, dict):
                continue
            keystrokes = args.get("keystrokes")
            if not isinstance(keystrokes, str) or not keystrokes.strip():
                continue
            try:
                d = max(0.0, float(args.get("duration") or 0.0))
            except (TypeError, ValueError):
                continue
            if d > max_action_command_duration_s:
                max_action_command_duration_s = d
    # LLM-time estimate: the residual of turn-to-turn wall-clock after
    # subtracting the tool-side floor. This conflates network round-trip
    # and any tool overrun beyond the agent's hint, but for traces with
    # honest hints it tracks LLM inference latency closely.
    total_gap_s = (timestamps[-1] - timestamps[0]).total_seconds()
    llm_time_s = max(0.0, total_gap_s - tool_time_s)
    return {
        "num_steps": len(steps),
        "num_agent_steps": agent_count,
        "trace_duration_s": duration_s,
        "llm_time_s": llm_time_s,
        "tool_time_s": tool_time_s,
        "action_turns": action_turns,
        "wait_turns": wait_turns,
        "action_tool_time_s": action_tool_time_s,
        "wait_tool_time_s": wait_tool_time_s,
        "max_action_command_duration_s": max_action_command_duration_s,
    }


def _iter_dataset_rows(dataset_path: Path) -> Iterable[dict[str, object]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"failed to parse {dataset_path}:{line_number}: {exc}"
                ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="path to terminus_replay.jsonl (one task per line)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="output CSV path",
    )
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(f"dataset file not found: {args.dataset}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task_id",
        "trial_id",
        "model_label",
        "trace_response_count",
        "trace_reward",
        "num_steps",
        "num_agent_steps",
        "trace_duration_s",
        "llm_time_s",
        "tool_time_s",
        "action_turns",
        "wait_turns",
        "action_tool_time_s",
        "wait_tool_time_s",
        "max_action_command_duration_s",
        "llm_fraction",
        "tool_fraction",
        "trace_path",
    ]
    rows_written = 0
    skipped = 0
    with args.output.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fieldnames)
        writer.writeheader()
        for row in _iter_dataset_rows(args.dataset):
            llm_service_config = row.get("llm_service_config") or {}
            raw_trace_path = llm_service_config.get("trace_path")
            if not isinstance(raw_trace_path, str) or not raw_trace_path:
                skipped += 1
                continue
            trace_path = _resolve_trace_path(args.dataset, raw_trace_path)
            if not trace_path.is_file():
                skipped += 1
                print(f"warning: trace file not found, skipping: {trace_path}")
                continue
            try:
                summary = _summarize_trajectory(trace_path)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                print(f"warning: failed to summarize {trace_path}: {exc}")
                continue
            llm_time = float(summary["llm_time_s"])
            tool_time = float(summary["tool_time_s"])
            denom = llm_time + tool_time
            llm_fraction = llm_time / denom if denom > 0 else 0.0
            tool_fraction = tool_time / denom if denom > 0 else 0.0
            writer.writerow(
                {
                    "task_id": row.get("task_id", ""),
                    "trial_id": row.get("trace_trial_id", ""),
                    "model_label": row.get("trace_model_label", ""),
                    "trace_response_count": row.get("trace_response_count", ""),
                    "trace_reward": row.get("trace_reward", ""),
                    "num_steps": summary["num_steps"],
                    "num_agent_steps": summary["num_agent_steps"],
                    "trace_duration_s": f"{summary['trace_duration_s']:.3f}",
                    "llm_time_s": f"{llm_time:.3f}",
                    "tool_time_s": f"{tool_time:.3f}",
                    "action_turns": summary.get("action_turns", 0),
                    "wait_turns": summary.get("wait_turns", 0),
                    "action_tool_time_s": f"{summary.get('action_tool_time_s', 0.0):.3f}",
                    "wait_tool_time_s": f"{summary.get('wait_tool_time_s', 0.0):.3f}",
                    "max_action_command_duration_s": f"{summary.get('max_action_command_duration_s', 0.0):.3f}",
                    "llm_fraction": f"{llm_fraction:.4f}",
                    "tool_fraction": f"{tool_fraction:.4f}",
                    "trace_path": str(trace_path),
                }
            )
            rows_written += 1
    print(f"wrote {rows_written} rows to {args.output} ({skipped} skipped)")


if __name__ == "__main__":
    main()

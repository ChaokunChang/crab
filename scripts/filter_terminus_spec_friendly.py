"""Filter the terminus replay dataset down to traces that should benefit
the most from speculative execution.

Speculation's wall-clock saving per accepted turn is bounded by
`min(draft_exec_duration, oracle_wait_duration)` — the overlap window
between the next turn's command execution and that turn's LLM round
trip. So for a trace to actually win wall-time from spec it needs:

* **Action turns, not wait turns.** A pure-wait turn has empty
  keystrokes and does no real work — `draft_exec ≈ 0`, so even if the
  spec service "accepts" the turn the saving collapses to zero. Worse,
  a build-heavy trace with many wait turns inflates the tool_time_s
  column without contributing to recoverable savings, fooling
  per-turn-floor predicates that don't distinguish the two.
* **Substantial wall-clock on BOTH sides of each action turn.** If
  per-action LLM time is high but per-action tool time is tiny, the
  overlap is bounded by the tiny side; same in reverse. The traces that
  spec helps most have roughly balanced action-tool and per-action LLM
  time.
* **Enough action turns to amortize per-fork CRIU/restore overhead.**
* **A duration cap so the benchmark stays practical to iterate on.**

This is a successor to the older filter that used raw `tool_time_s`
and `min(avg_llm/turn, avg_tool/turn)` — those metrics are inflated by
wait turns and produced a "spec_friendly" set whose build-heavy traces
saw 0–16% acceptance and no wall-clock improvement vs the baseline
replay (see the nofault-vs-spec spec_friendly comparison in the
2026-04-28 run pair).

Inputs:

* `--input-dataset` — the source jsonl (typically the full
  `results/datasets/terminus_replay.jsonl`, but the older
  `terminus_replay_tool_dominated.jsonl` is also accepted via the
  legacy `--tool-dominated-dataset` alias).
* `--trace-stats` — the timing CSV produced by
  `extract_terminus_trace_durations.py`. Must include the action /
  wait breakdown columns added in the 2026-04-28 update; older CSVs
  silently behave as if every turn were an action turn (which makes
  the wait-fraction predicate a no-op).
* `--output` — destination jsonl.

Usage:
    python3 scripts/filter_terminus_spec_friendly.py \\
        --input-dataset results/datasets/terminus_replay.jsonl \\
        --trace-stats logs/terminus_trace_durations.csv \\
        --output results/datasets/terminus_replay_spec_friendly.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.generate_terminus_replay_dataset import (  # noqa: E402
    _DEFAULT_EXCLUDED_TRIAL_DIRS,
)

# `send_keys` clamps each command's `min_timeout_sec` to 60 s in replay
# (matches upstream Terminus2's `_handle_llm_interaction` cap). Per-turn
# averages should reflect what spec can ACTUALLY overlap, so we cap each
# tool-call duration at this value before averaging.
_PER_COMMAND_DURATION_CAP_S = 60.0


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


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value)) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _load_trace_stats(stats_path: Path) -> dict[str, dict[str, float]]:
    """trial_id -> normalized timing stats (seconds + counts).

    Backwards compatible with old CSVs that lack the action/wait
    breakdown columns: when the columns are missing, treat every turn
    as an action turn (so per-action averages collapse to per-turn
    averages, and the wait-fraction predicate becomes a no-op).
    """
    out: dict[str, dict[str, float]] = {}
    with stats_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            trial = (row.get("trial_id") or "").strip()
            if not trial:
                continue
            n = max(1, _safe_int(row.get("num_agent_steps")))
            llm = _safe_float(row.get("llm_time_s"))
            tool = _safe_float(row.get("tool_time_s"))
            dur = _safe_float(row.get("trace_duration_s"))
            action_turns = _safe_int(row.get("action_turns"), default=n)
            wait_turns = _safe_int(row.get("wait_turns"), default=0)
            action_tool = _safe_float(
                row.get("action_tool_time_s"), default=tool
            )
            max_action_cmd = _safe_float(row.get("max_action_command_duration_s"))
            # Per-action averages — what spec can actually overlap on a
            # turn the agent will accept.
            action_turns_safe = max(1, action_turns)
            avg_llm_per_action = llm / action_turns_safe
            avg_tool_per_action = action_tool / action_turns_safe
            avg_tool_per_action_capped = min(
                avg_tool_per_action, _PER_COMMAND_DURATION_CAP_S
            )
            min_per_action_s = min(avg_llm_per_action, avg_tool_per_action_capped)
            wait_fraction = wait_turns / max(1, action_turns + wait_turns)
            # Upper bound on total saved-time at the default
            # acceptance_rate=0.5: each accepted action turn saves
            # min(per-turn LLM, per-turn capped action-tool); wait
            # turns contribute zero.
            expected_saved_s = (
                0.5 * action_turns * min_per_action_s
            )
            out[trial] = {
                "num_agent_steps": float(n),
                "action_turns": float(action_turns),
                "wait_turns": float(wait_turns),
                "wait_fraction": wait_fraction,
                "llm_time_s": llm,
                "tool_time_s": tool,
                "action_tool_time_s": action_tool,
                "max_action_command_duration_s": max_action_cmd,
                "trace_duration_s": dur,
                "avg_llm_per_action_s": avg_llm_per_action,
                "avg_action_tool_per_turn_s": avg_tool_per_action,
                "avg_action_tool_per_turn_capped_s": avg_tool_per_action_capped,
                "min_per_action_s": min_per_action_s,
                "expected_saved_s": expected_saved_s,
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-dataset",
        type=Path,
        help="Source dataset jsonl (typically the full terminus_replay.jsonl).",
    )
    input_group.add_argument(
        "--tool-dominated-dataset",
        type=Path,
        help=(
            "Legacy alias for --input-dataset; kept so existing pipeline "
            "scripts that point at terminus_replay_tool_dominated.jsonl "
            "still work."
        ),
    )
    parser.add_argument("--trace-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-action-turns",
        type=int,
        default=8,
        help=(
            "Reject traces with fewer ACTION turns than this (default 8). "
            "Action turns are agent steps with at least one non-empty "
            "keystrokes tool_call; wait turns are excluded. Spec needs "
            "enough action turns to amortize per-fork CRIU restore."
        ),
    )
    parser.add_argument(
        "--max-wait-turn-fraction",
        type=float,
        default=0.3,
        help=(
            "Reject traces where wait_turns / (action_turns + wait_turns) "
            "exceeds this (default 0.3). Wait-dominated traces (build "
            "loops with many `keystrokes='', duration=60` polls) cannot "
            "produce wall-time wins from spec — the per-turn savings cap "
            "min(draft_exec, oracle_wait) collapses to ~0 on a wait turn."
        ),
    )
    parser.add_argument(
        "--min-per-action-s",
        type=float,
        default=8.0,
        help=(
            "Reject traces whose min(avg_llm_per_action, "
            "avg_action_tool_per_turn_capped) is below this (default 8s). "
            "This is the per-accepted-action-turn saving cap, capped at "
            f"{_PER_COMMAND_DURATION_CAP_S:.0f}s per command to mirror "
            "send_keys's clamp."
        ),
    )
    parser.add_argument(
        "--min-expected-saved-s",
        type=float,
        default=60.0,
        help=(
            "Reject traces whose expected total saved time is below this. "
            "Computed as 0.5 * action_turns * min_per_action_s (i.e. the "
            "default acceptance_rate=0.5 applied to the per-action "
            "saving cap). Default 60s keeps traces where spec can "
            "plausibly recover ≥1 minute of wall-clock."
        ),
    )
    parser.add_argument(
        "--max-trace-duration-s",
        type=float,
        default=1500.0,
        help="Upper bound on trace_duration_s for benchmark practicality.",
    )
    parser.add_argument(
        "--max-action-command-duration-s",
        type=float,
        default=60.0,
        help=(
            "Reject traces where any single action command's duration "
            "exceeds this (default 60s = the send_keys per-command cap). "
            "A long-running command like `pip install`, `make all`, "
            "`apt-get install …` keeps the inspector's process_changed "
            "signal quiet for tens of seconds; the scheduler then elects "
            "fs-only checkpoints, which `_SpecForkManager.ensure_fork` "
            "refuses to base a fork on. Traces with even a few > ~60s "
            "actions hit that fs-only-skip guard repeatedly and spec "
            "never fires (see logs/TODO_inspector_short_lived_process_signal.md "
            "for the underlying inspector limitation). Setting this knob "
            "below the per-command cap excludes those traces. Set to a "
            "large number (e.g. 9999) to disable."
        ),
    )
    parser.add_argument(
        "--min-llm-fraction",
        type=float,
        default=0.0,
        help=(
            "Reject traces where avg_llm_per_action / "
            "(avg_llm_per_action + avg_action_tool_per_turn_capped) is "
            "below this. Default 0.0 disables (we only need the per-turn "
            "min, balance is a soft preference). Set to e.g. 0.3 to "
            "exclude purely-tool-bound traces."
        ),
    )
    parser.add_argument(
        "--max-llm-fraction",
        type=float,
        default=1.0,
        help=(
            "Reject traces where avg_llm_per_action / "
            "(avg_llm_per_action + avg_action_tool_per_turn_capped) is "
            "above this. Default 1.0 disables; set to e.g. 0.85 to "
            "exclude LLM-bound traces with no real tool work to "
            "speculate."
        ),
    )
    parser.add_argument(
        "--exclude-trial-dir",
        action="append",
        dest="exclude_trial_dirs",
        default=None,
        help=(
            "Skip the given trace_trial_dir values. Augments the built-in "
            "known-broken-trace blacklist; pass --no-default-excludes to "
            "disable the built-in blacklist."
        ),
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        default=False,
        help="Disable the built-in known-broken-trace trial-dir blacklist.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help=(
            "Optional debug CSV: emit per-trace metrics + accept/reject "
            "decision so the predicate cutoffs can be tuned without "
            "re-running the filter."
        ),
    )
    parser.add_argument(
        "--max-per-task-id",
        type=int,
        default=3,
        help=(
            "Cap the number of kept traces per task_id (default 3). The "
            "harness samples sandboxes by row order, so without a cap a "
            "10-sandbox run lands all sandboxes on the alphabetically "
            "first 1-2 tasks — typical 'spec-unfriendly' build-heavy "
            "ones. Capping forces diversity. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--sort-by",
        choices=("expected_saved_s", "min_per_action_s", "input_order"),
        default="expected_saved_s",
        help=(
            "Sort kept traces by this metric, descending (default "
            "expected_saved_s). The harness picks rows in order, so "
            "putting the most-spec-friendly traces first directly raises "
            "the spec accept rate of small N-sandbox runs."
        ),
    )
    args = parser.parse_args()

    input_dataset = args.input_dataset or args.tool_dominated_dataset
    if not input_dataset.is_file():
        raise SystemExit(f"dataset not found: {input_dataset}")
    if not args.trace_stats.is_file():
        raise SystemExit(f"trace stats not found: {args.trace_stats}")

    stats = _load_trace_stats(args.trace_stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    user_excluded_dirs = set(args.exclude_trial_dirs or ())
    if args.no_default_excludes:
        excluded_trial_dirs = user_excluded_dirs
    else:
        excluded_trial_dirs = user_excluded_dirs | set(_DEFAULT_EXCLUDED_TRIAL_DIRS)

    counts = {"total": 0, "blacklisted_trial_dir": 0, "missing_stats": 0, "kept": 0}
    rejections: dict[str, int] = {}
    summary_rows: list[dict[str, object]] = []
    # Two-pass: first collect kept rows in memory so we can sort + cap by
    # task before emitting. The dataset is small (a few thousand rows
    # max), so this is fine.
    kept_entries: list[tuple[float, dict[str, object], dict[str, float]]] = []
    for row in _iter_dataset_rows(input_dataset):
        counts["total"] += 1
        trial_dir = str(row.get("trace_trial_dir") or "").strip()
        trial = str(row.get("trace_trial_id") or "").strip()
        if trial_dir in excluded_trial_dirs:
            counts["blacklisted_trial_dir"] += 1
            continue
        entry = stats.get(trial)
        if entry is None:
            counts["missing_stats"] += 1
            continue
        llm_per = entry["avg_llm_per_action_s"]
        tool_per_capped = entry["avg_action_tool_per_turn_capped_s"]
        denom = llm_per + tool_per_capped
        llm_balance = (llm_per / denom) if denom > 0 else 0.0
        decision = "kept"
        if entry["action_turns"] < args.min_action_turns:
            rejections["action_turns"] = rejections.get("action_turns", 0) + 1
            decision = "rej:action_turns"
        elif entry["wait_fraction"] > args.max_wait_turn_fraction:
            rejections["wait_fraction"] = rejections.get("wait_fraction", 0) + 1
            decision = "rej:wait_fraction"
        elif entry["min_per_action_s"] < args.min_per_action_s:
            rejections["min_per_action_s"] = (
                rejections.get("min_per_action_s", 0) + 1
            )
            decision = "rej:min_per_action_s"
        elif entry["expected_saved_s"] < args.min_expected_saved_s:
            rejections["expected_saved_s"] = (
                rejections.get("expected_saved_s", 0) + 1
            )
            decision = "rej:expected_saved_s"
        elif entry["trace_duration_s"] > args.max_trace_duration_s:
            rejections["trace_duration_s"] = (
                rejections.get("trace_duration_s", 0) + 1
            )
            decision = "rej:trace_duration_s"
        elif entry["max_action_command_duration_s"] > args.max_action_command_duration_s:
            rejections["max_action_command_duration_s"] = (
                rejections.get("max_action_command_duration_s", 0) + 1
            )
            decision = "rej:max_action_command_duration_s"
        elif llm_balance < args.min_llm_fraction:
            rejections["llm_balance_low"] = (
                rejections.get("llm_balance_low", 0) + 1
            )
            decision = "rej:llm_balance_low"
        elif llm_balance > args.max_llm_fraction:
            rejections["llm_balance_high"] = (
                rejections.get("llm_balance_high", 0) + 1
            )
            decision = "rej:llm_balance_high"
        if decision == "kept":
            sort_key_value = entry.get(args.sort_by, 0.0) if args.sort_by != "input_order" else float(counts["total"])
            kept_entries.append((float(sort_key_value), row, entry))
        if args.summary_csv is not None:
            summary_rows.append(
                {
                    "decision": decision,
                    "task_id": row.get("task_id", ""),
                    "trial_id": trial,
                    "trial_dir": trial_dir,
                    "action_turns": int(entry["action_turns"]),
                    "wait_turns": int(entry["wait_turns"]),
                    "wait_fraction": f"{entry['wait_fraction']:.3f}",
                    "trace_duration_s": f"{entry['trace_duration_s']:.1f}",
                    "avg_llm_per_action_s": f"{llm_per:.2f}",
                    "avg_action_tool_per_turn_capped_s": f"{tool_per_capped:.2f}",
                    "min_per_action_s": f"{entry['min_per_action_s']:.2f}",
                    "max_action_command_duration_s": f"{entry['max_action_command_duration_s']:.1f}",
                    "expected_saved_s": f"{entry['expected_saved_s']:.1f}",
                    "llm_balance": f"{llm_balance:.3f}",
                }
            )

    # Sort by chosen metric, descending (input_order: stable, ascending).
    if args.sort_by == "input_order":
        kept_entries.sort(key=lambda t: t[0])
    else:
        kept_entries.sort(key=lambda t: -t[0])

    # Cap per-task to keep small N-sandbox runs diverse.
    per_task_count: dict[str, int] = {}
    final_entries: list[dict[str, object]] = []
    capped_count = 0
    for _key, row, _entry in kept_entries:
        task_id = str(row.get("task_id") or "").strip()
        if args.max_per_task_id > 0:
            cur = per_task_count.get(task_id, 0)
            if cur >= args.max_per_task_id:
                capped_count += 1
                continue
            per_task_count[task_id] = cur + 1
        final_entries.append(row)
    counts["kept"] = len(final_entries)

    with args.output.open("w", encoding="utf-8") as out:
        for row in final_entries:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"input rows:           {counts['total']}")
    print(f"blacklisted dirs:     {counts['blacklisted_trial_dir']}")
    print(f"missing trace stats:  {counts['missing_stats']}")
    print(f"kept (spec-friendly): {counts['kept']} -> {args.output}")
    if capped_count:
        print(f"capped per task_id:   {capped_count} (max={args.max_per_task_id})")
    if rejections:
        print("rejections by predicate:")
        for k, v in sorted(rejections.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v}")
    if args.sort_by != "input_order":
        print(f"sort order:           {args.sort_by} desc")

    if args.summary_csv is not None and summary_rows:
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"per-trace summary -> {args.summary_csv}")


if __name__ == "__main__":
    main()

"""Filter the terminus_replay benchmark dataset by per-trace timing stats.

Reads the per-trace timing CSV produced by
`scripts/extract_terminus_trace_durations.py`, joins it with the original
benchmark dataset on `trace_trial_id`, and writes filtered subsets that
target specific evaluation goals.

Built-in subsets (each maps to a separate `--output-*` flag):

* tool-dominated (`--output-tool-dominated`):
  rows where `llm_fraction < 0.5`, i.e. the trace spent more than half of
  its wall-clock running shell commands. These are the most promising
  workloads for speculative execution because there is genuine tool-side
  work to overlap with the next-turn LLM round-trip.

* tool-dominated short (`--output-tool-dominated-short`):
  same as above plus `trace_duration_s < 300` so the benchmark finishes
  quickly enough for iterative tuning.

Usage:
    python3 scripts/filter_terminus_replay_dataset.py \\
        --dataset results/datasets/terminus_replay.jsonl \\
        --trace-stats logs/terminus_trace_durations.csv \\
        --output-tool-dominated results/datasets/terminus_replay_tool_dominated.jsonl \\
        --output-tool-dominated-short results/datasets/terminus_replay_tool_dominated_short.jsonl
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


def _iter_dataset_rows(dataset_path: Path) -> Iterable[tuple[int, dict[str, object]]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"failed to parse {dataset_path}:{line_number}: {exc}"
                ) from exc


def _load_trace_stats(stats_path: Path) -> dict[str, dict[str, float]]:
    """trial_id -> {llm_fraction, tool_fraction, trace_duration_s}."""
    stats: dict[str, dict[str, float]] = {}
    with stats_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trial = (row.get("trial_id") or "").strip()
            if not trial:
                continue
            try:
                stats[trial] = {
                    "llm_fraction": float(row.get("llm_fraction") or 0.0),
                    "tool_fraction": float(row.get("tool_fraction") or 0.0),
                    "trace_duration_s": float(row.get("trace_duration_s") or 0.0),
                }
            except ValueError:
                continue
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace-stats", type=Path, required=True)
    parser.add_argument(
        "--output-tool-dominated",
        type=Path,
        required=True,
        help="rows with llm_fraction < 0.5",
    )
    parser.add_argument(
        "--output-tool-dominated-short",
        type=Path,
        required=True,
        help="rows with llm_fraction < 0.5 AND trace_duration_s < 300",
    )
    parser.add_argument(
        "--llm-fraction-threshold",
        type=float,
        default=0.5,
        help="upper bound on llm_fraction for tool-dominated subsets (default 0.5)",
    )
    parser.add_argument(
        "--short-trace-duration-s",
        type=float,
        default=300.0,
        help="upper bound on trace_duration_s for the tool-dominated-short subset (default 300)",
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
    args = parser.parse_args()

    if not args.dataset.is_file():
        raise SystemExit(f"dataset not found: {args.dataset}")
    if not args.trace_stats.is_file():
        raise SystemExit(f"trace stats not found: {args.trace_stats}")

    stats = _load_trace_stats(args.trace_stats)

    user_excluded_dirs = set(args.exclude_trial_dirs or ())
    if args.no_default_excludes:
        excluded_trial_dirs = user_excluded_dirs
    else:
        excluded_trial_dirs = user_excluded_dirs | set(_DEFAULT_EXCLUDED_TRIAL_DIRS)

    args.output_tool_dominated.parent.mkdir(parents=True, exist_ok=True)
    args.output_tool_dominated_short.parent.mkdir(parents=True, exist_ok=True)

    counts = {
        "total": 0,
        "missing_stats": 0,
        "blacklisted_trial_dir": 0,
        "tool_dom": 0,
        "tool_dom_short": 0,
    }
    with (
        args.output_tool_dominated.open("w", encoding="utf-8") as out_tool,
        args.output_tool_dominated_short.open("w", encoding="utf-8") as out_short,
    ):
        for _, row in _iter_dataset_rows(args.dataset):
            counts["total"] += 1
            trial_dir = str(row.get("trace_trial_dir") or "").strip()
            if trial_dir in excluded_trial_dirs:
                counts["blacklisted_trial_dir"] += 1
                continue
            trial = str(row.get("trace_trial_id") or "").strip()
            entry = stats.get(trial)
            if entry is None:
                counts["missing_stats"] += 1
                continue
            if entry["llm_fraction"] >= args.llm_fraction_threshold:
                continue
            line = json.dumps(row, ensure_ascii=False) + "\n"
            out_tool.write(line)
            counts["tool_dom"] += 1
            if entry["trace_duration_s"] < args.short_trace_duration_s:
                out_short.write(line)
                counts["tool_dom_short"] += 1

    print(f"input rows:           {counts['total']}")
    print(f"blacklisted dirs:     {counts['blacklisted_trial_dir']}")
    print(f"missing trace stats:  {counts['missing_stats']}")
    print(
        f"tool-dominated rows:  {counts['tool_dom']} "
        f"-> {args.output_tool_dominated}"
    )
    print(
        f"tool-dominated short: {counts['tool_dom_short']} "
        f"-> {args.output_tool_dominated_short}"
    )


if __name__ == "__main__":
    main()

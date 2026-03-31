#!/usr/bin/env python3
"""Download Claude Code trajectories from HuggingFace and save as per-trial trace files."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Claude Code traces from HuggingFace")
    parser.add_argument(
        "--dataset",
        default="yoonholee/terminalbench-trajectories",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--agent",
        default="claude-code",
        help="Agent name to filter",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6@anthropic",
        help="Model name to filter",
    )
    parser.add_argument(
        "--traces-root",
        type=Path,
        default=ROOT / "results" / "traces" / "claude-code",
        help="Output directory for trace files",
    )
    return parser.parse_args()


def download_and_save(
    *,
    dataset_id: str,
    agent: str,
    model: str,
    traces_root: Path,
) -> None:
    from datasets import load_dataset

    print(f"Loading dataset {dataset_id} ...")
    ds = load_dataset(dataset_id, split="train")
    filtered = ds.filter(lambda x: x["agent"] == agent and x["model"] == model)
    print(f"Filtered to {len(filtered)} rows (agent={agent!r}, model={model!r})")

    if len(filtered) == 0:
        print("No matching rows found. Exiting.")
        sys.exit(1)

    by_task: dict[str, list] = defaultdict(list)
    for row in filtered:
        by_task[row["task_name"]].append(row)

    # Sort trials within each task for deterministic ordering
    for task_name in by_task:
        by_task[task_name].sort(key=lambda r: r["trial_name"])

    total_written = 0
    for task_name, rows in sorted(by_task.items()):
        task_dir = traces_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        for trial_index, row in enumerate(rows):
            steps = json.loads(row["steps"]) if row["steps"] else []
            trace = {
                "trajectory_format": "claude-code-1.0",
                "task_name": task_name,
                "agent": row["agent"],
                "model": row["model"],
                "trial_index": trial_index,
                "trial_name": row["trial_name"],
                "reward": row["reward"],
                "duration_seconds": row["duration_seconds"],
                "steps": steps,
            }
            trace_path = task_dir / f"{task_name}.trial-{trial_index}.trace.json"
            trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
            total_written += 1

    print(f"Wrote {total_written} trace files across {len(by_task)} tasks to {traces_root}")


def main() -> None:
    args = parse_args()
    download_and_save(
        dataset_id=args.dataset,
        agent=args.agent,
        model=args.model,
        traces_root=args.traces_root.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()

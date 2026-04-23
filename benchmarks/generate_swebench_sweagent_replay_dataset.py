#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.llm_services.mini_swe_trace_replay.service import parse_replay_trace
from integrations.sandboxes import swebench as swebench_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the SWE-Bench + MiniSwe replay benchmark dataset JSONL")
    parser.add_argument(
        "--traces-root",
        type=Path,
        default=ROOT / "results" / "traces" / "mini-swe-agent" / "runs" / "minimax_verified_test_seed42_n200_resolved_128",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "datasets" / "swebench_miniswe_replay.jsonl",
    )
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=ROOT / "results" / "datasets" / "swebench_sweagent_replay_tasks",
    )
    return parser.parse_args()


def _relative_path(path: Path, *, output_path: Path) -> str:
    return os.path.relpath(path.resolve(), start=output_path.resolve().parent)


def _trace_path_for_instance(traces_root: Path, instance_id: str) -> Path:
    path = traces_root / instance_id / f"{instance_id}.traj.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing mini-swe replay trace for {instance_id}: {path}")
    return path


def build_dataset_row(
    *,
    instance: dict[str, object],
    trace_path: Path,
    task_root: Path,
    output_path: Path,
) -> dict[str, object]:
    instance_id = str(instance["instance_id"])
    trace = parse_replay_trace(trace_path)
    image_ref = swebench_support.remote_instance_image_ref(instance_id)
    swebench_support.write_task_assets(task_root=task_root, instance=instance, image_ref=image_ref)
    return {
        "task_id": instance_id,
        "agent_type": "mini_swe",
        "llm_service_type": "mini_swe_trace_replay",
        "task_description": {"prompt": str(instance["problem_statement"])},
        "task_config": {
            "options": {
                "task_id": instance_id,
                "swebench_instance_id": instance_id,
                "swebench_repo": str(instance["repo"]),
                "swebench_version": str(instance["version"]),
                "swebench_base_commit": str(instance["base_commit"]),
                "max_agent_timeout_sec": swebench_support.DEFAULT_AGENT_TIMEOUT_S,
                "max_test_timeout_sec": swebench_support.DEFAULT_TEST_TIMEOUT_S,
            }
        },
        "docker_compose_file": _relative_path(task_root / "docker-compose.yaml", output_path=output_path),
        "service_name": "client",
        "task_root": _relative_path(task_root, output_path=output_path),
        "llm_service_config": {
            "trace_path": _relative_path(trace.trace_path, output_path=output_path),
        },
        "trace_response_count": len(trace.assistant_messages),
        "trace_malformed_line_count": 0,
    }


def generate_dataset(*, traces_root: Path, tasks_root: Path, output_path: Path) -> list[dict[str, object]]:
    rows_by_instance = swebench_support.load_verified_dataset_rows()
    trace_dirs = sorted(path for path in traces_root.iterdir() if path.is_dir())
    dataset_rows: list[dict[str, object]] = []
    for trace_dir in trace_dirs:
        instance_id = trace_dir.name
        instance = rows_by_instance.get(instance_id)
        if instance is None:
            print(f"{instance_id} missing from cached SWE-Bench Verified dataset, skipping", file=sys.stderr)
            continue
        try:
            dataset_rows.append(
                build_dataset_row(
                    instance=instance,
                    trace_path=_trace_path_for_instance(traces_root, instance_id),
                    task_root=tasks_root / instance_id,
                    output_path=output_path,
                )
            )
        except Exception as exc:
            print(f"{instance_id} can not be parsed, skipped with error {exc}", file=sys.stderr)
    return dataset_rows


def write_dataset(rows: list[dict[str, object]], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_path = args.out.expanduser().resolve()
    rows = generate_dataset(
        traces_root=args.traces_root.expanduser().resolve(),
        tasks_root=args.tasks_root.expanduser().resolve(),
        output_path=output_path,
    )
    write_dataset(rows, output_path=output_path)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

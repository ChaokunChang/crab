#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.llm_services.iflow_trace_replay.service import parse_replay_trace


_TRACE_RESPONSE_LINE_RE = re.compile(r'"type"\s*:\s*"response"')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Termnius + IFlow replay benchmark dataset JSONL")
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=ROOT / "results" / "original-tasks",
    )
    parser.add_argument(
        "--traces-root",
        type=Path,
        default=ROOT / "results" / "2026-02-24__20-20-40_passed",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "datasets" / "termnius_iflow_replay.jsonl",
    )
    return parser.parse_args()


def _relative_path(path: Path, *, output_path: Path) -> str:
    return os.path.relpath(path.resolve(), start=output_path.resolve().parent)


def load_task_yaml(task_yaml: Path) -> dict[str, object]:
    payload = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"task yaml must be an object: {task_yaml}")
    return payload


def resolve_trace_log(task_trace_root: Path) -> Path:
    matches = sorted(task_trace_root.glob("*/agent-logs/proxy_server_trajectory.log"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one replay trace for {task_trace_root}, found {len(matches)}")
    return matches[0]


def resolve_service_name(compose_file: Path) -> str:
    payload = yaml.safe_load(compose_file.read_text(encoding="utf-8")) or {}
    services = payload.get("services")
    if not isinstance(services, dict) or not services:
        raise ValueError(f"compose file does not define services: {compose_file}")
    if "client" in services:
        return "client"
    if len(services) == 1:
        return str(next(iter(services)))
    raise ValueError(f"compose file {compose_file} defines multiple services and no 'client' service")


def count_trace_response_lines(trace_log: Path) -> int:
    count = 0
    for raw_line in trace_log.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            if _TRACE_RESPONSE_LINE_RE.search(line):
                count += 1
            continue
        if isinstance(entry, dict) and entry.get("type") == "response":
            count += 1
    return count


def build_dataset_row(
    *,
    task_id: str,
    task_root: Path,
    trace_log: Path,
    output_path: Path,
) -> dict[str, object]:
    task_yaml = task_root / "task.yaml"
    compose_file = task_root / "docker-compose.yaml"
    run_tests = task_root / "run-tests.sh"
    missing = [path for path in (task_yaml, compose_file, run_tests, trace_log) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required task or trace files for {task_id}: {missing}")
    task = load_task_yaml(task_yaml)
    trace = parse_replay_trace(trace_log)
    raw_trace_response_count = count_trace_response_lines(trace_log)
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"task {task_id} is missing instruction text")
    return {
        "task_id": task_id,
        "agent_type": "iflow",
        "llm_service_type": "iflow_trace_replay",
        "task_description": {"prompt": instruction},
        "task_config": {
            "options": {
                "task_id": task_id,
                "max_agent_timeout_sec": task.get("max_agent_timeout_sec"),
                "max_test_timeout_sec": task.get("max_test_timeout_sec"),
                "run_tests_in_same_shell": bool(task.get("run_tests_in_same_shell", False)),
            }
        },
        "docker_compose_file": _relative_path(compose_file, output_path=output_path),
        "service_name": resolve_service_name(compose_file),
        "task_root": _relative_path(task_root, output_path=output_path),
        "llm_service_config": {
            "trace_path": _relative_path(trace.trace_path, output_path=output_path),
        },
        "trace_response_count": raw_trace_response_count,
        "trace_malformed_line_count": len(trace.malformed_lines),
    }


def generate_dataset(*, tasks_root: Path, traces_root: Path, output_path: Path) -> list[dict[str, object]]:
    task_dirs = {path.name: path for path in tasks_root.iterdir() if path.is_dir()}
    trace_dirs = {path.name: path for path in traces_root.iterdir() if path.is_dir()}
    task_ids = sorted(task_dirs.keys() & trace_dirs.keys())
    rows = [
        build_dataset_row(
            task_id=task_id,
            task_root=task_dirs[task_id],
            trace_log=resolve_trace_log(trace_dirs[task_id]),
            output_path=output_path,
        )
        for task_id in task_ids
    ]
    return rows


def write_dataset(rows: list[dict[str, object]], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    rows = generate_dataset(
        tasks_root=args.tasks_root.expanduser().resolve(),
        traces_root=args.traces_root.expanduser().resolve(),
        output_path=args.out.expanduser().resolve(),
    )
    write_dataset(rows, output_path=args.out.expanduser().resolve())
    print(f"wrote {len(rows)} rows to {args.out.expanduser().resolve()}")


if __name__ == "__main__":
    main()

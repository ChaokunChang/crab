#!/usr/bin/env python3
"""Generate the Claude Code ATIF replay benchmark dataset JSONL."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.llm_services.claude_code_trace_replay.service import (
    is_strictly_replayable_trace,
    load_trace_payload,
    parse_replay_trace,
)

_PREFERRED_DEDUP_TRIAL_IDS: dict[str, tuple[str, ...]] = {
    "mailman": (
        "46c2e780-0160-4acf-a1e6-668cc5ca506b",
        "cae5a195-8f94-46fb-9b94-7a4a1f703599",
    ),
}

_DEFAULT_EXCLUDED_TASK_IDS: tuple[str, ...] = (
    "git-multibranch",
    "hf-model-inference",
)

_DEFAULT_EXCLUDED_TRIAL_IDS: tuple[str, ...] = (
    "41ed59d6-46bb-4c5d-af81-a1ff97d1a3b8",
    "b45538f5-0862-45f0-9758-f97e8fb29037",
    "8b56ce95-9f8b-4020-839d-5d1cc6dc9a10",
    "d19326ec-e1ff-476b-9b76-83103b1c8694",
    "4437cac5-d01b-43b8-ac37-f1d6f26cea89",
)


@dataclass(frozen=True)
class TraceCandidate:
    task_id: str
    trace_file: Path
    trial_id: str
    task_checksum: str | None
    result: str | None
    total_steps: int | None
    agent_version: str | None
    strict_replayable: bool
    merged_response_count: int
    trace_duration_seconds: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the Claude Code trace replay dataset JSONL")
    parser.add_argument(
        "--tasks-root",
        type=Path,
        default=ROOT / "results" / "original-tasks",
    )
    parser.add_argument(
        "--traces-root",
        type=Path,
        default=ROOT / "results" / "traces" / "tbench-claude-code-claude-opus4.6-trajectories",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "datasets" / "claude_code_replay.jsonl",
    )
    parser.add_argument(
        "--strict-replayable",
        action="store_true",
        default=False,
        help="Keep only successful traces whose API-side tool inputs can be reconstructed cleanly",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=False,
        help="Keep only one successful trace per task",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        dest="exclude_tasks",
        default=None,
        help=(
            "Exclude a task id from the generated dataset. "
            "May be passed multiple times. "
            f"Default excludes: {', '.join(_DEFAULT_EXCLUDED_TASK_IDS)}"
        ),
    )
    parser.add_argument(
        "--exclude-trial-id",
        action="append",
        dest="exclude_trial_ids",
        default=None,
        help=(
            "Exclude a specific trial id from the generated dataset. "
            "May be passed multiple times. "
            f"Default excludes: {', '.join(_DEFAULT_EXCLUDED_TRIAL_IDS)}"
        ),
    )
    parser.add_argument(
        "--include-trial-id",
        action="append",
        dest="include_trial_ids",
        default=None,
        help="Keep only the specified trial ids. May be passed multiple times.",
    )
    return parser.parse_args()


def _relative_path(path: Path, *, output_path: Path) -> str:
    return os.path.relpath(path.resolve(), start=output_path.resolve().parent)


def load_task_yaml(task_yaml: Path) -> dict[str, object]:
    payload = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"task yaml must be an object: {task_yaml}")
    return payload


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


def load_manifest(traces_root: Path) -> dict[str, Any]:
    manifest_path = traces_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing Claude Code manifest: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must decode to an object: {manifest_path}")
    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list):
        raise ValueError(f"manifest is missing a trajectories list: {manifest_path}")
    return payload


def _manifest_result_is_pass(result: object) -> bool:
    return isinstance(result, str) and "Pass" in result


def _trace_duration_seconds(trace_file: Path) -> float | None:
    payload = load_trace_payload(trace_file)
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        return None
    first_timestamp: str | None = None
    last_timestamp: str | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        timestamp = step.get("timestamp")
        if isinstance(timestamp, str) and timestamp.strip():
            first_timestamp = timestamp
            break
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        timestamp = step.get("timestamp")
        if isinstance(timestamp, str) and timestamp.strip():
            last_timestamp = timestamp
            break
    if first_timestamp is None or last_timestamp is None:
        return None
    from datetime import datetime

    def _parse(value: str) -> float:
        raw_value = value[:-1] + "+00:00" if value.endswith("Z") else value
        return datetime.fromisoformat(raw_value).timestamp()

    try:
        return max(0.0, _parse(last_timestamp) - _parse(first_timestamp))
    except ValueError:
        return None


def _build_candidate(trajectory: dict[str, Any], *, traces_root: Path) -> TraceCandidate:
    task_id = str(trajectory.get("task_name", "")).strip()
    if not task_id:
        raise ValueError("trajectory is missing task_name")
    trial_id = str(trajectory.get("trial_id", "")).strip()
    if not trial_id:
        raise ValueError(f"trajectory for task {task_id} is missing trial_id")
    results_path = trajectory.get("results_path")
    if not isinstance(results_path, str) or not results_path.strip():
        raise ValueError(f"trajectory for trial {trial_id} is missing results_path")
    results_dir = traces_root / results_path.strip()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"missing results directory for trial {trial_id}: {results_dir}")
    task_dirs = sorted(path for path in results_dir.iterdir() if path.is_dir())
    if len(task_dirs) != 1:
        raise ValueError(
            f"results_path for trial {trial_id} must contain exactly one task directory, found {len(task_dirs)}"
        )
    trace_file = task_dirs[0] / "agent" / "trajectory.json"
    if not trace_file.exists():
        raise FileNotFoundError(f"missing full trajectory for trial {trial_id}: {trace_file}")
    parsed = parse_replay_trace(trace_file)
    strict_replayable = is_strictly_replayable_trace(trace_file)
    total_steps = trajectory.get("total_steps")
    return TraceCandidate(
        task_id=task_id,
        trace_file=trace_file,
        trial_id=trial_id,
        task_checksum=str(trajectory["task_checksum"]) if trajectory.get("task_checksum") is not None else None,
        result=str(trajectory["result"]) if trajectory.get("result") is not None else None,
        total_steps=int(total_steps) if isinstance(total_steps, int) else None,
        agent_version=str(trajectory["agent_version"]) if trajectory.get("agent_version") is not None else None,
        strict_replayable=strict_replayable,
        merged_response_count=len(parsed.agent_steps),
        trace_duration_seconds=_trace_duration_seconds(trace_file),
    )


def _candidate_rank_key(candidate: TraceCandidate) -> tuple[int, int, int, str]:
    preferred_trials = _PREFERRED_DEDUP_TRIAL_IDS.get(candidate.task_id, ())
    preferred_rank = preferred_trials.index(candidate.trial_id) if candidate.trial_id in preferred_trials else len(
        preferred_trials
    )
    return (
        preferred_rank,
        0 if candidate.strict_replayable else 1,
        candidate.merged_response_count,
        candidate.total_steps if candidate.total_steps is not None else sys.maxsize,
        candidate.trial_id,
    )


def build_dataset_row(
    *,
    task_root: Path,
    candidate: TraceCandidate,
    output_path: Path,
) -> dict[str, object]:
    task_yaml = task_root / "task.yaml"
    compose_file = task_root / "docker-compose.yaml"
    run_tests = task_root / "run-tests.sh"
    missing = [path for path in (task_yaml, compose_file, run_tests, candidate.trace_file) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing required files for {candidate.task_id}: {missing}")
    task = load_task_yaml(task_yaml)
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"task {candidate.task_id} is missing instruction text")
    task_options: dict[str, object] = {
        "task_id": candidate.task_id,
        "max_agent_timeout_sec": task.get("max_agent_timeout_sec"),
        "max_test_timeout_sec": task.get("max_test_timeout_sec"),
        "run_tests_in_same_shell": bool(task.get("run_tests_in_same_shell", False)),
    }
    if candidate.agent_version is not None:
        task_options["trace_agent_version"] = candidate.agent_version
    return {
        "task_id": candidate.task_id,
        "agent_type": "claude_code",
        "llm_service_type": "claude_code_trace_replay",
        "task_description": {"prompt": instruction},
        "task_config": {
            "options": task_options,
        },
        "docker_compose_file": _relative_path(compose_file, output_path=output_path),
        "service_name": resolve_service_name(compose_file),
        "task_root": _relative_path(task_root, output_path=output_path),
        "llm_service_config": {
            "trace_path": _relative_path(candidate.trace_file, output_path=output_path),
        },
        "trace_response_count": candidate.merged_response_count,
        "trace_replay_progress_count": candidate.merged_response_count,
        "trace_malformed_line_count": 0,
        "trace_duration_seconds": candidate.trace_duration_seconds,
        "trace_trial_id": candidate.trial_id,
        "trace_task_checksum": candidate.task_checksum,
        "trace_result": candidate.result,
        "trace_total_steps": candidate.total_steps,
        "trace_agent_version": candidate.agent_version,
    }


def _round_robin_candidates(by_task: dict[str, list[TraceCandidate]]) -> list[TraceCandidate]:
    queues = {
        task_id: deque(sorted(candidates, key=_candidate_rank_key))
        for task_id, candidates in sorted(by_task.items())
    }
    ordered: list[TraceCandidate] = []
    while True:
        progressed = False
        for task_id in sorted(queues):
            queue = queues[task_id]
            if not queue:
                continue
            ordered.append(queue.popleft())
            progressed = True
        if not progressed:
            break
    return ordered


def generate_dataset(
    *,
    tasks_root: Path,
    traces_root: Path,
    output_path: Path,
    strict_replayable: bool = False,
    deduplicate: bool = False,
    exclude_tasks: set[str] | None = None,
    exclude_trial_ids: set[str] | None = None,
    include_trial_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    manifest = load_manifest(traces_root)
    task_dirs = {path.name: path for path in tasks_root.iterdir() if path.is_dir()}
    candidates_by_task: dict[str, list[TraceCandidate]] = defaultdict(list)
    skipped = 0
    excluded_task_ids = set(_DEFAULT_EXCLUDED_TASK_IDS)
    if exclude_tasks:
        excluded_task_ids.update(task_id.strip() for task_id in exclude_tasks if task_id.strip())
    excluded_trial_ids = set(_DEFAULT_EXCLUDED_TRIAL_IDS)
    if exclude_trial_ids:
        excluded_trial_ids.update(trial_id.strip() for trial_id in exclude_trial_ids if trial_id.strip())
    included_trial_ids = {
        trial_id.strip()
        for trial_id in (include_trial_ids or set())
        if trial_id.strip()
    }

    trajectories = manifest.get("trajectories", [])
    assert isinstance(trajectories, list)
    for trajectory in trajectories:
        if not isinstance(trajectory, dict):
            continue
        if not _manifest_result_is_pass(trajectory.get("result")):
            continue
        task_id = str(trajectory.get("task_name", "")).strip()
        trial_id = str(trajectory.get("trial_id", "")).strip()
        if included_trial_ids and trial_id not in included_trial_ids:
            skipped += 1
            continue
        if task_id in excluded_task_ids:
            skipped += 1
            continue
        if trial_id in excluded_trial_ids and trial_id not in included_trial_ids:
            skipped += 1
            continue
        if not task_id or task_id not in task_dirs:
            skipped += 1
            continue
        try:
            candidate = _build_candidate(trajectory, traces_root=traces_root)
        except Exception as exc:
            print(f"{task_id} skipped: {exc}")
            skipped += 1
            continue
        if strict_replayable and not candidate.strict_replayable:
            skipped += 1
            continue
        candidates_by_task[task_id].append(candidate)

    selected: list[TraceCandidate]
    if deduplicate:
        selected = [
            min(candidates, key=_candidate_rank_key)
            for task_id, candidates in sorted(candidates_by_task.items())
            if candidates
        ]
    else:
        selected = _round_robin_candidates(candidates_by_task)

    rows: list[dict[str, object]] = []
    for candidate in selected:
        try:
            rows.append(
                build_dataset_row(
                    task_root=task_dirs[candidate.task_id],
                    candidate=candidate,
                    output_path=output_path,
                )
            )
        except Exception as exc:
            print(f"{candidate.task_id} skipped: {exc}")
            skipped += 1

    if skipped > 0:
        print(f"skipped {skipped} trajectories")
    return rows


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
        tasks_root=args.tasks_root.expanduser().resolve(),
        traces_root=args.traces_root.expanduser().resolve(),
        output_path=output_path,
        strict_replayable=args.strict_replayable,
        deduplicate=args.deduplicate,
        exclude_tasks=set(args.exclude_tasks or ()),
        exclude_trial_ids=set(args.exclude_trial_ids or ()),
        include_trial_ids=set(args.include_trial_ids or ()),
    )
    write_dataset(rows, output_path=output_path)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

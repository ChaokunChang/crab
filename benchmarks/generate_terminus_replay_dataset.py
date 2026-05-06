#!/usr/bin/env python3
"""Generate the Terminus + TerminalBench trace replay benchmark dataset JSONL.

The Terminus traces live under ``results/traces/tbench-terminus-all/submissions/...``
in a per-model layout. This generator walks every model directory, keeps trials
whose verifier reward is non-zero, and emits one JSONL row per replayable trial
pointing to the original Terminal-Bench task definition under
``results/original-tasks``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from integrations.llm_services.terminus_trace_replay.service import parse_replay_trace

_DEFAULT_TRACES_ROOT = (
    ROOT
    / "results"
    / "traces"
    / "tbench-terminus-all"
    / "submissions"
    / "terminal-bench"
    / "2.0"
)
_DEFAULT_TASKS_ROOT = ROOT / "results" / "original-tasks"
_DEFAULT_OUTPUT = ROOT / "results" / "datasets" / "terminus_replay.jsonl"

_TASK_TRIAL_DIR_RE = re.compile(r"^(?P<task>[A-Za-z0-9._-]+)__(?P<suffix>[A-Za-z0-9._-]+)$")

# Trial directories whose recorded reward=1.0 was an environment-specific
# fluke and which fail verification on replay. Each entry should carry a
# brief note explaining the failure mode so future maintainers can tell
# whether the underlying issue has been fixed and the trace can be
# re-enabled.
#
# bn-fit-modify__4dq9Ruc:
#   step 28 calls `cpd_d.std` from pgmpy's LinearGaussianCPD and treats
#   the value as variance, then uses `np.sqrt(d_var)` as the noise std
#   when sampling D. In the pgmpy version present at replay time `std`
#   is the actual standard deviation (not variance), so the sampled D
#   distribution is too narrow and the verifier's KS test fails with
#   p_value=0.0. The other bn-fit-modify trace (`sopXX3R`) does the
#   sampling correctly and passes.
_DEFAULT_EXCLUDED_TRIAL_DIRS: frozenset[str] = frozenset({
    "bn-fit-modify__4dq9Ruc",
})


@dataclass(frozen=True)
class TraceCandidate:
    task_id: str
    trial_dir_name: str
    trial_id: str | None
    model_name: str | None
    model_label: str
    run_label: str
    trace_file: Path
    config_file: Path
    result_file: Path
    reward: float
    parser_name: str | None
    response_count: int
    initial_user_prompt: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Terminus + TerminalBench trace replay benchmark dataset JSONL"
    )
    parser.add_argument("--tasks-root", type=Path, default=_DEFAULT_TASKS_ROOT)
    parser.add_argument("--traces-root", type=Path, default=_DEFAULT_TRACES_ROOT)
    parser.add_argument("--out", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument(
        "--include-model",
        action="append",
        dest="include_models",
        default=None,
        help="Restrict to the given model labels (Terminus2__ directory name). May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-model",
        action="append",
        dest="exclude_models",
        default=None,
        help="Skip the given model labels. May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-task",
        action="append",
        dest="exclude_tasks",
        default=None,
        help="Skip the given task ids. May be passed multiple times.",
    )
    parser.add_argument(
        "--exclude-trial-dir",
        action="append",
        dest="exclude_trial_dirs",
        default=None,
        help=(
            "Skip the given trial directory names (e.g. bn-fit-modify__4dq9Ruc). "
            "Augments the built-in known-broken-trace blacklist; pass --no-default-excludes "
            "to disable the built-in blacklist. May be passed multiple times."
        ),
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        default=False,
        help="Disable the built-in known-broken-trace trial-dir blacklist.",
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        default=False,
        help="Keep only one successful trial per (task_id, model). Useful for clean per-model splits.",
    )
    return parser.parse_args()


def _relative_path(path: Path, *, output_path: Path) -> str:
    return os.path.relpath(path.resolve(), start=output_path.resolve().parent)


def _load_task_yaml(task_yaml: Path) -> dict[str, Any]:
    payload = yaml.safe_load(task_yaml.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"task yaml must be an object: {task_yaml}")
    return payload


def _resolve_service_name(compose_file: Path) -> str:
    payload = yaml.safe_load(compose_file.read_text(encoding="utf-8")) or {}
    services = payload.get("services")
    if not isinstance(services, dict) or not services:
        raise ValueError(f"compose file does not define services: {compose_file}")
    if "client" in services:
        return "client"
    if len(services) == 1:
        return str(next(iter(services)))
    raise ValueError(
        f"compose file {compose_file} defines multiple services and no 'client' service"
    )


def _load_result_reward(result_file: Path) -> float | None:
    if not result_file.exists():
        return None
    payload = json.loads(result_file.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    verifier = payload.get("verifier_result")
    if not isinstance(verifier, dict):
        return None
    rewards = verifier.get("rewards")
    if not isinstance(rewards, dict):
        return None
    raw_reward = rewards.get("reward")
    try:
        return float(raw_reward)
    except (TypeError, ValueError):
        return None


def _load_trial_metadata(config_file: Path, result_file: Path) -> tuple[str | None, str | None, str | None]:
    trial_id: str | None = None
    model_name: str | None = None
    parser_name: str | None = None
    if result_file.exists():
        payload = json.loads(result_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raw_trial_id = payload.get("id")
            if isinstance(raw_trial_id, str) and raw_trial_id.strip():
                trial_id = raw_trial_id.strip()
            agent_info = payload.get("agent_info")
            if isinstance(agent_info, dict):
                model_info = agent_info.get("model_info")
                if isinstance(model_info, dict):
                    raw_name = model_info.get("name")
                    if isinstance(raw_name, str) and raw_name.strip():
                        model_name = raw_name.strip()
    if config_file.exists():
        try:
            payload = json.loads(config_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            agent_cfg = payload.get("agent")
            if isinstance(agent_cfg, dict):
                raw_model = agent_cfg.get("model_name")
                if isinstance(raw_model, str) and raw_model.strip() and not model_name:
                    model_name = raw_model.strip()
                kwargs = agent_cfg.get("kwargs")
                if isinstance(kwargs, dict):
                    raw_parser = kwargs.get("parser_name")
                    if isinstance(raw_parser, str) and raw_parser.strip():
                        parser_name = raw_parser.strip().lower()
    return trial_id, model_name, parser_name


def _iter_trial_dirs(traces_root: Path) -> list[tuple[str, str, Path]]:
    """Yield (model_label, run_label, trial_dir) for every trial under traces_root."""
    results: list[tuple[str, str, Path]] = []
    if not traces_root.exists():
        return results
    for model_dir in sorted(traces_root.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith("Terminus2__"):
            continue
        model_label = model_dir.name
        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run_label = run_dir.name
            for trial_dir in sorted(run_dir.iterdir()):
                if not trial_dir.is_dir():
                    continue
                if not _TASK_TRIAL_DIR_RE.match(trial_dir.name):
                    continue
                results.append((model_label, run_label, trial_dir))
    return results


def _build_candidate(
    *,
    model_label: str,
    run_label: str,
    trial_dir: Path,
) -> TraceCandidate | None:
    match = _TASK_TRIAL_DIR_RE.match(trial_dir.name)
    if match is None:
        return None
    task_id = match.group("task")
    trace_file = trial_dir / "agent" / "trajectory.json"
    if not trace_file.exists():
        return None
    config_file = trial_dir / "config.json"
    result_file = trial_dir / "result.json"
    reward = _load_result_reward(result_file)
    if reward is None or reward <= 0.0:
        return None
    trial_id, model_name, parser_name = _load_trial_metadata(config_file, result_file)
    parsed = parse_replay_trace(trace_file)
    return TraceCandidate(
        task_id=task_id,
        trial_dir_name=trial_dir.name,
        trial_id=trial_id,
        model_name=model_name or parsed.model_name,
        model_label=model_label,
        run_label=run_label,
        trace_file=trace_file,
        config_file=config_file,
        result_file=result_file,
        reward=reward,
        parser_name=parser_name,
        response_count=len(parsed.assistant_messages),
        initial_user_prompt=parsed.initial_user_prompt,
    )


def _build_dataset_row(
    *,
    candidate: TraceCandidate,
    task_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    task_yaml = task_root / "task.yaml"
    compose_file = task_root / "docker-compose.yaml"
    run_tests = task_root / "run-tests.sh"
    missing = [
        path
        for path in (task_yaml, compose_file, run_tests, candidate.trace_file)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing required files for {candidate.task_id} ({candidate.trial_dir_name}): {missing}"
        )
    task = _load_task_yaml(task_yaml)
    instruction = task.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError(f"task {candidate.task_id} is missing instruction text")
    task_options: dict[str, Any] = {
        "task_id": candidate.task_id,
        "max_agent_timeout_sec": task.get("max_agent_timeout_sec"),
        "max_test_timeout_sec": task.get("max_test_timeout_sec"),
        "run_tests_in_same_shell": bool(task.get("run_tests_in_same_shell", False)),
    }
    return {
        "task_id": candidate.task_id,
        "agent_type": "terminus",
        "llm_service_type": "terminus_trace_replay",
        "task_description": {"prompt": instruction},
        "task_config": {"options": task_options},
        "docker_compose_file": _relative_path(compose_file, output_path=output_path),
        "service_name": _resolve_service_name(compose_file),
        "task_root": _relative_path(task_root, output_path=output_path),
        "llm_service_config": {
            "trace_path": _relative_path(candidate.trace_file, output_path=output_path),
        },
        "trace_response_count": candidate.response_count,
        "trace_replay_progress_count": candidate.response_count,
        "trace_malformed_line_count": 0,
        "trace_trial_id": candidate.trial_id,
        "trace_trial_dir": candidate.trial_dir_name,
        "trace_model_label": candidate.model_label,
        "trace_run_label": candidate.run_label,
        "trace_model_name": candidate.model_name,
        "trace_parser_name": candidate.parser_name,
        "trace_reward": candidate.reward,
    }


def generate_dataset(
    *,
    tasks_root: Path,
    traces_root: Path,
    output_path: Path,
    include_models: set[str] | None = None,
    exclude_models: set[str] | None = None,
    exclude_tasks: set[str] | None = None,
    exclude_trial_dirs: set[str] | None = None,
    deduplicate: bool = False,
) -> list[dict[str, Any]]:
    task_dirs = {path.name: path for path in tasks_root.iterdir() if path.is_dir()}
    candidates_by_key: dict[tuple[str, str], list[TraceCandidate]] = defaultdict(list)
    skipped = 0
    excluded_models = {label for label in (exclude_models or set()) if label}
    included_models = {label for label in (include_models or set()) if label}
    excluded_tasks = {task_id for task_id in (exclude_tasks or set()) if task_id}
    excluded_trial_dirs = {name for name in (exclude_trial_dirs or set()) if name}

    for model_label, run_label, trial_dir in _iter_trial_dirs(traces_root):
        if included_models and model_label not in included_models:
            skipped += 1
            continue
        if model_label in excluded_models:
            skipped += 1
            continue
        if trial_dir.name in excluded_trial_dirs:
            skipped += 1
            continue
        match = _TASK_TRIAL_DIR_RE.match(trial_dir.name)
        if match is None:
            skipped += 1
            continue
        task_id = match.group("task")
        if task_id in excluded_tasks:
            skipped += 1
            continue
        if task_id not in task_dirs:
            print(f"{trial_dir.name} skipped: no original task at {tasks_root / task_id}")
            skipped += 1
            continue
        try:
            candidate = _build_candidate(
                model_label=model_label, run_label=run_label, trial_dir=trial_dir
            )
        except Exception as exc:
            print(f"{trial_dir.name} skipped while parsing trace: {exc}")
            skipped += 1
            continue
        if candidate is None:
            skipped += 1
            continue
        candidates_by_key[(candidate.task_id, candidate.model_label)].append(candidate)

    selected: list[TraceCandidate] = []
    for (_, _), candidates in sorted(candidates_by_key.items()):
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.reward,
                candidate.run_label,
                candidate.trial_dir_name,
            ),
        )
        if deduplicate:
            selected.append(ordered[0])
        else:
            selected.extend(ordered)

    rows: list[dict[str, Any]] = []
    for candidate in selected:
        try:
            rows.append(
                _build_dataset_row(
                    candidate=candidate,
                    task_root=task_dirs[candidate.task_id],
                    output_path=output_path,
                )
            )
        except Exception as exc:
            print(f"{candidate.trial_dir_name} skipped while building row: {exc}")
            skipped += 1

    if skipped:
        print(f"skipped {skipped} trial directories")
    return rows


def write_dataset(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_path = args.out.expanduser().resolve()
    user_excluded_dirs = set(args.exclude_trial_dirs or ())
    if args.no_default_excludes:
        excluded_trial_dirs = user_excluded_dirs
    else:
        excluded_trial_dirs = user_excluded_dirs | set(_DEFAULT_EXCLUDED_TRIAL_DIRS)
    rows = generate_dataset(
        tasks_root=args.tasks_root.expanduser().resolve(),
        traces_root=args.traces_root.expanduser().resolve(),
        output_path=output_path,
        include_models=set(args.include_models or ()),
        exclude_models=set(args.exclude_models or ()),
        exclude_tasks=set(args.exclude_tasks or ()),
        exclude_trial_dirs=excluded_trial_dirs,
        deduplicate=args.deduplicate,
    )
    write_dataset(rows, output_path=output_path)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()

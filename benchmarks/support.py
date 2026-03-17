from __future__ import annotations

import argparse
import csv
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from agent_cr import CheckpointId, CheckpointManifest
from integrations.agents import TaskConfig, TaskDescription


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def bounded_probability(raw: str) -> float:
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(f"expected probability in [0.0, 1.0], got {raw}")
    return value


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--agent-type", choices=["simulated", "iflow"], default="simulated")
    parser.add_argument("--llm-service-type", choices=["simulated", "manual", "simulated_for_iflow"], default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", default="")
    parser.add_argument("--transfer-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--work-dir-host-root",
        type=Path,
        default=None,
        help="Host directory root for per-sandbox /work bind mounts",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )


def wait_for(
    predicate,
    *,
    timeout_s: float = 30.0,
    interval_s: float = 0.2,
    raise_on_timeout: bool = True,
):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    if raise_on_timeout:
        raise RuntimeError("timed out waiting for predicate")
    return False


def total_actions(payload: dict[str, object]) -> int:
    return int(payload.get("total_actions", 0))


def compute_summary(rows: list[dict[str, object]], metric_keys: Iterable[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not rows:
        return summary
    for key in metric_keys:
        summary[key] = sum(float(row[key]) for row in rows) / len(rows)
    return summary


def select_injected_indices(
    population_size: int,
    *,
    iteration: int,
    rate: float,
    first_forced_iteration: int,
    rng: random.Random,
) -> list[int]:
    if population_size <= 0:
        return []
    if first_forced_iteration > 0 and iteration < first_forced_iteration:
        return []
    selected = [index for index in range(population_size) if rng.random() < rate]
    if first_forced_iteration > 0 and iteration == first_forced_iteration and 0 not in selected:
        selected.insert(0, 0)
    return sorted(set(selected))


def resolve_work_dir_host_path(work_dir_host_root: Path | None, sandbox_name: str) -> Path | None:
    if work_dir_host_root is None:
        return None
    return work_dir_host_root.expanduser().resolve() / sandbox_name


def resolve_checkpoint_copy_plan(
    checkpoint_order: list[CheckpointId],
    manifests: dict[CheckpointId, CheckpointManifest],
    checkpoint_id: CheckpointId,
) -> list[tuple[CheckpointId, bool, bool]]:
    manifest = manifests[checkpoint_id]
    plan: list[tuple[CheckpointId, bool, bool]] = [
        (checkpoint_id, bool(manifest.process_artifacts), bool(manifest.filesystem_artifacts))
    ]
    need_process = not bool(manifest.process_artifacts)
    need_filesystem = not bool(manifest.filesystem_artifacts)
    if not need_process and not need_filesystem:
        return plan

    try:
        current_index = checkpoint_order.index(checkpoint_id)
        candidates = list(reversed(checkpoint_order[:current_index]))
    except ValueError:
        candidates = list(reversed(checkpoint_order))

    for candidate_id in candidates:
        if not need_process and not need_filesystem:
            break
        candidate = manifests[candidate_id]
        copy_process = need_process and bool(candidate.process_artifacts)
        copy_filesystem = need_filesystem and bool(candidate.filesystem_artifacts)
        if not copy_process and not copy_filesystem:
            continue
        plan.insert(0, (candidate_id, copy_process, copy_filesystem))
        if copy_process:
            need_process = False
        if copy_filesystem:
            need_filesystem = False

    if need_process or need_filesystem:
        raise ValueError(f"unable to resolve restore dependencies for checkpoint {checkpoint_id}")
    return plan


@dataclass(frozen=True)
class TreeSearchCheckpointRecord:
    checkpoint_id: CheckpointId
    replay_actions: int
    checkpoint_ms: float = 0.0


def build_tree_search_checkpoint_index(
    manifests: Iterable[CheckpointManifest],
    *,
    initial_steps: int | None = None,
    require_complete: bool = False,
) -> dict[int, TreeSearchCheckpointRecord]:
    indexed: dict[int, TreeSearchCheckpointRecord] = {}
    for manifest in manifests:
        raw_step = manifest.metadata.get("tree_search_step")
        if raw_step is None:
            continue
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid tree_search_step={raw_step!r} for checkpoint {manifest.checkpoint_id}"
            ) from exc
        if step <= 0:
            continue
        if initial_steps is not None and step > initial_steps:
            continue
        if step in indexed:
            raise ValueError(f"duplicate tree-search checkpoint for step {step}")
        indexed[step] = TreeSearchCheckpointRecord(
            checkpoint_id=manifest.checkpoint_id,
            replay_actions=step,
        )

    if require_complete and initial_steps is not None:
        missing = [step for step in range(1, initial_steps + 1) if step not in indexed]
        if missing:
            raise ValueError(f"missing tree-search checkpoints for steps {missing}")
    return dict(sorted(indexed.items()))


@dataclass(frozen=True)
class BenchmarkTaskRecord:
    agent_type: str
    task_description: TaskDescription
    task_config: TaskConfig
    llm_service_type: str | None = None
    docker_compose_file: Path | None = None
    env_file: Path | None = None
    service_name: str | None = None


def write_rows(path: str, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

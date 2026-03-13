#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import KeepAllCheckpointManager, RequestContext, RequestInterceptorHook, SchedulerConfig, TreeSearchCheckpointingPolicy

from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
    TreeSearchCheckpointRecord,
    add_common_args,
    compute_summary,
    configure_logging,
    write_rows,
)

if TYPE_CHECKING:
    from benchmarks.real_host_scenario_base import SandboxHandle


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR tree-search real-host benchmark")
    parser.add_argument("--sandboxes", type=int, default=1)
    parser.add_argument("--initial-steps", type=int, default=6)
    parser.add_argument("--replay-points", type=int, default=2)
    parser.add_argument("--fork-steps", type=int, default=3)
    parser.add_argument("--auto-cr", action="store_true")
    parser.add_argument("--replay-mode", choices=["sequential", "concurrent"], default="sequential")
    add_common_args(parser)
    return parser.parse_args()


def choose_replay_steps(initial_steps: int, replay_points: int) -> list[int]:
    if initial_steps <= 1 or replay_points <= 0:
        return []
    candidates = list(range(1, initial_steps))
    if replay_points >= len(candidates):
        return candidates
    stride = max(1, len(candidates) // replay_points)
    selected = candidates[::stride][:replay_points]
    return selected


class TreeSearchStepHook(RequestInterceptorHook):
    def __init__(self, harness: RealHostScenarioHarness):
        self._harness = harness
        self._steps: dict[object, int] = defaultdict(int)

    def on_request_start(self, context: RequestContext) -> None:
        step = self._steps.get(context.sandbox_id, 0) + 1
        self._steps[context.sandbox_id] = step
        logger.debug("TreeSearch observed request_start sandbox=%s step=%d", context.sandbox_id, step)
        self._harness.set_snapshot_metadata_by_id(context.sandbox_id, tree_search_step=step)

    def on_request_end(self, context: RequestContext) -> None:
        _ = context


@dataclass(frozen=True)
class PreparedReplayFork:
    source_index: int
    replay_step: int
    checkpoint: TreeSearchCheckpointRecord
    fork: SandboxHandle


@dataclass(frozen=True)
class RestoredReplayFork:
    prepared: PreparedReplayFork
    restore_started_at: float
    recovery_finished_at: float
    ready_at: float
    restore_ms: float
    recovery_ms: float
    readiness_ms: float
    end_to_end_recovery_ms: float


def collect_manual_checkpoint_indexes(
    harness: RealHostScenarioHarness,
    sources: list[SandboxHandle],
    *,
    initial_steps: int,
    source_indices: dict[str, int],
) -> dict[str, dict[int, TreeSearchCheckpointRecord]]:
    checkpoints_by_source: dict[str, dict[int, TreeSearchCheckpointRecord]] = {
        str(source.sandbox_id): {} for source in sources
    }
    for step in range(1, initial_steps + 1):
        for source in sources:
            harness.wait_for_action_delta(source, delta=1)
            harness.set_snapshot_metadata(source, tree_search_step=step)
            checkpoint_started = time.perf_counter()
            checkpoint_result = harness.checkpoint_manual(source, leave_running=True)
            checkpoint_finished = time.perf_counter()
            if checkpoint_result is None:
                raise RuntimeError("tree-search benchmark expected a checkpoint at each step")
            checkpoint_ms = (checkpoint_finished - checkpoint_started) * 1000.0
            checkpoints_by_source[str(source.sandbox_id)][step] = TreeSearchCheckpointRecord(
                checkpoint_id=checkpoint_result.checkpoint_id,
                replay_actions=step,
                checkpoint_ms=checkpoint_ms,
            )
            logger.info(
                "TreeSearch manual checkpoint source_index=%d step=%d checkpoint=%s checkpoint_ms=%.3f",
                source_indices[str(source.sandbox_id)],
                step,
                checkpoint_result.checkpoint_id,
                checkpoint_ms,
            )
    return checkpoints_by_source


def collect_auto_checkpoint_indexes(
    harness: RealHostScenarioHarness,
    sources: list[SandboxHandle],
    *,
    initial_steps: int,
    source_indices: dict[str, int],
) -> dict[str, dict[int, TreeSearchCheckpointRecord]]:
    checkpoints_by_source: dict[str, dict[int, TreeSearchCheckpointRecord]] = {}
    logger.info(
        "TreeSearch running auto rollout sandboxes=%d initial_steps=%d",
        len(sources),
        initial_steps,
    )
    for source in sources:
        harness.wait_for_action_delta(source, delta=initial_steps)
    for source in sources:
        checkpoints_by_step = harness.wait_for_tree_search_checkpoints(
            source.sandbox_id,
            initial_steps=initial_steps,
        )
        checkpoints_by_source[str(source.sandbox_id)] = checkpoints_by_step
        logger.info(
            "TreeSearch collected source checkpoints source_index=%d steps=%s",
            source_indices[str(source.sandbox_id)],
            sorted(checkpoints_by_step.keys()),
        )
    return checkpoints_by_source


def prepare_replay_fork(
    harness: RealHostScenarioHarness,
    source: SandboxHandle,
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord],
    *,
    source_index: int,
    replay_step: int,
) -> PreparedReplayFork:
    checkpoint = checkpoints_by_step[replay_step]
    fork = harness.clone_tree_search_checkpoint_to_fork(
        source,
        checkpoint.checkpoint_id,
        f"tree-fork-{source_index}-{replay_step}",
    )
    logger.info(
        "TreeSearch prepared fork source_index=%d fork=%s replay_step=%d checkpoint=%s",
        source_index,
        fork.sandbox_id,
        replay_step,
        checkpoint.checkpoint_id,
    )
    return PreparedReplayFork(
        source_index=source_index,
        replay_step=replay_step,
        checkpoint=checkpoint,
        fork=fork,
    )


def restore_prepared_replay_fork(
    harness: RealHostScenarioHarness,
    prepared: PreparedReplayFork,
) -> RestoredReplayFork:
    logger.info(
        "TreeSearch restoring fork source_index=%d fork=%s replay_step=%d checkpoint=%s",
        prepared.source_index,
        prepared.fork.sandbox_id,
        prepared.replay_step,
        prepared.checkpoint.checkpoint_id,
    )
    restore_started = time.perf_counter()
    restore_result = harness.restore_once(prepared.fork, prepared.checkpoint.checkpoint_id)
    if restore_result.status.value != "succeeded":
        raise RuntimeError(
            f"tree-search restore failed for step {prepared.replay_step}: {restore_result.message}"
        )
    recovery_finished = time.perf_counter()
    prepared.fork.last_status = harness.poll_status(prepared.fork)
    ready_at = time.perf_counter()
    logger.info(
        "TreeSearch fork ready source_index=%d fork=%s replay_step=%d recovery_ms=%.3f readiness_ms=%.3f end_to_end_recovery_ms=%.3f",
        prepared.source_index,
        prepared.fork.sandbox_id,
        prepared.replay_step,
        (recovery_finished - restore_started) * 1000.0,
        (ready_at - recovery_finished) * 1000.0,
        (ready_at - restore_started) * 1000.0,
    )
    return RestoredReplayFork(
        prepared=prepared,
        restore_started_at=restore_started,
        recovery_finished_at=recovery_finished,
        ready_at=ready_at,
        restore_ms=(restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
        recovery_ms=(recovery_finished - restore_started) * 1000.0,
        readiness_ms=(ready_at - recovery_finished) * 1000.0,
        end_to_end_recovery_ms=(ready_at - restore_started) * 1000.0,
    )


def run_replay_progress(
    harness: RealHostScenarioHarness,
    restored: RestoredReplayFork,
    *,
    source_sandbox_id: str,
    retained_source_checkpoints: int,
    fork_steps: int,
) -> dict[str, object]:
    for _ in range(fork_steps):
        harness.wait_for_action_delta(restored.prepared.fork, delta=1)
    progress_finished = time.perf_counter()
    return {
        "source_index": restored.prepared.source_index,
        "source_sandbox_id": source_sandbox_id,
        "fork_sandbox_id": str(restored.prepared.fork.sandbox_id),
        "replay_step": restored.prepared.replay_step,
        "checkpoint_ms": restored.prepared.checkpoint.checkpoint_ms,
        "restore_ms": restored.restore_ms,
        "recovery_ms": restored.recovery_ms,
        "readiness_ms": restored.readiness_ms,
        "end_to_end_recovery_ms": restored.end_to_end_recovery_ms,
        "replay_progress_ms": (progress_finished - restored.ready_at) * 1000.0,
        "fanout_ms": (progress_finished - restored.restore_started_at) * 1000.0,
        "replay_actions": restored.prepared.checkpoint.replay_actions,
        "retained_source_checkpoints": retained_source_checkpoints,
    }


def cleanup_replay_fork(harness: RealHostScenarioHarness, prepared: PreparedReplayFork) -> None:
    harness.deactivate_sandbox_runtime(prepared.fork)
    harness.storage.delete_all_checkpoints(prepared.fork.sandbox_id)
    harness.destroy_sandbox_dataset(prepared.fork)


def run_tree_search_benchmark(
    args: argparse.Namespace,
    harness: RealHostScenarioHarness,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    replay_steps = choose_replay_steps(args.initial_steps, args.replay_points)
    logger.info(
        "TreeSearch sandboxes=%d auto_cr=%s replay_mode=%s replay_steps=%s",
        args.sandboxes,
        args.auto_cr,
        args.replay_mode,
        replay_steps,
    )
    sources = [harness.launch_tree_search_sandbox(f"tree-source-{source_index}") for source_index in range(args.sandboxes)]
    source_indices = {str(source.sandbox_id): index for index, source in enumerate(sources)}

    if args.auto_cr:
        harness.drain_request_state_changes()
        system = getattr(harness, "system", None)
        if system is not None:
            system.start()

    try:
        if args.auto_cr:
            checkpoints_by_source = collect_auto_checkpoint_indexes(
                harness,
                sources,
                initial_steps=args.initial_steps,
                source_indices=source_indices,
            )
            system = getattr(harness, "system", None)
            if system is not None:
                system.stop()
            harness.drain_request_state_changes()
        else:
            checkpoints_by_source = collect_manual_checkpoint_indexes(
                harness,
                sources,
                initial_steps=args.initial_steps,
                source_indices=source_indices,
            )

        for source in sources:
            harness.deactivate_sandbox_runtime(source)

        for source in sources:
            source_sandbox_id = str(source.sandbox_id)
            source_index = source_indices[source_sandbox_id]
            checkpoints_by_step = checkpoints_by_source[source_sandbox_id]
            retained_source_checkpoints = len(checkpoints_by_step)
            prepared_forks: list[PreparedReplayFork] = []
            try:
                if args.replay_mode == "concurrent":
                    prepared_forks = [
                        prepare_replay_fork(
                            harness,
                            source,
                            checkpoints_by_step,
                            source_index=source_index,
                            replay_step=replay_step,
                        )
                        for replay_step in replay_steps
                    ]
                    if prepared_forks:
                        with ThreadPoolExecutor(max_workers=len(prepared_forks)) as executor:
                            restored_forks = list(
                                executor.map(
                                    lambda prepared: restore_prepared_replay_fork(harness, prepared),
                                    prepared_forks,
                                )
                            )
                        with ThreadPoolExecutor(max_workers=len(restored_forks)) as executor:
                            rows.extend(
                                executor.map(
                                    lambda restored: run_replay_progress(
                                        harness,
                                        restored,
                                        source_sandbox_id=source_sandbox_id,
                                        retained_source_checkpoints=retained_source_checkpoints,
                                        fork_steps=args.fork_steps,
                                    ),
                                    restored_forks,
                                )
                            )
                else:
                    for replay_step in replay_steps:
                        prepared = prepare_replay_fork(
                            harness,
                            source,
                            checkpoints_by_step,
                            source_index=source_index,
                            replay_step=replay_step,
                        )
                        prepared_forks.append(prepared)
                        restored = restore_prepared_replay_fork(harness, prepared)
                        rows.append(
                            run_replay_progress(
                                harness,
                                restored,
                                source_sandbox_id=source_sandbox_id,
                                retained_source_checkpoints=retained_source_checkpoints,
                                fork_steps=args.fork_steps,
                            )
                        )
                        cleanup_replay_fork(harness, prepared)
                        prepared_forks.remove(prepared)
            finally:
                for prepared in list(prepared_forks):
                    cleanup_replay_fork(harness, prepared)
    finally:
        for source in sources:
            harness.storage.delete_all_checkpoints(source.sandbox_id)
            harness.destroy_sandbox_dataset(source)
    return rows


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    scheduler_config = SchedulerConfig(
        min_checkpoint_interval_seconds=0.0,
        force_checkpoint_after_seconds=0.0,
        require_change_signal=False,
    )
    with RealHostScenarioHarness(
        provider=args.provider,
        transfer_delay_ms=args.transfer_delay_ms,
        scheduler_config=scheduler_config,
        scheduler_policy=TreeSearchCheckpointingPolicy(),
        checkpoint_manager_factory=lambda base: KeepAllCheckpointManager(base),
        max_workers=max(1, args.sandboxes * max(1, args.replay_points + 1)),
        auto_cr=args.auto_cr,
        work_dir_host_root=args.work_dir_host_root,
    ) as harness:
        if args.auto_cr:
            harness.add_interceptor_hook(TreeSearchStepHook(harness))
        rows = run_tree_search_benchmark(args, harness)
    write_rows(args.out, rows)
    summary = compute_summary(
        rows,
        [
            "checkpoint_ms",
            "restore_ms",
            "recovery_ms",
            "readiness_ms",
            "end_to_end_recovery_ms",
            "replay_progress_ms",
            "fanout_ms",
            "retained_source_checkpoints",
        ],
    )
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")


if __name__ == "__main__":
    main()

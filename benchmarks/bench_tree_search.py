#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
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
    parser.add_argument("--iters", type=int, default=1)
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
    replay_step: int
    checkpoint: TreeSearchCheckpointRecord
    fork: SandboxHandle


def collect_manual_checkpoint_index(
    harness: RealHostScenarioHarness,
    source: SandboxHandle,
    *,
    initial_steps: int,
    iteration: int,
) -> dict[int, TreeSearchCheckpointRecord]:
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord] = {}
    for step in range(1, initial_steps + 1):
        harness.wait_for_action_delta(source, delta=1)
        harness.set_snapshot_metadata(source, tree_search_step=step)
        checkpoint_started = time.perf_counter()
        checkpoint_result = harness.checkpoint_if_due(source)
        checkpoint_finished = time.perf_counter()
        if checkpoint_result is None:
            raise RuntimeError("tree-search benchmark expected a checkpoint at each step")
        checkpoint_ms = (checkpoint_finished - checkpoint_started) * 1000.0
        checkpoints_by_step[step] = TreeSearchCheckpointRecord(
            checkpoint_id=checkpoint_result.checkpoint_id,
            replay_actions=step,
            checkpoint_ms=checkpoint_ms,
        )
        logger.info(
            "TreeSearch manual checkpoint iteration=%d step=%d checkpoint=%s checkpoint_ms=%.3f",
            iteration,
            step,
            checkpoint_result.checkpoint_id,
            checkpoint_ms,
        )
    return checkpoints_by_step


def collect_auto_checkpoint_index(
    harness: RealHostScenarioHarness,
    source: SandboxHandle,
    *,
    initial_steps: int,
    iteration: int,
) -> dict[int, TreeSearchCheckpointRecord]:
    logger.info(
        "TreeSearch running auto rollout iteration=%d initial_steps=%d",
        iteration,
        initial_steps,
    )
    harness.wait_for_action_delta(source, delta=initial_steps)
    checkpoints_by_step = harness.wait_for_tree_search_checkpoints(
        source.sandbox_id,
        initial_steps=initial_steps,
    )
    logger.info(
        "TreeSearch collected source checkpoints iteration=%d steps=%s",
        iteration,
        sorted(checkpoints_by_step.keys()),
    )
    return checkpoints_by_step


def prepare_replay_fork(
    harness: RealHostScenarioHarness,
    source: SandboxHandle,
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord],
    *,
    replay_step: int,
    iteration: int,
) -> PreparedReplayFork:
    checkpoint = checkpoints_by_step[replay_step]
    fork = harness.clone_checkpoint_to_fork(
        source,
        checkpoint.checkpoint_id,
        f"tree-fork-{iteration}-{replay_step}",
    )
    # Restored forks inherit the source runtime image, including the original status endpoint.
    fork.status_port = source.status_port
    logger.info(
        "TreeSearch prepared fork iteration=%d fork=%s replay_step=%d checkpoint=%s",
        iteration,
        fork.sandbox_id,
        replay_step,
        checkpoint.checkpoint_id,
    )
    return PreparedReplayFork(
        replay_step=replay_step,
        checkpoint=checkpoint,
        fork=fork,
    )


def wait_for_replay_progress(
    harness: RealHostScenarioHarness,
    prepared: PreparedReplayFork,
    *,
    fork_steps: int,
) -> None:
    for _ in range(fork_steps):
        harness.wait_for_action_delta(prepared.fork, delta=1)


def run_prepared_replay_fork(
    harness: RealHostScenarioHarness,
    prepared: PreparedReplayFork,
    *,
    iteration: int,
    source_sandbox_id: str,
    retained_source_checkpoints: int,
    fork_steps: int,
) -> dict[str, object]:
    logger.info(
        "TreeSearch restoring fork iteration=%d fork=%s replay_step=%d checkpoint=%s",
        iteration,
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
    wait_for_replay_progress(harness, prepared, fork_steps=fork_steps)
    progress_finished = time.perf_counter()
    logger.info(
        "TreeSearch fork ready iteration=%d fork=%s replay_step=%d recovery_ms=%.3f readiness_ms=%.3f end_to_end_recovery_ms=%.3f fanout_ms=%.3f",
        iteration,
        prepared.fork.sandbox_id,
        prepared.replay_step,
        (recovery_finished - restore_started) * 1000.0,
        (ready_at - recovery_finished) * 1000.0,
        (ready_at - restore_started) * 1000.0,
        (progress_finished - restore_started) * 1000.0,
    )
    return {
        "iter": iteration,
        "source_sandbox_id": source_sandbox_id,
        "fork_sandbox_id": str(prepared.fork.sandbox_id),
        "replay_step": prepared.replay_step,
        "checkpoint_ms": prepared.checkpoint.checkpoint_ms,
        "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
        "recovery_ms": (recovery_finished - restore_started) * 1000.0,
        "readiness_ms": (ready_at - recovery_finished) * 1000.0,
        "end_to_end_recovery_ms": (ready_at - restore_started) * 1000.0,
        "replay_progress_ms": (progress_finished - ready_at) * 1000.0,
        "fanout_ms": (progress_finished - restore_started) * 1000.0,
        "replay_actions": prepared.checkpoint.replay_actions,
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
    for iteration in range(args.iters):
        if args.auto_cr:
            harness.drain_request_state_changes()
            system = getattr(harness, "system", None)
            if system is not None:
                system.start()
        logger.info(
            "TreeSearch iteration=%d auto_cr=%s replay_mode=%s",
            iteration,
            args.auto_cr,
            args.replay_mode,
        )
        source = harness.launch_sandbox(f"tree-source-{iteration}")
        prepared_forks: list[PreparedReplayFork] = []
        try:
            if args.auto_cr:
                checkpoints_by_step = collect_auto_checkpoint_index(
                    harness,
                    source,
                    initial_steps=args.initial_steps,
                    iteration=iteration,
                )
            else:
                checkpoints_by_step = collect_manual_checkpoint_index(
                    harness,
                    source,
                    initial_steps=args.initial_steps,
                    iteration=iteration,
                )

            if args.auto_cr:
                system = getattr(harness, "system", None)
                if system is not None:
                    system.stop()
                harness.drain_request_state_changes()
            harness.deactivate_sandbox_runtime(source)
            replay_steps = choose_replay_steps(args.initial_steps, args.replay_points)
            logger.info(
                "TreeSearch selected replay_steps iteration=%d replay_steps=%s",
                iteration,
                replay_steps,
            )
            retained_source_checkpoints = len(checkpoints_by_step)
            if args.replay_mode == "concurrent":
                prepared_forks = [
                    prepare_replay_fork(
                        harness,
                        source,
                        checkpoints_by_step,
                        replay_step=replay_step,
                        iteration=iteration,
                    )
                    for replay_step in replay_steps
                ]
                replay_queue = list(prepared_forks)
            else:
                replay_queue = []

            for replay_step in replay_steps:
                prepared = (
                    replay_queue.pop(0)
                    if args.replay_mode == "concurrent"
                    else prepare_replay_fork(
                        harness,
                        source,
                        checkpoints_by_step,
                        replay_step=replay_step,
                        iteration=iteration,
                    )
                )
                try:
                    rows.append(
                        run_prepared_replay_fork(
                            harness,
                            prepared,
                            iteration=iteration,
                            source_sandbox_id=str(source.sandbox_id),
                            retained_source_checkpoints=retained_source_checkpoints,
                            fork_steps=args.fork_steps,
                        )
                    )
                finally:
                    cleanup_replay_fork(harness, prepared)
                    if prepared in prepared_forks:
                        prepared_forks.remove(prepared)
        finally:
            for prepared in list(prepared_forks):
                cleanup_replay_fork(harness, prepared)
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
        max_workers=max(1, args.replay_points + 1),
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

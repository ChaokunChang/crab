#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import logging
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import KeepAllCheckpointManager, RequestContext, RequestInterceptorHook, SchedulerConfig, TreeSearchCheckpointingPolicy

from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
    add_common_args,
    compute_summary,
    configure_logging,
    total_actions,
    write_rows,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR tree-search real-host benchmark")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--initial-steps", type=int, default=6)
    parser.add_argument("--replay-points", type=int, default=2)
    parser.add_argument("--fork-steps", type=int, default=3)
    parser.add_argument("--auto-cr", action="store_true")
    parser.add_argument("--rollout-seconds", type=float, default=6.0)
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


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    scheduler_config = SchedulerConfig(
        min_checkpoint_interval_seconds=0.0,
        force_checkpoint_after_seconds=0.0,
        require_change_signal=False,
    )
    rows: list[dict[str, object]] = []
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
        for iteration in range(args.iters):
            logger.info("TreeSearch iteration=%d auto_cr=%s", iteration, args.auto_cr)
            source = harness.launch_sandbox(f"tree-source-{iteration}")
            checkpoints_by_step: dict[int, tuple[object, int, float]] = {}
            if args.auto_cr:
                logger.info(
                    "TreeSearch running auto rollout iteration=%d rollout_seconds=%.3f",
                    iteration,
                    args.rollout_seconds,
                )
                deadline = time.time() + args.rollout_seconds
                while time.time() < deadline:
                    time.sleep(0.2)
                harness.inject_fault(source)
                harness.wait_for_checkpoint_count_stable(source.sandbox_id)
                for manifest in harness.list_checkpoint_manifests(source.sandbox_id):
                    step = int(manifest.metadata.get("tree_search_step", 0))
                    if step <= 0:
                        continue
                    checkpoints_by_step[step] = (manifest.checkpoint_id, 0, 0.0)
                logger.info(
                    "TreeSearch collected source checkpoints iteration=%d steps=%s",
                    iteration,
                    sorted(checkpoints_by_step.keys()),
                )
                if not checkpoints_by_step:
                    raise RuntimeError("tree-search auto mode did not produce any checkpoints")
                max_steps = max(checkpoints_by_step)
            else:
                for step in range(1, args.initial_steps + 1):
                    current = harness.wait_for_action_delta(source, delta=1)
                    harness.set_snapshot_metadata(source, tree_search_step=step)
                    t0 = time.perf_counter()
                    checkpoint_result = harness.checkpoint_if_due(source)
                    t1 = time.perf_counter()
                    if checkpoint_result is None:
                        raise RuntimeError("tree-search benchmark expected a checkpoint at each step")
                    checkpoints_by_step[step] = (
                        checkpoint_result.checkpoint_id,
                        total_actions(current),
                        (t1 - t0) * 1000.0,
                    )
                    logger.info(
                        "TreeSearch manual checkpoint iteration=%d step=%d checkpoint=%s checkpoint_ms=%.3f",
                        iteration,
                        step,
                        checkpoint_result.checkpoint_id,
                        (t1 - t0) * 1000.0,
                    )
                harness.inject_fault(source)
                max_steps = args.initial_steps
            replay_steps = choose_replay_steps(max_steps, args.replay_points)
            logger.info("TreeSearch selected replay_steps iteration=%d replay_steps=%s", iteration, replay_steps)
            source_port = source.status_port
            fork_ids: list[str] = []
            for replay_step in replay_steps:
                checkpoint_id, replay_actions, checkpoint_ms = checkpoints_by_step[replay_step]
                fork = harness.clone_checkpoint_to_fork(
                    source,
                    checkpoint_id,
                    f"tree-fork-{iteration}-{replay_step}",
                )
                fork_ids.append(str(fork.sandbox_id))
                fork.status_port = source_port
                logger.info(
                    "TreeSearch restoring fork iteration=%d fork=%s replay_step=%d checkpoint=%s",
                    iteration,
                    fork.sandbox_id,
                    replay_step,
                    checkpoint_id,
                )
                restore_started = time.perf_counter()
                restore_result = harness.restore_once(fork, checkpoint_id)
                if restore_result.status.value != "succeeded":
                    raise RuntimeError(f"tree-search restore failed for step {replay_step}: {restore_result.message}")
                recovery_finished = time.perf_counter()
                fork.last_status = harness.poll_status(fork)
                ready_at = time.perf_counter()
                if args.auto_cr:
                    progress_started = time.perf_counter()
                    time.sleep(max(0.5, args.fork_steps * 0.5))
                    fork.last_status = harness.poll_status(fork)
                    progress_finished = time.perf_counter()
                else:
                    progress_started = time.perf_counter()
                    for _ in range(args.fork_steps):
                        harness.wait_for_action_delta(fork, delta=1)
                    progress_finished = time.perf_counter()
                logger.info(
                    "TreeSearch fork ready iteration=%d fork=%s replay_step=%d recovery_ms=%.3f readiness_ms=%.3f end_to_end_recovery_ms=%.3f fanout_ms=%.3f",
                    iteration,
                    fork.sandbox_id,
                    replay_step,
                    (recovery_finished - restore_started) * 1000.0,
                    (ready_at - recovery_finished) * 1000.0,
                    (ready_at - restore_started) * 1000.0,
                    (progress_finished - restore_started) * 1000.0,
                )
                rows.append(
                    {
                        "iter": iteration,
                        "source_sandbox_id": str(source.sandbox_id),
                        "fork_sandbox_id": str(fork.sandbox_id),
                        "replay_step": replay_step,
                        "checkpoint_ms": checkpoint_ms,
                        "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
                        "recovery_ms": (recovery_finished - restore_started) * 1000.0,
                        "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                        "end_to_end_recovery_ms": (ready_at - restore_started) * 1000.0,
                        "replay_progress_ms": (progress_finished - ready_at) * 1000.0,
                        "fanout_ms": (progress_finished - restore_started) * 1000.0,
                        "replay_actions": replay_actions,
                        "retained_source_checkpoints": len(harness.storage.list_checkpoints(source.sandbox_id)),
                    }
                )
                harness.inject_fault(fork)
                harness.destroy_sandbox_dataset(fork)
            for fork_id in fork_ids:
                harness.storage.delete_all_checkpoints(fork_id)
            harness.storage.delete_all_checkpoints(source.sandbox_id)
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

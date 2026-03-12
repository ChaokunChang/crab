#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import KeepAllCheckpointManager, SchedulerConfig, TreeSearchCheckpointingPolicy

from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
    add_common_args,
    compute_summary,
    configure_logging,
    total_actions,
    wait_for,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR tree-search real-host benchmark")
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--initial-steps", type=int, default=6)
    parser.add_argument("--replay-points", type=int, default=2)
    parser.add_argument("--fork-steps", type=int, default=3)
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
    ) as harness:
        for iteration in range(args.iters):
            source = harness.launch_sandbox(f"tree-source-{iteration}")
            checkpoints_by_step: dict[int, tuple[object, int, float]] = {}
            for step in range(1, args.initial_steps + 1):
                current = harness.wait_for_action_delta(source, delta=1)
                harness.set_snapshot_metadata(source, tree_search_step=step)
                t0 = time.perf_counter()
                checkpoint_result = harness.checkpoint_if_due(source)
                t1 = time.perf_counter()
                if checkpoint_result is None:
                    raise RuntimeError("tree-search benchmark expected a checkpoint at each step")
                checkpoints_by_step[step] = (checkpoint_result.checkpoint_id, total_actions(current), (t1 - t0) * 1000.0)
            replay_steps = choose_replay_steps(args.initial_steps, args.replay_points)
            source_port = source.status_port
            harness.inject_fault(source)
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
                restore_started = time.perf_counter()
                restore_result = harness.restore_once(fork, checkpoint_id)
                if restore_result.status.value != "succeeded":
                    raise RuntimeError(f"tree-search restore failed for step {replay_step}: {restore_result.message}")
                wait_for(lambda: total_actions(harness.poll_status(fork)) >= replay_actions, timeout_s=45.0)
                fork.last_status = harness.poll_status(fork)
                for _ in range(args.fork_steps):
                    harness.wait_for_action_delta(fork, delta=1)
                restore_finished = time.perf_counter()
                rows.append(
                    {
                        "iter": iteration,
                        "source_sandbox_id": str(source.sandbox_id),
                        "fork_sandbox_id": str(fork.sandbox_id),
                        "replay_step": replay_step,
                        "checkpoint_ms": checkpoint_ms,
                        "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
                        "fanout_ms": (restore_finished - restore_started) * 1000.0,
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
    summary = compute_summary(rows, ["checkpoint_ms", "restore_ms", "fanout_ms", "retained_source_checkpoints"])
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")


if __name__ == "__main__":
    main()

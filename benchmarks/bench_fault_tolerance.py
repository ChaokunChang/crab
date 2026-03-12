#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import FaultToleranceCheckpointingPolicy, LatestOnlyCheckpointManager, SchedulerConfig

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
    parser = argparse.ArgumentParser(description="Agent-CR fault-tolerance real-host benchmark")
    parser.add_argument("--sandboxes", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    scheduler_config = SchedulerConfig(
        min_checkpoint_interval_seconds=0.0,
        force_checkpoint_after_seconds=0.0,
        require_change_signal=True,
        prefer_checkpoint_during_llm_request=True,
        require_llm_request_for_checkpoint=False,
    )
    rows: list[dict[str, object]] = []
    with RealHostScenarioHarness(
        provider=args.provider,
        transfer_delay_ms=args.transfer_delay_ms,
        scheduler_config=scheduler_config,
        scheduler_policy=FaultToleranceCheckpointingPolicy(scheduler_config),
        checkpoint_manager_factory=lambda base: LatestOnlyCheckpointManager(base),
        max_workers=args.sandboxes,
    ) as harness:
        sandboxes = [harness.launch_sandbox(f"fault-{index}") for index in range(args.sandboxes)]
        for iteration in range(args.iters):
            for sandbox in sandboxes:
                current = harness.wait_for_progress(sandbox, minimum_actions=6)
                checkpoint_actions = total_actions(current)
                t0 = time.perf_counter()
                checkpoint_result = harness.checkpoint_if_due(sandbox)
                t1 = time.perf_counter()
                if checkpoint_result is None:
                    continue
                pre_fault = harness.wait_for_action_delta(sandbox, delta=2)
                fault_started = time.perf_counter()
                harness.inject_fault(sandbox)
                restore_result = harness.restore_once(sandbox, checkpoint_result.checkpoint_id)
                wait_for(lambda: total_actions(harness.poll_status(sandbox)) >= checkpoint_actions, timeout_s=45.0)
                restored_status = harness.poll_status(sandbox)
                sandbox.last_status = restored_status
                post_restore = harness.wait_for_action_delta(sandbox, delta=1)
                recovered_at = time.perf_counter()
                rows.append(
                    {
                        "iter": iteration,
                        "sandbox_id": str(sandbox.sandbox_id),
                        "checkpoint_ms": (t1 - t0) * 1000.0,
                        "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
                        "recovery_ms": (recovered_at - fault_started) * 1000.0,
                        "checkpoint_actions": checkpoint_actions,
                        "pre_fault_actions": total_actions(pre_fault),
                        "post_restore_actions": total_actions(post_restore),
                        "lost_actions": max(0, total_actions(pre_fault) - checkpoint_actions),
                        "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                    }
                )
    write_rows(args.out, rows)
    summary = compute_summary(rows, ["checkpoint_ms", "restore_ms", "recovery_ms", "lost_actions"])
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")


if __name__ == "__main__":
    main()

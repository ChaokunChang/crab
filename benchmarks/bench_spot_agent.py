#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import DeleteAfterRestoreCheckpointManager, SchedulerConfig, SpotPreemptionCheckpointingPolicy

from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
    add_common_args,
    compute_summary,
    configure_logging,
    write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR spot-agent real-host benchmark")
    parser.add_argument("--sandboxes", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--grace-period-seconds", type=float, default=60.0)
    add_common_args(parser)
    return parser.parse_args()


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
        scheduler_policy=SpotPreemptionCheckpointingPolicy(scheduler_config),
        checkpoint_manager_factory=lambda base: DeleteAfterRestoreCheckpointManager(base),
        max_workers=args.sandboxes,
    ) as harness:
        sandboxes = [harness.launch_sandbox(f"spot-{index}") for index in range(args.sandboxes)]
        for iteration in range(args.iters):
            for sandbox in sandboxes:
                harness.wait_for_progress(sandbox, minimum_actions=6)
                harness.set_snapshot_metadata(
                    sandbox,
                    preemption_notice=True,
                    preemption_grace_remaining_seconds=args.grace_period_seconds,
                )
                started = time.perf_counter()
                checkpoint_result = harness.checkpoint_if_due(sandbox)
                if checkpoint_result is None:
                    raise RuntimeError("spot benchmark expected a checkpoint after preemption notice")
                checkpoint_finished = time.perf_counter()
                restore_result = harness.restore_once(sandbox, checkpoint_result.checkpoint_id)
                restored = time.perf_counter()
                harness.poll_status(sandbox)
                harness.clear_snapshot_metadata(
                    sandbox,
                    "preemption_notice",
                    "preemption_grace_remaining_seconds",
                )
                rows.append(
                    {
                        "iter": iteration,
                        "sandbox_id": str(sandbox.sandbox_id),
                        "grace_period_ms": args.grace_period_seconds * 1000.0,
                        "checkpoint_ms": (checkpoint_finished - started) * 1000.0,
                        "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
                        "migration_ms": (restored - started) * 1000.0,
                        "budget_slack_ms": args.grace_period_seconds * 1000.0 - (restored - started) * 1000.0,
                        "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                    }
                )
    write_rows(args.out, rows)
    summary = compute_summary(rows, ["checkpoint_ms", "restore_ms", "migration_ms", "budget_slack_ms"])
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")


if __name__ == "__main__":
    main()

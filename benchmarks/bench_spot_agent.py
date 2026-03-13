#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import random
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import DeleteAfterRestoreCheckpointManager, SchedulerConfig, SpotPreemptionCheckpointingPolicy
from agent_cr.models import utc_now

from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
    add_common_args,
    bounded_probability,
    compute_summary,
    configure_logging,
    select_injected_indices,
    write_rows,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR spot-agent real-host benchmark")
    parser.add_argument("--sandboxes", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--grace-period-seconds", type=float, default=60.0)
    parser.add_argument("--auto-cr", action="store_true")
    parser.add_argument("--preemption-rate", type=bounded_probability, default=0.5)
    parser.add_argument("--first-preempt-iteration", type=int, default=0)
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
        auto_cr=args.auto_cr,
        work_dir_host_root=args.work_dir_host_root,
    ) as harness:
        sandboxes = [harness.launch_sandbox(f"spot-{index}") for index in range(args.sandboxes)]
        rng = random.Random(0)
        for iteration in range(1, args.iters + 1):
            logger.info("SpotAgent iteration=%d auto_cr=%s", iteration, args.auto_cr)
            if args.auto_cr:
                injected = set(
                    select_injected_indices(
                        len(sandboxes),
                        iteration=iteration,
                        rate=args.preemption_rate,
                        first_forced_iteration=args.first_preempt_iteration,
                        rng=rng,
                    )
                )
                logger.info(
                    "SpotAgent iteration=%d selected_preemption_targets=%s",
                    iteration,
                    [str(sandboxes[index].sandbox_id) for index in sorted(injected)],
                )
                for index, sandbox in enumerate(sandboxes):
                    harness.wait_for_progress(sandbox, minimum_actions=6)
                    if index not in injected:
                        rows.append(
                            {
                                "iter": iteration,
                                "sandbox_id": str(sandbox.sandbox_id),
                                "event_injected": 0,
                                "grace_period_ms": args.grace_period_seconds * 1000.0,
                                "recovery_status": "none",
                                "recovery_ms": 0.0,
                                "readiness_ms": 0.0,
                                "end_to_end_recovery_ms": 0.0,
                                "migration_ms": 0.0,
                                "budget_slack_ms": args.grace_period_seconds * 1000.0,
                                "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                            }
                        )
                        continue
                    event_started = time.perf_counter()
                    observed_after = utc_now()
                    logger.info(
                        "SpotAgent injecting preemption iteration=%d sandbox=%s grace_remaining_seconds=%.3f",
                        iteration,
                        sandbox.sandbox_id,
                        args.grace_period_seconds,
                    )
                    harness.notify_preemption(
                        sandbox,
                        grace_remaining_seconds=args.grace_period_seconds,
                    )
                    migration_started = time.perf_counter()
                    record = harness.wait_for_recovery(
                        sandbox,
                        event_type="preemption",
                        observed_after=observed_after,
                    )
                    recovery_finished = time.perf_counter()
                    harness.poll_status(sandbox)
                    ready_at = time.perf_counter()
                    logger.info(
                        "SpotAgent recovery finished iteration=%d sandbox=%s status=%s recovery_ms=%.3f readiness_ms=%.3f end_to_end_recovery_ms=%.3f",
                        iteration,
                        sandbox.sandbox_id,
                        record.status,
                        (recovery_finished - migration_started) * 1000.0,
                        (ready_at - recovery_finished) * 1000.0,
                        (ready_at - event_started) * 1000.0,
                    )
                    rows.append(
                        {
                            "iter": iteration,
                            "sandbox_id": str(sandbox.sandbox_id),
                            "event_injected": 1,
                            "grace_period_ms": args.grace_period_seconds * 1000.0,
                            "recovery_status": record.status,
                            "recovery_ms": (recovery_finished - migration_started) * 1000.0,
                            "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                            "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                            "migration_ms": (ready_at - event_started) * 1000.0,
                            "budget_slack_ms": args.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0,
                            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                        }
                    )
            else:
                for sandbox in sandboxes:
                    harness.wait_for_progress(sandbox, minimum_actions=6)
                    event_started = time.perf_counter()
                    harness.set_snapshot_metadata(
                        sandbox,
                        preemption_notice=True,
                        preemption_grace_remaining_seconds=args.grace_period_seconds,
                    )
                    recovery_started = time.perf_counter()
                    checkpoint_result = harness.checkpoint_if_due(sandbox)
                    if checkpoint_result is None:
                        raise RuntimeError("spot benchmark expected a checkpoint after preemption notice")
                    checkpoint_finished = time.perf_counter()
                    restore_result = harness.restore_once(sandbox, checkpoint_result.checkpoint_id)
                    recovery_finished = time.perf_counter()
                    harness.poll_status(sandbox)
                    ready_at = time.perf_counter()
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
                            "checkpoint_ms": (checkpoint_finished - recovery_started) * 1000.0,
                            "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
                            "recovery_ms": (recovery_finished - recovery_started) * 1000.0,
                            "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                            "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                            "migration_ms": (ready_at - event_started) * 1000.0,
                            "budget_slack_ms": args.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0,
                            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                        }
                    )
    write_rows(args.out, rows)
    if args.auto_cr:
        event_rows = [row for row in rows if int(row["event_injected"]) == 1]
        summary = (
            compute_summary(
                event_rows,
                ["recovery_ms", "readiness_ms", "end_to_end_recovery_ms", "migration_ms", "budget_slack_ms"],
            )
            if event_rows
            else {}
        )
    else:
        summary = compute_summary(
            rows,
            ["checkpoint_ms", "restore_ms", "recovery_ms", "readiness_ms", "end_to_end_recovery_ms", "migration_ms", "budget_slack_ms"],
        )
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")


if __name__ == "__main__":
    main()

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

from agent_cr import FaultToleranceCheckpointingPolicy, LatestOnlyCheckpointManager, SchedulerConfig
from agent_cr.models import utc_now

from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
    add_common_args,
    bounded_probability,
    compute_summary,
    configure_logging,
    select_injected_indices,
    total_actions,
    wait_for,
    write_rows,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR fault-tolerance real-host benchmark")
    parser.add_argument("--sandboxes", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--auto-cr", action="store_true")
    parser.add_argument("--fault-rate", type=bounded_probability, default=0.5)
    parser.add_argument("--first-fault-iteration", type=int, default=0)
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
        auto_cr=args.auto_cr,
    ) as harness:
        sandboxes = [harness.launch_sandbox(f"fault-{index}") for index in range(args.sandboxes)]
        rng = random.Random(0)
        for iteration in range(1, args.iters + 1):
            logger.info("FaultTolerance iteration=%d auto_cr=%s", iteration, args.auto_cr)
            if args.auto_cr:
                injected = set(
                    select_injected_indices(
                        len(sandboxes),
                        iteration=iteration,
                        rate=args.fault_rate,
                        first_forced_iteration=args.first_fault_iteration,
                        rng=rng,
                    )
                )
                logger.info(
                    "FaultTolerance iteration=%d selected_fault_targets=%s",
                    iteration,
                    [str(sandboxes[index].sandbox_id) for index in sorted(injected)],
                )
                for index, sandbox in enumerate(sandboxes):
                    current = harness.wait_for_progress(sandbox, minimum_actions=6)
                    if index not in injected:
                        rows.append(
                            {
                                "iter": iteration,
                                "sandbox_id": str(sandbox.sandbox_id),
                                "event_injected": 0,
                                "recovery_status": "none",
                                "recovery_ms": 0.0,
                                "readiness_ms": 0.0,
                                "end_to_end_recovery_ms": 0.0,
                                "pre_fault_actions": total_actions(current),
                                "post_recovery_actions": total_actions(current),
                                "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                            }
                        )
                        continue
                    pre_fault = harness.wait_for_action_delta(sandbox, delta=2)
                    logger.info(
                        "FaultTolerance injecting fault iteration=%d sandbox=%s pre_fault_actions=%d",
                        iteration,
                        sandbox.sandbox_id,
                        total_actions(pre_fault),
                    )
                    event_started = time.perf_counter()
                    harness.inject_fault(sandbox)
                    observed_after = utc_now()
                    harness.notify_fault(sandbox)
                    recovery_started = time.perf_counter()
                    record = harness.wait_for_recovery(
                        sandbox,
                        event_type="fault",
                        observed_after=observed_after,
                    )
                    recovery_finished = time.perf_counter()
                    post_recovery = harness.poll_status(sandbox)
                    ready_at = time.perf_counter()
                    sandbox.last_status = post_recovery
                    logger.info(
                        "FaultTolerance recovery finished iteration=%d sandbox=%s status=%s recovery_ms=%.3f readiness_ms=%.3f end_to_end_recovery_ms=%.3f",
                        iteration,
                        sandbox.sandbox_id,
                        record.status,
                        (recovery_finished - recovery_started) * 1000.0,
                        (ready_at - recovery_finished) * 1000.0,
                        (ready_at - event_started) * 1000.0,
                    )
                    rows.append(
                        {
                            "iter": iteration,
                            "sandbox_id": str(sandbox.sandbox_id),
                            "event_injected": 1,
                            "recovery_status": record.status,
                            "recovery_ms": (recovery_finished - recovery_started) * 1000.0,
                            "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                            "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                            "pre_fault_actions": total_actions(pre_fault),
                            "post_recovery_actions": total_actions(post_recovery),
                            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                        }
                    )
            else:
                for sandbox in sandboxes:
                    current = harness.wait_for_progress(sandbox, minimum_actions=6)
                    checkpoint_actions = total_actions(current)
                    t0 = time.perf_counter()
                    checkpoint_result = harness.checkpoint_if_due(sandbox)
                    t1 = time.perf_counter()
                    if checkpoint_result is None:
                        continue
                    pre_fault = harness.wait_for_action_delta(sandbox, delta=2)
                    event_started = time.perf_counter()
                    harness.inject_fault(sandbox)
                    recovery_started = time.perf_counter()
                    restore_result = harness.restore_once(sandbox, checkpoint_result.checkpoint_id)
                    recovery_finished = time.perf_counter()
                    restored_status = harness.poll_status(sandbox)
                    ready_at = time.perf_counter()
                    sandbox.last_status = restored_status
                    workload_resume_started = time.perf_counter()
                    wait_for(lambda: total_actions(harness.poll_status(sandbox)) >= checkpoint_actions, timeout_s=45.0)
                    post_restore = harness.wait_for_action_delta(sandbox, delta=1)
                    workload_resumed_at = time.perf_counter()
                    rows.append(
                        {
                            "iter": iteration,
                            "sandbox_id": str(sandbox.sandbox_id),
                            "checkpoint_ms": (t1 - t0) * 1000.0,
                            "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
                            "recovery_ms": (recovery_finished - recovery_started) * 1000.0,
                            "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                            "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                            "workload_resume_ms": (workload_resumed_at - workload_resume_started) * 1000.0,
                            "checkpoint_actions": checkpoint_actions,
                            "pre_fault_actions": total_actions(pre_fault),
                            "post_restore_actions": total_actions(post_restore),
                            "lost_actions": max(0, total_actions(pre_fault) - checkpoint_actions),
                            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                        }
                    )
    write_rows(args.out, rows)
    if args.auto_cr:
        event_rows = [row for row in rows if int(row["event_injected"]) == 1]
        summary = (
            compute_summary(event_rows, ["recovery_ms", "readiness_ms", "end_to_end_recovery_ms"])
            if event_rows
            else {}
        )
    else:
        summary = compute_summary(
            rows,
            [
                "checkpoint_ms",
                "restore_ms",
                "recovery_ms",
                "readiness_ms",
                "end_to_end_recovery_ms",
                "workload_resume_ms",
                "lost_actions",
            ],
        )
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")


if __name__ == "__main__":
    main()

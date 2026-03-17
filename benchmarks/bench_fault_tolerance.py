#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import random
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import FaultToleranceCheckpointingPolicy, JobStatus, LatestOnlyCheckpointManager, SchedulerConfig
from agent_cr.models import utc_now

from integrations.agents import SandboxHandle, TaskConfig, TaskDescription
from benchmarks.support import (
    add_common_args,
    bounded_probability,
    compute_summary,
    total_actions,
    wait_for,
    write_rows,
)
from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
)
from benchmarks.support import configure_logging

logger = logging.getLogger(__name__)


def benchmark_task_description() -> TaskDescription:
    return TaskDescription("Continuously work on the benchmark task inside the sandbox.")


def default_task_config() -> TaskConfig:
    return TaskConfig(minimum_actions=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR fault-tolerance real-host benchmark")
    parser.add_argument("--sandboxes", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--auto-cr", action="store_true")
    parser.add_argument("--fault-rate", type=bounded_probability, default=0.5)
    parser.add_argument("--first-fault-iteration", type=int, default=0)
    add_common_args(parser)
    return parser.parse_args()


def should_inject_fault(
    *,
    iteration: int,
    sandbox_index: int,
    rate: float,
    first_forced_iteration: int,
    rng: random.Random,
) -> bool:
    if first_forced_iteration > 0 and iteration < first_forced_iteration:
        return False
    if first_forced_iteration > 0 and iteration == first_forced_iteration and sandbox_index == 0:
        return True
    return rng.random() < rate


def run_fault_tolerance_sandbox(
    args: argparse.Namespace,
    harness: RealHostScenarioHarness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = random.Random(sandbox_index)
    for iteration in range(1, args.iters + 1):
        logger.info(
            "FaultTolerance sandbox=%s iteration=%d auto_cr=%s",
            sandbox.sandbox_id,
            iteration,
            args.auto_cr,
        )
        if args.auto_cr:
            assert sandbox.task_run is not None
            current = sandbox.task_run.wait_for_progress(minimum_actions=6)
            pre_fault = sandbox.task_run.wait_for_action_delta(delta=2)
            injected = should_inject_fault(
                iteration=iteration,
                sandbox_index=sandbox_index,
                rate=args.fault_rate,
                first_forced_iteration=args.first_fault_iteration,
                rng=rng,
            )
            if not injected:
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
            assert sandbox.task_run is not None
            post_recovery = sandbox.task_run.poll_status()
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
            assert sandbox.task_run is not None
            current = sandbox.task_run.wait_for_progress(minimum_actions=6)
            checkpoint_actions = total_actions(current)
            t0 = time.perf_counter()
            checkpoint_result = harness.checkpoint_if_due(sandbox)
            t1 = time.perf_counter()
            if checkpoint_result is None:
                continue
            if checkpoint_result.status != JobStatus.SUCCEEDED:
                logger.warning(
                    "FaultTolerance checkpoint failed iteration=%d sandbox=%s status=%s checkpoint=%s message=%s",
                    iteration,
                    sandbox.sandbox_id,
                    checkpoint_result.status.value,
                    checkpoint_result.checkpoint_id,
                    checkpoint_result.message,
                )
                continue
            pre_fault = sandbox.task_run.wait_for_action_delta(delta=2)
            event_started = time.perf_counter()
            harness.inject_fault(sandbox)
            recovery_started = time.perf_counter()
            restore_result = harness.restore_once(sandbox, checkpoint_result.checkpoint_id)
            recovery_finished = time.perf_counter()
            restored_status = sandbox.task_run.poll_status()
            ready_at = time.perf_counter()
            sandbox.last_status = restored_status
            workload_resume_started = time.perf_counter()
            wait_for(lambda: total_actions(sandbox.task_run.poll_status()) >= checkpoint_actions, timeout_s=45.0)
            post_restore = sandbox.task_run.wait_for_action_delta(delta=1)
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
    return rows


def run_fault_tolerance_benchmark(
    args: argparse.Namespace,
    harness: RealHostScenarioHarness,
) -> list[dict[str, object]]:
    dataset_path = getattr(args, "dataset", None)
    dataset = harness.load_dataset(dataset_path) if dataset_path is not None else None
    with ThreadPoolExecutor(max_workers=max(1, args.sandboxes)) as launcher:
        sandboxes = list(
            launcher.map(
                lambda index: harness.launch_task_record(
                    f"fault-{index}",
                    harness.select_task_record(
                        dataset,
                        sandbox_index=index,
                        default_agent_type=args.agent_type,
                        default_llm_service_type=args.llm_service_type,
                        default_task_description=benchmark_task_description(),
                        default_task_config=default_task_config(),
                    ),
                ),
                range(args.sandboxes),
            )
        )
    with ThreadPoolExecutor(max_workers=max(1, args.sandboxes)) as executor:
        row_groups = list(
            executor.map(
                lambda item: run_fault_tolerance_sandbox(
                    args,
                    harness,
                    sandbox_index=item[0],
                    sandbox=item[1],
                ),
                enumerate(sandboxes),
            )
        )
    rows = [row for group in row_groups for row in group]
    return sorted(rows, key=lambda row: (int(row["iter"]), str(row["sandbox_id"])))


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
    with RealHostScenarioHarness(
        provider=args.provider,
        transfer_delay_ms=args.transfer_delay_ms,
        scheduler_config=scheduler_config,
        scheduler_policy=FaultToleranceCheckpointingPolicy(scheduler_config),
        checkpoint_manager_factory=lambda base: LatestOnlyCheckpointManager(base),
        max_workers=args.sandboxes,
        auto_cr=args.auto_cr,
        work_dir_host_root=args.work_dir_host_root,
    ) as harness:
        rows = run_fault_tolerance_benchmark(args, harness)
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

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

from agent_cr import DeleteAfterRestoreCheckpointManager, SchedulerConfig, SpotPreemptionCheckpointingPolicy
from agent_cr.models import utc_now

from integrations.agents import SandboxHandle, TaskConfig, TaskDescription
from benchmarks.support import (
    add_common_args,
    average,
    bounded_probability,
    choose_replay_points,
    compute_summary,
    configure_logging,
    is_replay_llm_service_type,
    task_timeout_seconds,
    verification_timeout_seconds,
    write_rows,
)
from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
)

logger = logging.getLogger(__name__)


def benchmark_task_description() -> TaskDescription:
    return TaskDescription("Continuously work on the benchmark task inside the sandbox.")


def default_task_config() -> TaskConfig:
    return TaskConfig()


def _sandbox_benchmark_metadata(sandbox: SandboxHandle) -> dict[str, object]:
    metadata = sandbox.launch_metadata.get("benchmark", {})
    return metadata if isinstance(metadata, dict) else {}


def _task_id_for_sandbox(sandbox: SandboxHandle) -> str:
    metadata = _sandbox_benchmark_metadata(sandbox)
    raw_task_id = metadata.get("task_id")
    if isinstance(raw_task_id, str) and raw_task_id:
        return raw_task_id
    if sandbox.task_config is not None:
        raw_task_id = sandbox.task_config.options.get("task_id")
        if isinstance(raw_task_id, str) and raw_task_id:
            return raw_task_id
    return str(sandbox.sandbox_id)


def _trace_response_count_for_sandbox(sandbox: SandboxHandle) -> int:
    metadata = _sandbox_benchmark_metadata(sandbox)
    raw_value = metadata.get("trace_response_count")
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def _finalize_replay_row(
    args: argparse.Namespace,
    harness: RealHostScenarioHarness,
    sandbox: SandboxHandle,
    *,
    row: dict[str, object],
    task_error: str = "",
) -> dict[str, object]:
    verification = {
        "verification_status": "task_failed" if task_error else "verification_skipped",
        "verification_exit_code": -1,
        "verification_ms": 0.0,
    }
    if not task_error:
        try:
            harness.wait_for_task_completion(
                sandbox,
                timeout_s=task_timeout_seconds(sandbox.task_config or TaskConfig()),
            )
        except Exception as exc:
            task_error = str(exc)
            verification["verification_status"] = "task_failed"
        else:
            try:
                verification = harness.verify_task_accuracy(
                    sandbox,
                    timeout_s=verification_timeout_seconds(sandbox.task_config or TaskConfig()),
                )
            except Exception as exc:
                verification = {
                    "verification_status": "verification_error",
                    "verification_exit_code": -1,
                    "verification_ms": 0.0,
                    "verification_stdout": "",
                    "verification_stderr": str(exc),
                    "verification_command": "",
                }
    status = dict(sandbox.last_status)
    if sandbox.task_run is not None:
        try:
            status = sandbox.task_run.poll_status()
        except Exception:
            status = dict(sandbox.last_status)
    sandbox.last_status = dict(status)
    return {
        **row,
        "provider": args.provider,
        "agent_type": sandbox.agent_type,
        "sandbox_id": str(sandbox.sandbox_id),
        "task_id": _task_id_for_sandbox(sandbox),
        "trace_response_count": _trace_response_count_for_sandbox(sandbox),
        "replay_final_index": int(status.get("replay_next_response_index", status.get("total_actions", 0))),
        "task_error": task_error,
        "success_ratio": 1.0 if verification["verification_status"] == "passed" else 0.0,
        **verification,
    }


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


def should_inject_preemption(
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


def wait_for_iteration_progress(
    harness: RealHostScenarioHarness,
    sandbox: SandboxHandle,
    *,
    iteration: int,
) -> dict[str, object]:
    assert sandbox.task_run is not None
    if iteration == 1:
        return sandbox.task_run.wait_for_progress(minimum_actions=6)
    return sandbox.task_run.wait_for_action_delta(delta=1)


def run_spot_agent_sandbox(
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
            "SpotAgent sandbox=%s iteration=%d auto_cr=%s",
            sandbox.sandbox_id,
            iteration,
            args.auto_cr,
        )
        if args.auto_cr:
            wait_for_iteration_progress(harness, sandbox, iteration=iteration)
            injected = should_inject_preemption(
                iteration=iteration,
                sandbox_index=sandbox_index,
                rate=args.preemption_rate,
                first_forced_iteration=args.first_preempt_iteration,
                rng=rng,
            )
            if not injected:
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
            ready_at = recovery_finished
            if record.status in {"restored", "relaunched"}:
                assert sandbox.task_run is not None
                sandbox.last_status = sandbox.task_run.poll_status()
                ready_at = time.perf_counter()
            else:
                logger.warning(
                    "SpotAgent recovery did not restore sandbox=%s iteration=%d status=%s",
                    sandbox.sandbox_id,
                    iteration,
                    record.status,
                )
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
            if record.status not in {"restored", "relaunched"}:
                break
        else:
            wait_for_iteration_progress(harness, sandbox, iteration=iteration)
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
            assert sandbox.task_run is not None
            sandbox.task_run.poll_status()
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
    return rows


def run_replay_spot_agent_sandbox(
    args: argparse.Namespace,
    harness: RealHostScenarioHarness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
) -> dict[str, object]:
    trace_response_count = _trace_response_count_for_sandbox(sandbox)
    replay_points = choose_replay_points(trace_response_count, args.iters)
    rng = random.Random(sandbox_index)
    checkpoint_ms_values: list[float] = []
    restore_ms_values: list[float] = []
    recovery_ms_values: list[float] = []
    readiness_ms_values: list[float] = []
    end_to_end_recovery_ms_values: list[float] = []
    migration_ms_values: list[float] = []
    budget_slack_ms_values: list[float] = []
    iterations_executed = 0
    preemptions_injected = 0
    recoveries_succeeded = 0
    task_error = ""

    try:
        if sandbox.task_run is None:
            raise RuntimeError("replay spot benchmark expected sandbox.task_run")
        for iteration, checkpoint_target in enumerate(replay_points, start=1):
            iterations_executed += 1
            sandbox.task_run.wait_for_progress(minimum_actions=checkpoint_target)
            injected = should_inject_preemption(
                iteration=iteration,
                sandbox_index=sandbox_index,
                rate=args.preemption_rate,
                first_forced_iteration=args.first_preempt_iteration,
                rng=rng,
            )
            if not injected:
                continue
            preemptions_injected += 1
            event_started = time.perf_counter()
            if args.auto_cr:
                observed_after = utc_now()
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
                ready_at = recovery_finished
                if record.status in {"restored", "relaunched"}:
                    sandbox.last_status = sandbox.task_run.poll_status()
                    ready_at = time.perf_counter()
                    recoveries_succeeded += 1
                recovery_ms_values.append((recovery_finished - migration_started) * 1000.0)
                readiness_ms_values.append((ready_at - recovery_finished) * 1000.0)
                end_to_end_recovery_ms_values.append((ready_at - event_started) * 1000.0)
                migration_ms_values.append((ready_at - event_started) * 1000.0)
                budget_slack_ms_values.append(args.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0)
                if record.status not in {"restored", "relaunched"}:
                    raise RuntimeError(f"preemption recovery failed with status={record.status}")
                continue

            harness.set_snapshot_metadata(
                sandbox,
                preemption_notice=True,
                preemption_grace_remaining_seconds=args.grace_period_seconds,
            )
            recovery_started = time.perf_counter()
            checkpoint_result = harness.checkpoint_manual(sandbox, leave_running=True)
            checkpoint_finished = time.perf_counter()
            if checkpoint_result.status.value != "succeeded":
                raise RuntimeError(
                    f"checkpoint failed for sandbox {sandbox.sandbox_id}: "
                    f"{checkpoint_result.status.value} {checkpoint_result.message}"
                )
            restore_result = harness.restore_once(sandbox, checkpoint_result.checkpoint_id)
            recovery_finished = time.perf_counter()
            sandbox.task_run.poll_status()
            ready_at = time.perf_counter()
            harness.clear_snapshot_metadata(
                sandbox,
                "preemption_notice",
                "preemption_grace_remaining_seconds",
            )
            checkpoint_ms_values.append((checkpoint_finished - recovery_started) * 1000.0)
            restore_ms_values.append(
                (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0
            )
            recovery_ms_values.append((recovery_finished - recovery_started) * 1000.0)
            readiness_ms_values.append((ready_at - recovery_finished) * 1000.0)
            end_to_end_recovery_ms_values.append((ready_at - event_started) * 1000.0)
            migration_ms_values.append((ready_at - event_started) * 1000.0)
            budget_slack_ms_values.append(args.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0)
            recoveries_succeeded += 1
    except Exception as exc:
        task_error = str(exc)

    return _finalize_replay_row(
        args,
        harness,
        sandbox,
        row={
            "iterations_planned": len(replay_points),
            "iterations_executed": iterations_executed,
            "preemptions_injected": preemptions_injected,
            "recoveries_succeeded": recoveries_succeeded,
            "grace_period_ms": args.grace_period_seconds * 1000.0,
            "checkpoint_ms_avg": average(checkpoint_ms_values),
            "restore_ms_avg": average(restore_ms_values),
            "recovery_ms_avg": average(recovery_ms_values),
            "readiness_ms_avg": average(readiness_ms_values),
            "end_to_end_recovery_ms_avg": average(end_to_end_recovery_ms_values),
            "migration_ms_avg": average(migration_ms_values),
            "budget_slack_ms_avg": average(budget_slack_ms_values),
            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            "skipped_no_replay_checkpoint": 1 if not replay_points else 0,
        },
        task_error=task_error,
    )


def run_spot_agent_benchmark(
    args: argparse.Namespace,
    harness: RealHostScenarioHarness,
) -> list[dict[str, object]]:
    dataset_path = getattr(args, "dataset", None)
    dataset = harness.load_dataset(dataset_path) if dataset_path is not None else None
    with ThreadPoolExecutor(max_workers=max(1, args.sandboxes)) as launcher:
        sandboxes = list(
            launcher.map(
                lambda index: harness.launch_task_record(
                    f"spot-{index}",
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
    if sandboxes and any(is_replay_llm_service_type(sandbox.llm_service_type) for sandbox in sandboxes):
        with ThreadPoolExecutor(max_workers=max(1, args.sandboxes)) as executor:
            rows = list(
                executor.map(
                    lambda item: run_replay_spot_agent_sandbox(
                        args,
                        harness,
                        sandbox_index=item[0],
                        sandbox=item[1],
                    ),
                    enumerate(sandboxes),
                )
            )
        return sorted(rows, key=lambda row: str(row["sandbox_id"]))
    with ThreadPoolExecutor(max_workers=max(1, args.sandboxes)) as executor:
        row_groups = list(
            executor.map(
                lambda item: run_spot_agent_sandbox(
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
        require_change_signal=False,
    )
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
        rows = run_spot_agent_benchmark(args, harness)
    write_rows(args.out, rows)
    if rows and "verification_status" in rows[0]:
        summary = compute_summary(
            rows,
            [
                "checkpoint_ms_avg",
                "restore_ms_avg",
                "recovery_ms_avg",
                "readiness_ms_avg",
                "end_to_end_recovery_ms_avg",
                "migration_ms_avg",
                "budget_slack_ms_avg",
                "success_ratio",
            ],
        )
        for key, value in summary.items():
            print(f"{key}_avg: {value:.3f}")
        return
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

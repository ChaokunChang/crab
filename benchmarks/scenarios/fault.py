from __future__ import annotations

from dataclasses import dataclass
import random
import time

from agent_cr import FaultToleranceCheckpointingPolicy, JobStatus, LatestOnlyCheckpointManager, SchedulerConfig
from agent_cr.models import utc_now

from integrations.agents import SandboxHandle, TaskConfig, TaskDescription

from benchmarks.config import BenchmarkConfig
from benchmarks.core import (
    annotate_row,
    launch_task_records,
    parallel_map,
    resolve_task_records,
    trace_response_count_for_sandbox,
)
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from benchmarks.scenarios.common import (
    choose_replay_targets,
    finalize_replay_row,
    replay_status_is_complete,
    should_inject_event,
    wait_for_auto_replay_checkpoint,
)
from benchmarks.support import average, compute_summary, is_replay_llm_service_type, total_actions, wait_for


@dataclass(frozen=True)
class FaultOptions:
    injection_rate: float = 0.5
    first_injection_iteration: int = 0


def benchmark_task_description() -> TaskDescription:
    return TaskDescription("Continuously work on the benchmark task inside the sandbox.")


def default_task_config() -> TaskConfig:
    return TaskConfig()


def parse_fault_options(config: BenchmarkConfig) -> FaultOptions:
    rate = float(config.scenario_options.get("injection_rate", 0.5))
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"scenario_options.injection_rate must be in [0.0, 1.0], got {rate}")
    first = int(config.scenario_options.get("first_injection_iteration", 0))
    if first < 0:
        raise ValueError("scenario_options.first_injection_iteration must be >= 0")
    return FaultOptions(injection_rate=rate, first_injection_iteration=first)


def build_harness_settings(config: BenchmarkConfig) -> HarnessSettings:
    scheduler_config = SchedulerConfig(
        min_checkpoint_interval_seconds=0.0,
        force_checkpoint_after_seconds=0.0,
        require_change_signal=True,
        prefer_checkpoint_during_llm_request=True,
        require_llm_request_for_checkpoint=False,
    )
    return HarnessSettings(
        scheduler_config=scheduler_config,
        scheduler_policy=FaultToleranceCheckpointingPolicy(scheduler_config),
        checkpoint_manager_factory=lambda base: LatestOnlyCheckpointManager(base),
        max_workers=config.sandboxes,
    )


def run_manual_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox: SandboxHandle,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for iteration in range(1, config.iterations + 1):
        assert sandbox.task_run is not None
        current = sandbox.task_run.wait_for_progress(minimum_actions=6)
        checkpoint_actions = total_actions(current)
        checkpoint_started = time.perf_counter()
        checkpoint_result = harness.checkpoint_if_due(sandbox)
        checkpoint_finished = time.perf_counter()
        if checkpoint_result is None:
            continue
        if checkpoint_result.status != JobStatus.SUCCEEDED:
            continue
        pre_event = sandbox.task_run.wait_for_action_delta(delta=2)
        event_started = time.perf_counter()
        harness.inject_fault(sandbox)
        recovery_started = time.perf_counter()
        restore_result = harness.restore_once(sandbox, checkpoint_result.checkpoint_id)
        recovery_finished = time.perf_counter()
        post_recovery = sandbox.task_run.poll_status()
        ready_at = time.perf_counter()
        sandbox.last_status = post_recovery
        workload_resume_started = time.perf_counter()
        wait_for(lambda: total_actions(sandbox.task_run.poll_status()) >= checkpoint_actions, timeout_s=45.0)
        post_resume = sandbox.task_run.wait_for_action_delta(delta=1)
        workload_resumed_at = time.perf_counter()
        success_ratio = 1.0 if restore_result.status.value == "succeeded" else 0.0
        rows.append(
            annotate_row(
                config,
                sandbox,
                iteration=iteration,
                success_ratio=success_ratio,
                row={
                    "event_type": "fault",
                    "event_injected": 1,
                    "recovery_status": restore_result.status.value,
                    "checkpoint_ms": (checkpoint_finished - checkpoint_started) * 1000.0,
                    "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
                    "recovery_ms": (recovery_finished - recovery_started) * 1000.0,
                    "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                    "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                    "workload_resume_ms": (workload_resumed_at - workload_resume_started) * 1000.0,
                    "checkpoint_actions": checkpoint_actions,
                    "pre_event_actions": total_actions(pre_event),
                    "post_recovery_actions": total_actions(post_resume),
                    "lost_actions": max(0, total_actions(pre_event) - checkpoint_actions),
                    "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                },
            )
        )
    return rows


def run_auto_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: FaultOptions,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = random.Random(sandbox_index)
    for iteration in range(1, config.iterations + 1):
        assert sandbox.task_run is not None
        current = sandbox.task_run.wait_for_progress(minimum_actions=6)
        pre_event = sandbox.task_run.wait_for_action_delta(delta=2)
        injected = should_inject_event(
            iteration=iteration,
            sandbox_index=sandbox_index,
            rate=options.injection_rate,
            first_injection_iteration=options.first_injection_iteration,
            rng=rng,
        )
        if not injected:
            rows.append(
                annotate_row(
                    config,
                    sandbox,
                    iteration=iteration,
                    success_ratio=1.0,
                    row={
                        "event_type": "fault",
                        "event_injected": 0,
                        "recovery_status": "none",
                        "checkpoint_ms": 0.0,
                        "restore_ms": 0.0,
                        "recovery_ms": 0.0,
                        "readiness_ms": 0.0,
                        "end_to_end_recovery_ms": 0.0,
                        "workload_resume_ms": 0.0,
                        "checkpoint_actions": 0,
                        "pre_event_actions": total_actions(current),
                        "post_recovery_actions": total_actions(current),
                        "lost_actions": 0,
                        "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                    },
                )
            )
            continue
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
        success_ratio = 1.0 if record.status in {"restored", "relaunched"} else 0.0
        ready_at = recovery_finished
        post_recovery = dict(sandbox.last_status)
        if success_ratio > 0.0:
            post_recovery = sandbox.task_run.poll_status()
            ready_at = time.perf_counter()
            sandbox.last_status = post_recovery
        rows.append(
            annotate_row(
                config,
                sandbox,
                iteration=iteration,
                success_ratio=success_ratio,
                row={
                    "event_type": "fault",
                    "event_injected": 1,
                    "recovery_status": record.status,
                    "checkpoint_ms": 0.0,
                    "restore_ms": 0.0,
                    "recovery_ms": (recovery_finished - recovery_started) * 1000.0,
                    "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                    "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                    "workload_resume_ms": 0.0,
                    "checkpoint_actions": 0,
                    "pre_event_actions": total_actions(pre_event),
                    "post_recovery_actions": total_actions(post_recovery),
                    "lost_actions": max(0, total_actions(pre_event) - total_actions(post_recovery)),
                    "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                },
            )
        )
        if success_ratio == 0.0:
            break
    return rows


def run_replay_manual_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: FaultOptions,
) -> dict[str, object]:
    replay_points = choose_replay_targets(sandbox, config.iterations)
    rng = random.Random(sandbox_index)
    checkpoint_ms_values: list[float] = []
    restore_ms_values: list[float] = []
    recovery_ms_values: list[float] = []
    readiness_ms_values: list[float] = []
    end_to_end_recovery_ms_values: list[float] = []
    lost_actions_values: list[float] = []
    iterations_executed = 0
    events_injected = 0
    recoveries_succeeded = 0
    task_error = ""

    try:
        if sandbox.task_run is None:
            raise RuntimeError("replay fault benchmark expected sandbox.task_run")
        for iteration, checkpoint_target in enumerate(replay_points, start=1):
            iterations_executed += 1
            try:
                current = sandbox.task_run.wait_for_progress(minimum_actions=checkpoint_target)
            except RuntimeError as exc:
                current = sandbox.task_run.poll_status()
                if (
                    "iflow replay task finished before reaching replay action count" in str(exc)
                    and str(current.get("state", "")) == "finished"
                ):
                    sandbox.last_status = dict(current)
                    break
                raise
            if replay_status_is_complete(
                current,
                trace_response_count=trace_response_count_for_sandbox(sandbox),
            ):
                sandbox.last_status = dict(current)
                break
            injected = should_inject_event(
                iteration=iteration,
                sandbox_index=sandbox_index,
                rate=options.injection_rate,
                first_injection_iteration=options.first_injection_iteration,
                rng=rng,
            )
            if not injected:
                continue
            events_injected += 1
            checkpoint_started = time.perf_counter()
            checkpoint_result = harness.checkpoint_manual(sandbox, leave_running=True)
            checkpoint_finished = time.perf_counter()
            checkpoint_ms_values.append((checkpoint_finished - checkpoint_started) * 1000.0)
            if checkpoint_result.status != JobStatus.SUCCEEDED:
                raise RuntimeError(
                    f"checkpoint failed for sandbox {sandbox.sandbox_id}: "
                    f"{checkpoint_result.status.value} {checkpoint_result.message}"
                )
            pre_event = sandbox.task_run.wait_for_action_delta(delta=1)
            event_started = time.perf_counter()
            harness.inject_fault(sandbox)
            recovery_started = time.perf_counter()
            restore_result = harness.restore_once(sandbox, checkpoint_result.checkpoint_id)
            recovery_finished = time.perf_counter()
            post_recovery = sandbox.task_run.poll_status()
            ready_at = time.perf_counter()
            if restore_result.status.value == "succeeded":
                recoveries_succeeded += 1
            restore_ms_values.append(
                (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0
            )
            recovery_ms_values.append((recovery_finished - recovery_started) * 1000.0)
            readiness_ms_values.append((ready_at - recovery_finished) * 1000.0)
            end_to_end_recovery_ms_values.append((ready_at - event_started) * 1000.0)
            lost_actions_values.append(max(0, total_actions(pre_event) - checkpoint_target))
            sandbox.last_status = post_recovery
    except Exception as exc:
        task_error = str(exc)

    return finalize_replay_row(
        config,
        harness,
        sandbox,
        row={
            "event_type": "fault",
            "iterations_planned": len(replay_points),
            "iterations_executed": iterations_executed,
            "events_injected": events_injected,
            "recoveries_succeeded": recoveries_succeeded,
            "checkpoint_ms_avg": average(checkpoint_ms_values),
            "restore_ms_avg": average(restore_ms_values),
            "recovery_ms_avg": average(recovery_ms_values),
            "readiness_ms_avg": average(readiness_ms_values),
            "end_to_end_recovery_ms_avg": average(end_to_end_recovery_ms_values),
            "lost_actions_avg": average(lost_actions_values),
            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            "skipped_no_replay_checkpoint": 1 if not replay_points else 0,
        },
        task_error=task_error,
    )


def run_replay_auto_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: FaultOptions,
) -> dict[str, object]:
    replay_points = choose_replay_targets(sandbox, config.iterations)
    rng = random.Random(sandbox_index)
    recovery_ms_values: list[float] = []
    readiness_ms_values: list[float] = []
    end_to_end_recovery_ms_values: list[float] = []
    lost_actions_values: list[float] = []
    iterations_executed = 0
    events_injected = 0
    recoveries_succeeded = 0
    task_error = ""

    try:
        if sandbox.task_run is None:
            raise RuntimeError("replay fault benchmark expected sandbox.task_run")
        for iteration, checkpoint_target in enumerate(replay_points, start=1):
            iterations_executed += 1
            try:
                current = sandbox.task_run.wait_for_progress(minimum_actions=checkpoint_target)
            except RuntimeError as exc:
                current = sandbox.task_run.poll_status()
                if (
                    "iflow replay task finished before reaching replay action count" in str(exc)
                    and str(current.get("state", "")) == "finished"
                ):
                    sandbox.last_status = dict(current)
                    break
                raise
            if replay_status_is_complete(
                current,
                trace_response_count=trace_response_count_for_sandbox(sandbox),
            ):
                sandbox.last_status = dict(current)
                break
            injected = should_inject_event(
                iteration=iteration,
                sandbox_index=sandbox_index,
                rate=options.injection_rate,
                first_injection_iteration=options.first_injection_iteration,
                rng=rng,
            )
            if not injected:
                continue
            events_injected += 1
            checkpoint_manifest, checkpoint_actions = wait_for_auto_replay_checkpoint(
                harness,
                sandbox,
                minimum_actions=checkpoint_target,
                trace_response_count=trace_response_count_for_sandbox(sandbox),
            )
            if checkpoint_manifest is None:
                events_injected -= 1
                sandbox.last_status = dict(sandbox.task_run.poll_status())
                break
            pre_event = sandbox.task_run.poll_status()
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
            ready_at = recovery_finished
            post_recovery = dict(sandbox.last_status)
            if record.status in {"restored", "relaunched"}:
                post_recovery = sandbox.task_run.poll_status()
                ready_at = time.perf_counter()
                recoveries_succeeded += 1
            else:
                raise RuntimeError(f"fault recovery failed with status={record.status}")
            recovery_ms_values.append((recovery_finished - recovery_started) * 1000.0)
            readiness_ms_values.append((ready_at - recovery_finished) * 1000.0)
            end_to_end_recovery_ms_values.append((ready_at - event_started) * 1000.0)
            lost_actions_values.append(max(0, total_actions(pre_event) - total_actions(post_recovery)))
            _ = checkpoint_actions
            sandbox.last_status = post_recovery
    except Exception as exc:
        task_error = str(exc)

    return finalize_replay_row(
        config,
        harness,
        sandbox,
        row={
            "event_type": "fault",
            "iterations_planned": len(replay_points),
            "iterations_executed": iterations_executed,
            "events_injected": events_injected,
            "recoveries_succeeded": recoveries_succeeded,
            "checkpoint_ms_avg": 0.0,
            "restore_ms_avg": 0.0,
            "recovery_ms_avg": average(recovery_ms_values),
            "readiness_ms_avg": average(readiness_ms_values),
            "end_to_end_recovery_ms_avg": average(end_to_end_recovery_ms_values),
            "lost_actions_avg": average(lost_actions_values),
            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            "skipped_no_replay_checkpoint": 1 if not replay_points else 0,
        },
        task_error=task_error,
    )


def run_manual(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    if any(is_replay_llm_service_type(record.llm_service_type) for record in records):
        indexed_records = list(enumerate(records))

        def launch_and_run(item: tuple[int, object]) -> list[dict[str, object]]:
            sandbox_index, task_record = item
            sandbox = harness.launch_task_record(f"fault-{sandbox_index}", task_record)
            if is_replay_llm_service_type(sandbox.llm_service_type):
                return [run_replay_manual_sandbox(config, harness, sandbox_index=sandbox_index, sandbox=sandbox, options=parse_fault_options(config))]
            return run_manual_sandbox(config, harness, sandbox=sandbox)

        row_groups = parallel_map(indexed_records, launch_and_run, max_workers=config.sandboxes)
        rows = [row for group in row_groups for row in group]
        return sorted(rows, key=lambda row: (int(row["iteration"]), str(row["sandbox_id"])))

    sandboxes = launch_task_records(
        harness,
        sandbox_name_prefix="fault",
        records=records,
        max_workers=config.sandboxes,
    )
    row_groups = parallel_map(
        sandboxes,
        lambda sandbox: run_manual_sandbox(config, harness, sandbox=sandbox),
        max_workers=config.sandboxes,
    )
    rows = [row for group in row_groups for row in group]
    return sorted(rows, key=lambda row: (int(row["iteration"]), str(row["sandbox_id"])))


def run_auto(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    options = parse_fault_options(config)
    indexed_records = list(enumerate(records))

    def launch_and_run(item: tuple[int, object]) -> list[dict[str, object]]:
        sandbox_index, task_record = item
        sandbox = harness.launch_task_record(f"fault-{sandbox_index}", task_record)
        if is_replay_llm_service_type(sandbox.llm_service_type):
            return [run_replay_auto_sandbox(config, harness, sandbox_index=sandbox_index, sandbox=sandbox, options=options)]
        return run_auto_sandbox(config, harness, sandbox_index=sandbox_index, sandbox=sandbox, options=options)

    row_groups = parallel_map(indexed_records, launch_and_run, max_workers=config.sandboxes)
    rows = [row for group in row_groups for row in group]
    return sorted(rows, key=lambda row: (int(row["iteration"]), str(row["sandbox_id"])))


def summarize(config: BenchmarkConfig, rows: list[dict[str, object]]) -> dict[str, float]:
    if rows and "verification_status" in rows[0]:
        return compute_summary(
            rows,
            [
                "checkpoint_ms_avg",
                "restore_ms_avg",
                "recovery_ms_avg",
                "readiness_ms_avg",
                "end_to_end_recovery_ms_avg",
                "lost_actions_avg",
                "success_ratio",
            ],
        )
    if config.mode == "auto":
        event_rows = [row for row in rows if int(row["event_injected"]) == 1]
        return compute_summary(
            event_rows,
            ["recovery_ms", "readiness_ms", "end_to_end_recovery_ms"],
        ) if event_rows else {}
    return compute_summary(
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


SCENARIO = ScenarioDefinition(
    name="fault",
    supported_modes=frozenset({"manual", "auto"}),
    build_harness_settings=build_harness_settings,
    run_manual=run_manual,
    run_auto=run_auto,
    summarize=summarize,
)

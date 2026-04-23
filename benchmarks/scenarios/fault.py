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
    benchmark_phase_item_attributes,
    benchmark_phase_map,
    benchmark_setup_run_pipeline,
    emit_benchmark_phase_skipped,
    emit_row_telemetry,
    make_benchmark_sandbox_specs,
    resolve_task_records,
    setup_task_records_phase,
    start_prepared_task_record,
    trace_response_count_for_sandbox,
)
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from benchmarks.scenarios.common import (
    choose_replay_chunk_targets,
    cleanup_finished_replay_sandbox_after_run,
    finalize_replay_row,
    required_event_was_missed,
    resolve_scheduler_policy_override,
    replay_status_is_complete,
    should_inject_event,
    wait_for_auto_replay_checkpoint,
)
from benchmarks.support import average, compute_summary, compute_summary_aliases, is_replay_llm_service_type, total_actions, wait_for


@dataclass(frozen=True)
class FaultOptions:
    injection_rate: float = 0.5
    first_forced_event_chunk: int = 0
    delete_filesystem_checkpoints: bool = False

def benchmark_task_description() -> TaskDescription:
    return TaskDescription("Continuously work on the benchmark task inside the sandbox.")


def default_task_config() -> TaskConfig:
    return TaskConfig()


def parse_fault_options(config: BenchmarkConfig) -> FaultOptions:
    rate = float(config.scenario_options.get("injection_rate", 0.5))
    if rate < 0.0 or rate > 1.0:
        raise ValueError(f"scenario_options.injection_rate must be in [0.0, 1.0], got {rate}")
    raw_first = config.scenario_options.get(
        "first_forced_event_chunk",
        config.scenario_options.get("first_injection_iteration", 0),
    )
    first = int(raw_first)
    if first < 0:
        raise ValueError("scenario_options.first_forced_event_chunk must be >= 0")
    if config.iterations > 0 and first > config.iterations:
        raise ValueError("scenario_options.first_forced_event_chunk must be <= iterations")
    delete_filesystem_checkpoints = bool(config.scenario_options.get("delete_filesystem_checkpoints", False))
    return FaultOptions(
        injection_rate=rate,
        first_forced_event_chunk=first,
        delete_filesystem_checkpoints=delete_filesystem_checkpoints,
    )


def _resolved_replay_iterations(config: BenchmarkConfig, sandbox: SandboxHandle) -> int:
    if config.iterations != 0:
        return config.iterations
    return trace_response_count_for_sandbox(sandbox)


def _validate_replay_fault_options(options: FaultOptions, *, iterations: int) -> None:
    if options.first_forced_event_chunk > iterations:
        raise ValueError("scenario_options.first_forced_event_chunk must be <= iterations")


def build_harness_settings(config: BenchmarkConfig) -> HarnessSettings:
    options = parse_fault_options(config)
    scheduler_config = config.scheduler.apply(
        SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
            prefer_checkpoint_during_llm_request=True,
            require_llm_request_for_checkpoint=False,
        )
    )
    return HarnessSettings(
        scheduler_config=scheduler_config,
        scheduler_policy=resolve_scheduler_policy_override(
            config,
            scenario_default_policy=FaultToleranceCheckpointingPolicy(scheduler_config),
        ),
        checkpoint_manager_factory=lambda base: LatestOnlyCheckpointManager(
            base,
            delete_filesystem_checkpoints=options.delete_filesystem_checkpoints,
        ),
        max_workers=config.effective_max_workers,
    )


def _wait_for_mini_swe_command_window(
    sandbox: SandboxHandle,
    *,
    timeout_s: float = 60.0,
) -> dict[str, object]:
    assert sandbox.task_run is not None
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = sandbox.task_run.poll_status()
        if bool(status.get("command_in_flight", False)):
            return status
        if str(status.get("state", "")) == "finished":
            raise RuntimeError(
                f"mini_swe replay finished before a command-in-flight fault window was reached for sandbox {sandbox.sandbox_id}"
            )
        time.sleep(0.05)
    raise RuntimeError(f"timed out waiting for mini_swe command-in-flight fault window for sandbox {sandbox.sandbox_id}")


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
        row = annotate_row(
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
        emit_row_telemetry(harness, sandbox, row, iteration=iteration)
        rows.append(row)
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
            chunk_index=iteration,
            sandbox_index=sandbox_index,
            rate=options.injection_rate,
            first_forced_event_chunk=options.first_forced_event_chunk,
            rng=rng,
        )
        if not injected:
            row = annotate_row(
                config,
                sandbox,
                iteration=iteration,
                success_ratio=1.0,
                row={
                    "event_type": "fault",
                    "event_injected": 0,
                    "recovery_status": "none",
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
            emit_row_telemetry(harness, sandbox, row, iteration=iteration)
            rows.append(row)
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
        row = annotate_row(
            config,
            sandbox,
            iteration=iteration,
            success_ratio=success_ratio,
            row={
                "event_type": "fault",
                "event_injected": 1,
                "recovery_status": record.status,
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
        emit_row_telemetry(harness, sandbox, row, iteration=iteration)
        rows.append(row)
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
    run_result = _run_replay_manual_sandbox_run(
        config,
        harness,
        sandbox_index=sandbox_index,
        sandbox=sandbox,
        options=options,
    )
    return _finalize_replay_run_result(run_result)


def _run_replay_manual_sandbox_run(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: FaultOptions,
) -> dict[str, object]:
    replay_iterations = _resolved_replay_iterations(config, sandbox)
    _validate_replay_fault_options(options, iterations=replay_iterations)
    replay_chunk_targets = choose_replay_chunk_targets(sandbox, replay_iterations)
    emit_metric = getattr(harness, "emit_benchmark_metric", None)
    rng = random.Random(sandbox_index)
    trace_response_count = trace_response_count_for_sandbox(sandbox)
    checkpoint_ms_values: list[float] = []
    restore_ms_values: list[float] = []
    recovery_ms_values: list[float] = []
    readiness_ms_values: list[float] = []
    end_to_end_recovery_ms_values: list[float] = []
    lost_actions_values: list[float] = []
    chunks_completed = 0
    events_injected = 0
    recoveries_succeeded = 0
    task_error = ""
    required_event_became_unreachable = False

    try:
        if sandbox.task_run is None:
            raise RuntimeError("replay fault benchmark expected sandbox.task_run")
        for chunk_index, chunk_target in enumerate(replay_chunk_targets, start=1):
            chunks_completed += 1
            try:
                current = sandbox.task_run.wait_for_progress(minimum_actions=chunk_target)
            except RuntimeError as exc:
                current = sandbox.task_run.poll_status()
                if (
                    "iflow replay task finished before reaching replay action count" in str(exc)
                    and str(current.get("state", "")) == "finished"
                ):
                    required_event_became_unreachable = required_event_was_missed(
                        chunk_index=chunk_index,
                        events_injected=events_injected,
                        first_forced_event_chunk=options.first_forced_event_chunk,
                    )
                    sandbox.last_status = dict(current)
                    break
                raise
            injected = should_inject_event(
                chunk_index=chunk_index,
                sandbox_index=sandbox_index,
                rate=options.injection_rate,
                first_forced_event_chunk=options.first_forced_event_chunk,
                rng=rng,
            )
            if not injected:
                if replay_status_is_complete(current, trace_response_count=trace_response_count):
                    sandbox.last_status = dict(current)
                    break
                continue
            if str(current.get("state", "")) == "finished":
                required_event_became_unreachable = required_event_was_missed(
                    chunk_index=chunk_index,
                    events_injected=events_injected,
                    first_forced_event_chunk=options.first_forced_event_chunk,
                )
                sandbox.last_status = dict(current)
                break
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
            if sandbox.agent_type == "mini_swe":
                pre_event = _wait_for_mini_swe_command_window(sandbox)
            elif config.iterations == 0:
                pre_event = current
            else:
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
            checkpoint_ms = (checkpoint_finished - checkpoint_started) * 1000.0
            restore_ms = (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0
            recovery_ms = (recovery_finished - recovery_started) * 1000.0
            readiness_ms = (ready_at - recovery_finished) * 1000.0
            end_to_end_recovery_ms = (ready_at - event_started) * 1000.0
            lost_actions = max(0, total_actions(pre_event) - chunk_target)
            restore_ms_values.append(restore_ms)
            recovery_ms_values.append(recovery_ms)
            readiness_ms_values.append(readiness_ms)
            end_to_end_recovery_ms_values.append(end_to_end_recovery_ms)
            lost_actions_values.append(lost_actions)
            if callable(emit_metric):
                emit_metric("benchmark.checkpoint_ms", checkpoint_ms, sandbox, iteration=chunk_index, event_type="fault")
                emit_metric("benchmark.restore_ms", restore_ms, sandbox, iteration=chunk_index, event_type="fault")
                emit_metric("benchmark.recovery_ms", recovery_ms, sandbox, iteration=chunk_index, event_type="fault")
                emit_metric("benchmark.readiness_ms", readiness_ms, sandbox, iteration=chunk_index, event_type="fault")
                emit_metric("benchmark.end_to_end_recovery_ms", end_to_end_recovery_ms, sandbox, iteration=chunk_index, event_type="fault")
                emit_metric("benchmark.lost_actions", float(lost_actions), sandbox, iteration=chunk_index, event_type="fault")
            sandbox.last_status = post_recovery
    except Exception as exc:
        task_error = str(exc)

    if required_event_became_unreachable and not task_error:
        task_error = (
            "replay completed before the required event could be injected "
            f"(first_forced_event_chunk={options.first_forced_event_chunk})"
        )

    return {
        "config": config,
        "harness": harness,
        "sandbox": sandbox,
        "row": {
            "event_type": "fault",
            "chunks_planned": len(replay_chunk_targets),
            "chunks_completed": chunks_completed,
            "iterations_planned": len(replay_chunk_targets),
            "iterations_executed": chunks_completed,
            "events_injected": events_injected,
            "recoveries_succeeded": recoveries_succeeded,
            "checkpoint_ms_avg": average(checkpoint_ms_values),
            "restore_ms_avg": average(restore_ms_values),
            "recovery_ms_avg": average(recovery_ms_values),
            "readiness_ms_avg": average(readiness_ms_values),
            "end_to_end_recovery_ms_avg": average(end_to_end_recovery_ms_values),
            "lost_actions_avg": average(lost_actions_values),
            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            "skipped_no_replay_checkpoint": 1 if not replay_chunk_targets else 0,
        },
        "task_error": task_error,
        "verify_task_accuracy_result": True,
        "success_ratio": 1.0 if not task_error and recoveries_succeeded == events_injected else 0.0,
    }


def run_replay_auto_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: FaultOptions,
) -> dict[str, object]:
    run_result = _run_replay_auto_sandbox_run(
        config,
        harness,
        sandbox_index=sandbox_index,
        sandbox=sandbox,
        options=options,
    )
    return _finalize_replay_run_result(run_result)


def _run_replay_auto_sandbox_run(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: FaultOptions,
) -> dict[str, object]:
    replay_iterations = _resolved_replay_iterations(config, sandbox)
    _validate_replay_fault_options(options, iterations=replay_iterations)
    replay_chunk_targets = choose_replay_chunk_targets(sandbox, replay_iterations)
    emit_metric = getattr(harness, "emit_benchmark_metric", None)
    rng = random.Random(sandbox_index)
    trace_response_count = trace_response_count_for_sandbox(sandbox)
    recovery_ms_values: list[float] = []
    readiness_ms_values: list[float] = []
    end_to_end_recovery_ms_values: list[float] = []
    lost_actions_values: list[float] = []
    chunks_completed = 0
    events_injected = 0
    recoveries_succeeded = 0
    task_error = ""
    required_event_became_unreachable = False

    try:
        if sandbox.task_run is None:
            raise RuntimeError("replay fault benchmark expected sandbox.task_run")
        for chunk_index, chunk_target in enumerate(replay_chunk_targets, start=1):
            chunks_completed += 1
            try:
                current = sandbox.task_run.wait_for_progress(minimum_actions=chunk_target)
            except RuntimeError as exc:
                current = sandbox.task_run.poll_status()
                if (
                    "iflow replay task finished before reaching replay action count" in str(exc)
                    and str(current.get("state", "")) == "finished"
                ):
                    required_event_became_unreachable = required_event_was_missed(
                        chunk_index=chunk_index,
                        events_injected=events_injected,
                        first_forced_event_chunk=options.first_forced_event_chunk,
                    )
                    sandbox.last_status = dict(current)
                    break
                raise
            injected = should_inject_event(
                chunk_index=chunk_index,
                sandbox_index=sandbox_index,
                rate=options.injection_rate,
                first_forced_event_chunk=options.first_forced_event_chunk,
                rng=rng,
            )
            if not injected:
                if replay_status_is_complete(current, trace_response_count=trace_response_count):
                    sandbox.last_status = dict(current)
                    break
                continue
            if str(current.get("state", "")) == "finished":
                required_event_became_unreachable = required_event_was_missed(
                    chunk_index=chunk_index,
                    events_injected=events_injected,
                    first_forced_event_chunk=options.first_forced_event_chunk,
                )
                sandbox.last_status = dict(current)
                break
            # checkpoint_manifest, checkpoint_actions = wait_for_auto_replay_checkpoint(
            #     harness,
            #     sandbox,
            #     minimum_actions=chunk_target,
            #     trace_response_count=trace_response_count,
            # )
            # if checkpoint_manifest is None:
            #     required_event_became_unreachable = required_event_was_missed(
            #         chunk_index=chunk_index,
            #         events_injected=events_injected,
            #         first_forced_event_chunk=options.first_forced_event_chunk,
            #     )
            #     sandbox.last_status = dict(sandbox.task_run.poll_status())
            #     break
            if sandbox.agent_type == "mini_swe":
                pre_event = _wait_for_mini_swe_command_window(sandbox)
            else:
                pre_event = sandbox.task_run.poll_status()
            event_started = time.perf_counter()
            harness.inject_fault(sandbox)
            events_injected += 1
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
            recovery_ms = (recovery_finished - recovery_started) * 1000.0
            readiness_ms = (ready_at - recovery_finished) * 1000.0
            end_to_end_recovery_ms = (ready_at - event_started) * 1000.0
            lost_actions = max(0, total_actions(pre_event) - total_actions(post_recovery))
            recovery_ms_values.append(recovery_ms)
            readiness_ms_values.append(readiness_ms)
            end_to_end_recovery_ms_values.append(end_to_end_recovery_ms)
            lost_actions_values.append(lost_actions)
            if callable(emit_metric):
                emit_metric("benchmark.recovery_ms", recovery_ms, sandbox, iteration=chunk_index, event_type="fault")
                emit_metric("benchmark.readiness_ms", readiness_ms, sandbox, iteration=chunk_index, event_type="fault")
                emit_metric("benchmark.end_to_end_recovery_ms", end_to_end_recovery_ms, sandbox, iteration=chunk_index, event_type="fault")
                emit_metric("benchmark.lost_actions", float(lost_actions), sandbox, iteration=chunk_index, event_type="fault")
            # _ = checkpoint_actions
            sandbox.last_status = post_recovery
    except Exception as exc:
        task_error = str(exc)

    if required_event_became_unreachable and not task_error:
        task_error = (
            "replay completed before the required event could be injected "
            f"(first_forced_event_chunk={options.first_forced_event_chunk})"
        )

    return {
        "config": config,
        "harness": harness,
        "sandbox": sandbox,
        "row": {
            "event_type": "fault",
            "chunks_planned": len(replay_chunk_targets),
            "chunks_completed": chunks_completed,
            "iterations_planned": len(replay_chunk_targets),
            "iterations_executed": chunks_completed,
            "events_injected": events_injected,
            "recoveries_succeeded": recoveries_succeeded,
            "recovery_ms_avg": average(recovery_ms_values),
            "readiness_ms_avg": average(readiness_ms_values),
            "end_to_end_recovery_ms_avg": average(end_to_end_recovery_ms_values),
            "lost_actions_avg": average(lost_actions_values),
            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            "skipped_no_replay_checkpoint": 1 if not replay_chunk_targets else 0,
        },
        "task_error": task_error,
        "verify_task_accuracy_result": True,
        "success_ratio": 1.0 if not task_error and recoveries_succeeded == events_injected else 0.0,
    }


def _finalize_replay_run_result(run_result: dict[str, object]) -> dict[str, object]:
    return finalize_replay_row(
        run_result["config"],
        run_result["harness"],
        run_result["sandbox"],
        row=dict(run_result["row"]),
        task_error=str(run_result["task_error"]),
        verify_task_accuracy_result=bool(run_result["verify_task_accuracy_result"]),
        success_ratio=float(run_result["success_ratio"]),
    )


def _finalize_run_rows(run_result: dict[str, object]) -> list[dict[str, object]]:
    if run_result["kind"] == "replay":
        return [_finalize_replay_run_result(run_result["payload"])]
    return list(run_result["rows"])


def _item_phase_attributes(harness, *, phase: str, prepared, sandbox=None) -> dict[str, object]:
    return benchmark_phase_item_attributes(
        harness,
        phase=phase,
        sandbox_name=prepared.sandbox_name,
        sandbox=prepared.handle if sandbox is None else sandbox,
    )


def _run_prepared_manual_fault_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    prepared,
    options: FaultOptions,
) -> dict[str, object]:
    sandbox = start_prepared_task_record(harness, prepared)
    if is_replay_llm_service_type(sandbox.llm_service_type):
        payload = _run_replay_manual_sandbox_run(
            config,
            harness,
            sandbox_index=sandbox_index,
            sandbox=sandbox,
            options=options,
        )
        payload["task_error"] = cleanup_finished_replay_sandbox_after_run(
            config,
            harness,
            sandbox,
            task_error=str(payload["task_error"]),
        )
        return {
            "kind": "replay",
            "payload": payload,
        }
    return {
        "kind": "rows",
        "rows": run_manual_sandbox(config, harness, sandbox=sandbox),
    }


def _run_prepared_auto_fault_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    prepared,
    options: FaultOptions,
) -> dict[str, object]:
    sandbox = start_prepared_task_record(harness, prepared)
    if is_replay_llm_service_type(sandbox.llm_service_type):
        payload = _run_replay_auto_sandbox_run(
            config,
            harness,
            sandbox_index=sandbox_index,
            sandbox=sandbox,
            options=options,
        )
        payload["task_error"] = cleanup_finished_replay_sandbox_after_run(
            config,
            harness,
            sandbox,
            task_error=str(payload["task_error"]),
        )
        return {
            "kind": "replay",
            "payload": payload,
        }
    return {
        "kind": "rows",
        "rows": run_auto_sandbox(config, harness, sandbox_index=sandbox_index, sandbox=sandbox, options=options),
    }


def run_manual(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    options = parse_fault_options(config)
    specs = make_benchmark_sandbox_specs(
        sandbox_name_prefix="fault",
        records=records,
    )
    if config.phase_merging.setup_and_run:
        indexed_specs = list(enumerate(specs))
        run_results = benchmark_setup_run_pipeline(
            indexed_specs,
            setup_fn=lambda item: harness.setup_task_record(item[1].sandbox_name, item[1].task_record),
            run_fn=lambda item, prepared: _run_prepared_manual_fault_sandbox(
                config,
                harness,
                sandbox_index=item[0],
                prepared=prepared,
                options=options,
            ),
            setup_max_workers=config.effective_phase_workers.setup,
            run_max_workers=config.effective_phase_workers.run,
            harness=harness,
            setup_item_attributes=lambda item: benchmark_phase_item_attributes(
                harness,
                phase="setup",
                sandbox_name=item[1].sandbox_name,
                task_record=item[1].task_record,
            ),
            run_item_attributes=lambda _item, prepared: _item_phase_attributes(
                harness,
                phase="run",
                prepared=prepared,
            ),
            executor_pool=config.phase_merging.setup_and_run_executor_pool,
        )
    else:
        prepared = setup_task_records_phase(
            harness,
            specs=specs,
            max_workers=config.effective_phase_workers.setup,
        )
        indexed_prepared = list(enumerate(prepared))
        run_results = benchmark_phase_map(
            indexed_prepared,
            lambda item: _run_prepared_manual_fault_sandbox(
                config,
                harness,
                sandbox_index=item[0],
                prepared=item[1],
                options=options,
            ),
            phase="run",
            max_workers=config.effective_phase_workers.run,
            harness=harness,
            item_attributes=lambda item: _item_phase_attributes(
                harness,
                phase="run",
                prepared=item[1],
            ),
        )
    if config.verification_enabled:
        row_groups = benchmark_phase_map(
            run_results,
            _finalize_run_rows,
            phase="verification",
            max_workers=config.effective_phase_workers.verification,
            harness=harness,
            item_attributes=lambda item: benchmark_phase_item_attributes(
                harness,
                phase="verification",
                sandbox_name=str(item["payload"]["sandbox"].sandbox_id)
                if item["kind"] == "replay"
                else str(item["rows"][0]["sandbox_id"]) if item["rows"] else "",
                sandbox=item["payload"]["sandbox"] if item["kind"] == "replay" else None,
            ),
        )
    else:
        emit_benchmark_phase_skipped(
            phase="verification",
            sandbox_count=len(run_results),
            configured_max_workers=config.effective_phase_workers.verification,
        )
        row_groups = [_finalize_run_rows(item) for item in run_results]
    rows = [row for group in row_groups for row in group]
    return sorted(rows, key=lambda row: (int(row["iteration"]), str(row["sandbox_id"])))


def run_auto(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    options = parse_fault_options(config)
    specs = make_benchmark_sandbox_specs(
        sandbox_name_prefix="fault",
        records=records,
    )
    if config.phase_merging.setup_and_run:
        indexed_specs = list(enumerate(specs))
        run_results = benchmark_setup_run_pipeline(
            indexed_specs,
            setup_fn=lambda item: harness.setup_task_record(item[1].sandbox_name, item[1].task_record),
            run_fn=lambda item, prepared: _run_prepared_auto_fault_sandbox(
                config,
                harness,
                sandbox_index=item[0],
                prepared=prepared,
                options=options,
            ),
            setup_max_workers=config.effective_phase_workers.setup,
            run_max_workers=config.effective_phase_workers.run,
            harness=harness,
            setup_item_attributes=lambda item: benchmark_phase_item_attributes(
                harness,
                phase="setup",
                sandbox_name=item[1].sandbox_name,
                task_record=item[1].task_record,
            ),
            run_item_attributes=lambda _item, prepared: _item_phase_attributes(
                harness,
                phase="run",
                prepared=prepared,
            ),
            executor_pool=config.phase_merging.setup_and_run_executor_pool,
        )
    else:
        prepared = setup_task_records_phase(
            harness,
            specs=specs,
            max_workers=config.effective_phase_workers.setup,
        )
        indexed_prepared = list(enumerate(prepared))
        run_results = benchmark_phase_map(
            indexed_prepared,
            lambda item: _run_prepared_auto_fault_sandbox(
                config,
                harness,
                sandbox_index=item[0],
                prepared=item[1],
                options=options,
            ),
            phase="run",
            max_workers=config.effective_phase_workers.run,
            harness=harness,
            item_attributes=lambda item: _item_phase_attributes(
                harness,
                phase="run",
                prepared=item[1],
            ),
        )
    if config.verification_enabled:
        row_groups = benchmark_phase_map(
            run_results,
            _finalize_run_rows,
            phase="verification",
            max_workers=config.effective_phase_workers.verification,
            harness=harness,
            item_attributes=lambda item: benchmark_phase_item_attributes(
                harness,
                phase="verification",
                sandbox_name=str(item["payload"]["sandbox"].sandbox_id)
                if item["kind"] == "replay"
                else str(item["rows"][0]["sandbox_id"]) if item["rows"] else "",
                sandbox=item["payload"]["sandbox"] if item["kind"] == "replay" else None,
            ),
        )
    else:
        emit_benchmark_phase_skipped(
            phase="verification",
            sandbox_count=len(run_results),
            configured_max_workers=config.effective_phase_workers.verification,
        )
        row_groups = [_finalize_run_rows(item) for item in run_results]
    rows = [row for group in row_groups for row in group]
    return sorted(rows, key=lambda row: (int(row["iteration"]), str(row["sandbox_id"])))


def summarize(config: BenchmarkConfig, rows: list[dict[str, object]]) -> dict[str, float]:
    if rows and ("verification_status" in rows[0] or "chunks_planned" in rows[0] or "iterations_planned" in rows[0]):
        return compute_summary_aliases(
            rows,
            {
                "recovery_ms": "recovery_ms_avg",
                "readiness_ms": "readiness_ms_avg",
                "end_to_end_recovery_ms": "end_to_end_recovery_ms_avg",
                "lost_actions_avg": "lost_actions_avg",
                "success_ratio": "success_ratio",
            },
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

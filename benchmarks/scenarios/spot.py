from __future__ import annotations

from dataclasses import dataclass
import random
import time

from crab import DeleteAfterRestoreCheckpointManager, SchedulerConfig, SpotPreemptionCheckpointingPolicy
from crab.models import utc_now

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
)
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from benchmarks.scenarios.common import (
    choose_replay_chunk_targets,
    cleanup_finished_replay_sandbox_after_run,
    finalize_replay_row,
    resolve_scheduler_policy_override,
    should_inject_event,
    wait_for_iteration_progress,
)
from benchmarks.support import average, compute_summary, compute_summary_aliases, is_replay_llm_service_type


@dataclass(frozen=True)
class SpotOptions:
    injection_rate: float = 0.5
    first_forced_event_chunk: int = 0
    grace_period_seconds: float = 60.0


def benchmark_task_description() -> TaskDescription:
    return TaskDescription("Continuously work on the benchmark task inside the sandbox.")


def default_task_config() -> TaskConfig:
    return TaskConfig()


def parse_spot_options(config: BenchmarkConfig) -> SpotOptions:
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
    if first > config.iterations:
        raise ValueError("scenario_options.first_forced_event_chunk must be <= iterations")
    grace = float(config.scenario_options.get("grace_period_seconds", 60.0))
    if grace <= 0.0:
        raise ValueError("scenario_options.grace_period_seconds must be > 0")
    return SpotOptions(
        injection_rate=rate,
        first_forced_event_chunk=first,
        grace_period_seconds=grace,
    )


def build_harness_settings(config: BenchmarkConfig) -> HarnessSettings:
    scheduler_config = config.scheduler.apply(
        SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=False,
        )
    )
    return HarnessSettings(
        scheduler_config=scheduler_config,
        scheduler_policy=resolve_scheduler_policy_override(
            config,
            scenario_default_policy=SpotPreemptionCheckpointingPolicy(scheduler_config),
        ),
        checkpoint_manager_factory=lambda base: DeleteAfterRestoreCheckpointManager(base),
        max_workers=config.effective_max_workers,
    )


def run_manual_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox: SandboxHandle,
    options: SpotOptions,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for iteration in range(1, config.iterations + 1):
        wait_for_iteration_progress(sandbox, iteration=iteration)
        event_started = time.perf_counter()
        harness.set_snapshot_metadata(
            sandbox,
            preemption_notice=True,
            preemption_grace_remaining_seconds=options.grace_period_seconds,
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
        row = annotate_row(
            config,
            sandbox,
            iteration=iteration,
            success_ratio=1.0 if restore_result.status.value == "succeeded" else 0.0,
            row={
                "event_type": "preemption",
                "event_injected": 1,
                "recovery_status": restore_result.status.value,
                "grace_period_ms": options.grace_period_seconds * 1000.0,
                "checkpoint_ms": (checkpoint_finished - recovery_started) * 1000.0,
                "restore_ms": (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
                "recovery_ms": (recovery_finished - recovery_started) * 1000.0,
                "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                "migration_ms": (ready_at - event_started) * 1000.0,
                "budget_slack_ms": options.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0,
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
    options: SpotOptions,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    rng = random.Random(sandbox_index)
    for iteration in range(1, config.iterations + 1):
        wait_for_iteration_progress(sandbox, iteration=iteration)
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
                    "event_type": "preemption",
                    "event_injected": 0,
                    "recovery_status": "none",
                    "grace_period_ms": options.grace_period_seconds * 1000.0,
                    "recovery_ms": 0.0,
                    "readiness_ms": 0.0,
                    "end_to_end_recovery_ms": 0.0,
                    "migration_ms": 0.0,
                    "budget_slack_ms": options.grace_period_seconds * 1000.0,
                    "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                },
            )
            emit_row_telemetry(harness, sandbox, row, iteration=iteration)
            rows.append(row)
            continue
        event_started = time.perf_counter()
        observed_after = utc_now()
        harness.notify_preemption(
            sandbox,
            grace_remaining_seconds=options.grace_period_seconds,
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
        row = annotate_row(
            config,
            sandbox,
            iteration=iteration,
            success_ratio=1.0 if record.status in {"restored", "relaunched"} else 0.0,
            row={
                "event_type": "preemption",
                "event_injected": 1,
                "recovery_status": record.status,
                "grace_period_ms": options.grace_period_seconds * 1000.0,
                "recovery_ms": (recovery_finished - migration_started) * 1000.0,
                "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                "migration_ms": (ready_at - event_started) * 1000.0,
                "budget_slack_ms": options.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0,
                "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            },
        )
        emit_row_telemetry(harness, sandbox, row, iteration=iteration)
        rows.append(row)
        if record.status not in {"restored", "relaunched"}:
            break
    return rows


def run_replay_manual_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: SpotOptions,
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
    options: SpotOptions,
) -> dict[str, object]:
    replay_chunk_targets = choose_replay_chunk_targets(sandbox, config.iterations)
    emit_metric = getattr(harness, "emit_benchmark_metric", None)
    rng = random.Random(sandbox_index)
    checkpoint_ms_values: list[float] = []
    restore_ms_values: list[float] = []
    recovery_ms_values: list[float] = []
    readiness_ms_values: list[float] = []
    end_to_end_recovery_ms_values: list[float] = []
    migration_ms_values: list[float] = []
    budget_slack_ms_values: list[float] = []
    chunks_completed = 0
    events_injected = 0
    recoveries_succeeded = 0
    task_error = ""

    try:
        if sandbox.task_run is None:
            raise RuntimeError("replay spot benchmark expected sandbox.task_run")
        for chunk_index, chunk_target in enumerate(replay_chunk_targets, start=1):
            chunks_completed += 1
            sandbox.task_run.wait_for_progress(minimum_actions=chunk_target)
            injected = should_inject_event(
                chunk_index=chunk_index,
                sandbox_index=sandbox_index,
                rate=options.injection_rate,
                first_forced_event_chunk=options.first_forced_event_chunk,
                rng=rng,
            )
            if not injected:
                continue
            events_injected += 1
            event_started = time.perf_counter()
            harness.set_snapshot_metadata(
                sandbox,
                preemption_notice=True,
                preemption_grace_remaining_seconds=options.grace_period_seconds,
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
            restore_ms = (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0
            recovery_ms = (recovery_finished - recovery_started) * 1000.0
            readiness_ms = (ready_at - recovery_finished) * 1000.0
            end_to_end_recovery_ms = (ready_at - event_started) * 1000.0
            migration_ms = (ready_at - event_started) * 1000.0
            budget_slack_ms = options.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0
            restore_ms_values.append(restore_ms)
            recovery_ms_values.append(recovery_ms)
            readiness_ms_values.append(readiness_ms)
            end_to_end_recovery_ms_values.append(end_to_end_recovery_ms)
            migration_ms_values.append(migration_ms)
            budget_slack_ms_values.append(budget_slack_ms)
            if callable(emit_metric):
                emit_metric("benchmark.checkpoint_ms", checkpoint_ms_values[-1], sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.restore_ms", restore_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.recovery_ms", recovery_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.readiness_ms", readiness_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.end_to_end_recovery_ms", end_to_end_recovery_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.migration_ms", migration_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.budget_slack_ms", budget_slack_ms, sandbox, iteration=chunk_index, event_type="preemption")
            recoveries_succeeded += 1
    except Exception as exc:
        task_error = str(exc)

    return {
        "config": config,
        "harness": harness,
        "sandbox": sandbox,
        "row": {
            "event_type": "preemption",
            "chunks_planned": len(replay_chunk_targets),
            "chunks_completed": chunks_completed,
            "iterations_planned": len(replay_chunk_targets),
            "iterations_executed": chunks_completed,
            "events_injected": events_injected,
            "recoveries_succeeded": recoveries_succeeded,
            "grace_period_ms": options.grace_period_seconds * 1000.0,
            "checkpoint_ms_avg": average(checkpoint_ms_values),
            "restore_ms_avg": average(restore_ms_values),
            "recovery_ms_avg": average(recovery_ms_values),
            "readiness_ms_avg": average(readiness_ms_values),
            "end_to_end_recovery_ms_avg": average(end_to_end_recovery_ms_values),
            "migration_ms_avg": average(migration_ms_values),
            "budget_slack_ms_avg": average(budget_slack_ms_values),
            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            "skipped_no_replay_checkpoint": 1 if not replay_chunk_targets else 0,
        },
        "task_error": task_error,
        "verify_task_accuracy_result": False,
        "success_ratio": 1.0 if not task_error and recoveries_succeeded == events_injected else 0.0,
    }


def run_replay_auto_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: SpotOptions,
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
    options: SpotOptions,
) -> dict[str, object]:
    replay_chunk_targets = choose_replay_chunk_targets(sandbox, config.iterations)
    emit_metric = getattr(harness, "emit_benchmark_metric", None)
    rng = random.Random(sandbox_index)
    recovery_ms_values: list[float] = []
    readiness_ms_values: list[float] = []
    end_to_end_recovery_ms_values: list[float] = []
    migration_ms_values: list[float] = []
    budget_slack_ms_values: list[float] = []
    chunks_completed = 0
    events_injected = 0
    recoveries_succeeded = 0
    task_error = ""

    try:
        if sandbox.task_run is None:
            raise RuntimeError("replay spot benchmark expected sandbox.task_run")
        for chunk_index, chunk_target in enumerate(replay_chunk_targets, start=1):
            chunks_completed += 1
            sandbox.task_run.wait_for_progress(minimum_actions=chunk_target)
            injected = should_inject_event(
                chunk_index=chunk_index,
                sandbox_index=sandbox_index,
                rate=options.injection_rate,
                first_forced_event_chunk=options.first_forced_event_chunk,
                rng=rng,
            )
            if not injected:
                continue
            events_injected += 1
            event_started = time.perf_counter()
            observed_after = utc_now()
            harness.notify_preemption(
                sandbox,
                grace_remaining_seconds=options.grace_period_seconds,
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
            recovery_ms = (recovery_finished - migration_started) * 1000.0
            readiness_ms = (ready_at - recovery_finished) * 1000.0
            end_to_end_recovery_ms = (ready_at - event_started) * 1000.0
            migration_ms = (ready_at - event_started) * 1000.0
            budget_slack_ms = options.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0
            recovery_ms_values.append(recovery_ms)
            readiness_ms_values.append(readiness_ms)
            end_to_end_recovery_ms_values.append(end_to_end_recovery_ms)
            migration_ms_values.append(migration_ms)
            budget_slack_ms_values.append(budget_slack_ms)
            if callable(emit_metric):
                emit_metric("benchmark.recovery_ms", recovery_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.readiness_ms", readiness_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.end_to_end_recovery_ms", end_to_end_recovery_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.migration_ms", migration_ms, sandbox, iteration=chunk_index, event_type="preemption")
                emit_metric("benchmark.budget_slack_ms", budget_slack_ms, sandbox, iteration=chunk_index, event_type="preemption")
            if record.status not in {"restored", "relaunched"}:
                raise RuntimeError(f"preemption recovery failed with status={record.status}")
    except Exception as exc:
        task_error = str(exc)

    return {
        "config": config,
        "harness": harness,
        "sandbox": sandbox,
        "row": {
            "event_type": "preemption",
            "chunks_planned": len(replay_chunk_targets),
            "chunks_completed": chunks_completed,
            "iterations_planned": len(replay_chunk_targets),
            "iterations_executed": chunks_completed,
            "events_injected": events_injected,
            "recoveries_succeeded": recoveries_succeeded,
            "grace_period_ms": options.grace_period_seconds * 1000.0,
            "recovery_ms_avg": average(recovery_ms_values),
            "readiness_ms_avg": average(readiness_ms_values),
            "end_to_end_recovery_ms_avg": average(end_to_end_recovery_ms_values),
            "migration_ms_avg": average(migration_ms_values),
            "budget_slack_ms_avg": average(budget_slack_ms_values),
            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            "skipped_no_replay_checkpoint": 1 if not replay_chunk_targets else 0,
        },
        "task_error": task_error,
        "verify_task_accuracy_result": False,
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


def _item_phase_attributes(harness, *, phase: str, prepared) -> dict[str, object]:
    return benchmark_phase_item_attributes(
        harness,
        phase=phase,
        sandbox_name=prepared.sandbox_name,
        sandbox=prepared.handle,
    )


def _run_prepared_manual_spot_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    prepared,
    options: SpotOptions,
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
        "rows": run_manual_sandbox(config, harness, sandbox=sandbox, options=options),
    }


def _run_prepared_auto_spot_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    prepared,
    options: SpotOptions,
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
    options = parse_spot_options(config)
    specs = make_benchmark_sandbox_specs(
        sandbox_name_prefix="spot",
        records=records,
    )
    if config.phase_merging.setup_and_run:
        indexed_specs = list(enumerate(specs))
        run_results = benchmark_setup_run_pipeline(
            indexed_specs,
            setup_fn=lambda item: harness.setup_task_record(item[1].sandbox_name, item[1].task_record),
            run_fn=lambda item, prepared: _run_prepared_manual_spot_sandbox(
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
            lambda item: _run_prepared_manual_spot_sandbox(
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
    options = parse_spot_options(config)
    specs = make_benchmark_sandbox_specs(
        sandbox_name_prefix="spot",
        records=records,
    )
    if config.phase_merging.setup_and_run:
        indexed_specs = list(enumerate(specs))
        run_results = benchmark_setup_run_pipeline(
            indexed_specs,
            setup_fn=lambda item: harness.setup_task_record(item[1].sandbox_name, item[1].task_record),
            run_fn=lambda item, prepared: _run_prepared_auto_spot_sandbox(
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
            lambda item: _run_prepared_auto_spot_sandbox(
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
                "migration_ms": "migration_ms_avg",
                "budget_slack_ms": "budget_slack_ms_avg",
                "success_ratio": "success_ratio",
            },
        )
    if config.mode == "auto":
        event_rows = [row for row in rows if int(row["event_injected"]) == 1]
        return compute_summary(
            event_rows,
            ["recovery_ms", "readiness_ms", "end_to_end_recovery_ms", "migration_ms", "budget_slack_ms"],
        ) if event_rows else {}
    return compute_summary(
        rows,
        ["checkpoint_ms", "restore_ms", "recovery_ms", "readiness_ms", "end_to_end_recovery_ms", "migration_ms", "budget_slack_ms"],
    )


SCENARIO = ScenarioDefinition(
    name="spot",
    supported_modes=frozenset({"manual", "auto"}),
    build_harness_settings=build_harness_settings,
    run_manual=run_manual,
    run_auto=run_auto,
    summarize=summarize,
)

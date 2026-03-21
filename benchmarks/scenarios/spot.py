from __future__ import annotations

from dataclasses import dataclass
import random
import time

from agent_cr import DeleteAfterRestoreCheckpointManager, SchedulerConfig, SpotPreemptionCheckpointingPolicy
from agent_cr.models import utc_now

from integrations.agents import SandboxHandle, TaskConfig, TaskDescription

from benchmarks.config import BenchmarkConfig
from benchmarks.core import annotate_row, parallel_map, resolve_task_records
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from benchmarks.scenarios.common import (
    choose_replay_chunk_targets,
    finalize_replay_row,
    should_inject_event,
    wait_for_iteration_progress,
)
from benchmarks.support import average, compute_summary, is_replay_llm_service_type


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
    scheduler_config = SchedulerConfig(
        min_checkpoint_interval_seconds=0.0,
        force_checkpoint_after_seconds=0.0,
        require_change_signal=False,
    )
    return HarnessSettings(
        scheduler_config=scheduler_config,
        scheduler_policy=SpotPreemptionCheckpointingPolicy(scheduler_config),
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
        rows.append(
            annotate_row(
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
        )
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
            rows.append(
                annotate_row(
                    config,
                    sandbox,
                    iteration=iteration,
                    success_ratio=1.0,
                    row={
                        "event_type": "preemption",
                        "event_injected": 0,
                        "recovery_status": "none",
                        "grace_period_ms": options.grace_period_seconds * 1000.0,
                        "checkpoint_ms": 0.0,
                        "restore_ms": 0.0,
                        "recovery_ms": 0.0,
                        "readiness_ms": 0.0,
                        "end_to_end_recovery_ms": 0.0,
                        "migration_ms": 0.0,
                        "budget_slack_ms": options.grace_period_seconds * 1000.0,
                        "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                    },
                )
            )
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
        rows.append(
            annotate_row(
                config,
                sandbox,
                iteration=iteration,
                success_ratio=1.0 if record.status in {"restored", "relaunched"} else 0.0,
                row={
                    "event_type": "preemption",
                    "event_injected": 1,
                    "recovery_status": record.status,
                    "grace_period_ms": options.grace_period_seconds * 1000.0,
                    "checkpoint_ms": 0.0,
                    "restore_ms": 0.0,
                    "recovery_ms": (recovery_finished - migration_started) * 1000.0,
                    "readiness_ms": (ready_at - recovery_finished) * 1000.0,
                    "end_to_end_recovery_ms": (ready_at - event_started) * 1000.0,
                    "migration_ms": (ready_at - event_started) * 1000.0,
                    "budget_slack_ms": options.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0,
                    "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
                },
            )
        )
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
    replay_chunk_targets = choose_replay_chunk_targets(sandbox, config.iterations)
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
            restore_ms_values.append(
                (restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0
            )
            recovery_ms_values.append((recovery_finished - recovery_started) * 1000.0)
            readiness_ms_values.append((ready_at - recovery_finished) * 1000.0)
            end_to_end_recovery_ms_values.append((ready_at - event_started) * 1000.0)
            migration_ms_values.append((ready_at - event_started) * 1000.0)
            budget_slack_ms_values.append(
                options.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0
            )
            recoveries_succeeded += 1
    except Exception as exc:
        task_error = str(exc)

    return finalize_replay_row(
        config,
        harness,
        sandbox,
        row={
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
        task_error=task_error,
        verify_task_accuracy_result=False,
        success_ratio=1.0 if not task_error and recoveries_succeeded == events_injected else 0.0,
    )


def run_replay_auto_sandbox(
    config: BenchmarkConfig,
    harness,
    *,
    sandbox_index: int,
    sandbox: SandboxHandle,
    options: SpotOptions,
) -> dict[str, object]:
    replay_chunk_targets = choose_replay_chunk_targets(sandbox, config.iterations)
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
            recovery_ms_values.append((recovery_finished - migration_started) * 1000.0)
            readiness_ms_values.append((ready_at - recovery_finished) * 1000.0)
            end_to_end_recovery_ms_values.append((ready_at - event_started) * 1000.0)
            migration_ms_values.append((ready_at - event_started) * 1000.0)
            budget_slack_ms_values.append(
                options.grace_period_seconds * 1000.0 - (ready_at - event_started) * 1000.0
            )
            if record.status not in {"restored", "relaunched"}:
                raise RuntimeError(f"preemption recovery failed with status={record.status}")
    except Exception as exc:
        task_error = str(exc)

    return finalize_replay_row(
        config,
        harness,
        sandbox,
        row={
            "event_type": "preemption",
            "chunks_planned": len(replay_chunk_targets),
            "chunks_completed": chunks_completed,
            "iterations_planned": len(replay_chunk_targets),
            "iterations_executed": chunks_completed,
            "events_injected": events_injected,
            "recoveries_succeeded": recoveries_succeeded,
            "grace_period_ms": options.grace_period_seconds * 1000.0,
            "checkpoint_ms_avg": average([]),
            "restore_ms_avg": average([]),
            "recovery_ms_avg": average(recovery_ms_values),
            "readiness_ms_avg": average(readiness_ms_values),
            "end_to_end_recovery_ms_avg": average(end_to_end_recovery_ms_values),
            "migration_ms_avg": average(migration_ms_values),
            "budget_slack_ms_avg": average(budget_slack_ms_values),
            "retained_checkpoints": len(harness.storage.list_checkpoints(sandbox.sandbox_id)),
            "skipped_no_replay_checkpoint": 1 if not replay_chunk_targets else 0,
        },
        task_error=task_error,
        verify_task_accuracy_result=False,
        success_ratio=1.0 if not task_error and recoveries_succeeded == events_injected else 0.0,
    )


def run_manual(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    options = parse_spot_options(config)
    indexed_records = list(enumerate(records))
    rows: list[dict[str, object]] = []
    for sandbox_index, task_record in indexed_records:
        sandbox = harness.launch_task_record(f"spot-{sandbox_index}", task_record)
        if is_replay_llm_service_type(sandbox.llm_service_type):
            rows.append(
                run_replay_manual_sandbox(
                    config,
                    harness,
                    sandbox_index=sandbox_index,
                    sandbox=sandbox,
                    options=options,
                )
            )
        else:
            rows.extend(run_manual_sandbox(config, harness, sandbox=sandbox, options=options))
    return sorted(rows, key=lambda row: (int(row["iteration"]), str(row["sandbox_id"])))


def run_auto(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    options = parse_spot_options(config)
    indexed_records = list(enumerate(records))
    def launch_and_run(item: tuple[int, object]) -> list[dict[str, object]]:
        sandbox_index, task_record = item
        sandbox = harness.launch_task_record(f"spot-{sandbox_index}", task_record)
        if is_replay_llm_service_type(sandbox.llm_service_type):
            return [
                run_replay_auto_sandbox(
                    config,
                    harness,
                    sandbox_index=sandbox_index,
                    sandbox=sandbox,
                    options=options,
                )
            ]
        return run_auto_sandbox(config, harness, sandbox_index=sandbox_index, sandbox=sandbox, options=options)

    rows = [
        row
        for group in parallel_map(indexed_records, launch_and_run, max_workers=config.effective_max_workers)
        for row in group
    ]
    return sorted(rows, key=lambda row: (int(row["iteration"]), str(row["sandbox_id"])))


def summarize(config: BenchmarkConfig, rows: list[dict[str, object]]) -> dict[str, float]:
    if rows and ("verification_status" in rows[0] or "chunks_planned" in rows[0] or "iterations_planned" in rows[0]):
        return compute_summary(
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

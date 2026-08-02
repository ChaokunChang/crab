from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import random
import time

from crab import (
    KeepAllCheckpointManager,
    RequestContext,
    RequestInterceptorHook,
    SchedulerConfig,
    TreeSearchCheckpointingPolicy,
)
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
    parallel_map,
    resolve_task_records,
    load_task_dataset,
    setup_task_records_phase,
    start_prepared_task_record,
    task_completion_timeout_seconds,
)
from benchmarks.support import BenchmarkTaskRecord, effective_trace_replay_progress_count, is_replay_llm_service_type
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from benchmarks.support import (
    TreeSearchCheckpointRecord,
    build_tree_search_checkpoint_index,
    compute_summary,
    resolve_tree_search_replay_checkpoint,
)
from benchmarks.scenarios.common import resolve_scheduler_policy_override


_TASK_STOP_REQUEST_GRACE_SECONDS = 1.0
_TASK_FORCE_STOP_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class TreeOptions:
    source_steps: int = 6
    branch_points: int = 2
    fork_steps: int = 3
    replay_mode: str = "sequential"
    skip_if_no_meaningful_delta: bool = False
    checkpoint_forks: bool = False


def benchmark_task_description() -> TaskDescription:
    return TaskDescription("Continuously explore the search tree and make forward progress.")


def default_task_config() -> TaskConfig:
    return TaskConfig()


def parse_tree_options(config: BenchmarkConfig) -> TreeOptions:
    replay_mode = str(config.scenario_options.get("replay_mode", "sequential"))
    if replay_mode not in {"sequential", "concurrent"}:
        raise ValueError("scenario_options.replay_mode must be 'sequential' or 'concurrent'")
    source_steps = int(config.scenario_options.get("source_steps", 6))
    branch_points = int(config.scenario_options.get("branch_points", 2))
    fork_steps = int(config.scenario_options.get("fork_steps", 3))
    skip_if_no_meaningful_delta = bool(config.scenario_options.get("skip_if_no_meaningful_delta", False))
    checkpoint_forks = bool(config.scenario_options.get("checkpoint_forks", False))
    if source_steps <= 0:
        raise ValueError("scenario_options.source_steps must be > 0")
    if branch_points < 0:
        raise ValueError("scenario_options.branch_points must be >= 0")
    if fork_steps < 0:
        raise ValueError("scenario_options.fork_steps must be >= 0")
    return TreeOptions(
        source_steps=source_steps,
        branch_points=branch_points,
        fork_steps=fork_steps,
        replay_mode=replay_mode,
        skip_if_no_meaningful_delta=skip_if_no_meaningful_delta,
        checkpoint_forks=checkpoint_forks,
    )


def build_harness_settings(config: BenchmarkConfig) -> HarnessSettings:
    options = parse_tree_options(config)
    scheduler_config = config.scheduler.apply(
        SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=options.skip_if_no_meaningful_delta,
        )
    )
    return HarnessSettings(
        scheduler_config=scheduler_config,
        scheduler_policy=resolve_scheduler_policy_override(
            config,
            scenario_default_policy=TreeSearchCheckpointingPolicy(
                scheduler_config,
                skip_if_no_meaningful_delta=options.skip_if_no_meaningful_delta,
                checkpoint_forks=options.checkpoint_forks,
            ),
        ),
        checkpoint_manager_factory=lambda base: KeepAllCheckpointManager(base),
        max_workers=max(1, config.effective_max_workers * max(1, options.branch_points + 1)),
    )


def choose_replay_steps(source_steps: int, branch_points: int) -> list[int]:
    if source_steps <= 1 or branch_points <= 0:
        return []
    candidates = list(range(1, source_steps))
    if branch_points >= len(candidates):
        return candidates
    stride = max(1, len(candidates) // branch_points)
    return candidates[::stride][:branch_points]


def choose_random_replay_steps(total_steps: int, branch_points: int, *, seed_key: str) -> list[int]:
    if total_steps <= 1 or branch_points <= 0:
        return []
    candidates = list(range(1, total_steps))
    if branch_points >= len(candidates):
        return candidates
    seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, branch_points))


def uses_trace_driven_tree_replay(config: BenchmarkConfig, task_record: BenchmarkTaskRecord) -> bool:
    return is_replay_llm_service_type(task_record.llm_service_type or config.llm_service)


def replay_total_steps_for_task(
    config: BenchmarkConfig,
    task_record: BenchmarkTaskRecord,
) -> int | None:
    if not uses_trace_driven_tree_replay(config, task_record):
        return None
    return effective_trace_replay_progress_count(task_record)


def effective_source_steps_for_task(
    config: BenchmarkConfig,
    options: TreeOptions,
    task_record: BenchmarkTaskRecord,
) -> int:
    replay_total_steps = replay_total_steps_for_task(config, task_record)
    if replay_total_steps is not None:
        return replay_total_steps
    return options.source_steps


def choose_replay_steps_for_task(
    config: BenchmarkConfig,
    options: TreeOptions,
    task_record: BenchmarkTaskRecord,
    *,
    sandbox_name: str,
) -> list[int]:
    replay_total_steps = replay_total_steps_for_task(config, task_record)
    if replay_total_steps is None:
        return choose_replay_steps(options.source_steps, options.branch_points)
    task_key = task_record.task_id or sandbox_name
    return choose_random_replay_steps(
        replay_total_steps,
        options.branch_points,
        seed_key=f"{task_key}:{sandbox_name}:{replay_total_steps}:{options.branch_points}",
    )


class TreeSearchStepHook(RequestInterceptorHook):
    def __init__(self, harness) -> None:
        self._harness = harness
        self._steps: dict[object, int] = defaultdict(int)

    def on_request_start(self, context: RequestContext) -> None:
        step = self._steps.get(context.sandbox_id, 0) + 1
        self._steps[context.sandbox_id] = step
        self._harness.set_snapshot_metadata_by_id(context.sandbox_id, tree_search_step=step)

    def on_request_end(self, context: RequestContext) -> None:
        _ = context


@dataclass(frozen=True)
class PreparedReplayFork:
    source_index: int
    replay_step: int
    checkpoint_step: int
    checkpoint: TreeSearchCheckpointRecord
    fork: SandboxHandle


@dataclass(frozen=True)
class RestoredReplayFork:
    prepared: PreparedReplayFork
    restore_started_at: float
    recovery_finished_at: float
    ready_at: float
    restore_ms: float
    recovery_ms: float
    readiness_ms: float
    end_to_end_recovery_ms: float


def task_is_running(sandbox: SandboxHandle) -> bool:
    return sandbox.task_future is not None and not sandbox.task_future.done()


def wait_for_task_future(
    sandbox: SandboxHandle,
    *,
    timeout_s: float,
) -> bool:
    task_future = sandbox.task_future
    if task_future is None:
        return True
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if task_future.done():
            task_future.result()
            return True
        time.sleep(0.05)
    if task_future.done():
        task_future.result()
        return True
    return False


def wait_for_task_completion_or_stop(
    harness,
    sandbox: SandboxHandle,
    *,
    action_budget: int,
) -> None:
    if sandbox.task_run is None or sandbox.task_future is None:
        return
    waited_actions = 0
    while waited_actions < max(0, action_budget):
        if sandbox.task_future.done():
            sandbox.task_future.result()
            return
        try:
            sandbox.task_run.wait_for_action_delta(delta=1)
        except RuntimeError:
            if sandbox.task_future.done():
                sandbox.task_future.result()
                return
            raise
        waited_actions += 1
    if sandbox.task_future.done():
        sandbox.task_future.result()
        return
    sandbox.task_run.request_stop()
    if wait_for_task_future(sandbox, timeout_s=_TASK_STOP_REQUEST_GRACE_SECONDS):
        return
    harness.deactivate_sandbox_runtime(sandbox)
    if not wait_for_task_future(sandbox, timeout_s=_TASK_FORCE_STOP_GRACE_SECONDS):
        raise RuntimeError(f"tree-search task did not stop for sandbox {sandbox.sandbox_id}")


def collect_manual_checkpoint_indexes(
    harness,
    *,
    source: SandboxHandle,
    source_steps: int,
    checkpoint_steps: set[int] | None = None,
) -> dict[int, TreeSearchCheckpointRecord]:
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord] = {}
    assert source.task_run is not None
    for step in range(1, source_steps + 1):
        source.task_run.wait_for_action_delta(delta=1)
        if checkpoint_steps is not None and step not in checkpoint_steps:
            continue
        harness.set_snapshot_metadata(source, tree_search_step=step)
        checkpoint_started = time.perf_counter()
        checkpoint_result = harness.checkpoint_manual(source, leave_running=True)
        checkpoint_finished = time.perf_counter()
        if checkpoint_result is None:
            raise RuntimeError("tree-search benchmark expected a checkpoint at each step")
        checkpoints_by_step[step] = TreeSearchCheckpointRecord(
            checkpoint_id=checkpoint_result.checkpoint_id,
            replay_actions=step,
            checkpoint_ms=(checkpoint_finished - checkpoint_started) * 1000.0,
        )
    return checkpoints_by_step


def collect_auto_checkpoint_indexes(
    harness,
    *,
    source: SandboxHandle,
    source_steps: int,
    allow_sparse_steps: bool = False,
    wait_for_completion: bool = False,
) -> dict[int, TreeSearchCheckpointRecord]:
    if wait_for_completion:
        harness.wait_for_task_completion(
            source,
            timeout_s=task_completion_timeout_seconds(source),
        )
    else:
        assert source.task_run is not None
        source.task_run.wait_for_action_delta(delta=source_steps)
    manifests = harness.wait_for_tree_search_checkpoints(
        source.sandbox_id,
        initial_steps=source_steps,
        require_complete=not allow_sparse_steps,
    )
    if isinstance(manifests, dict):
        return manifests
    return build_tree_search_checkpoint_index(
        manifests,
        initial_steps=source_steps,
        require_complete=not allow_sparse_steps,
    )


def prepare_replay_fork(
    harness,
    source: SandboxHandle,
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord],
    *,
    source_index: int,
    replay_step: int,
) -> PreparedReplayFork:
    checkpoint_step, checkpoint = resolve_tree_search_replay_checkpoint(checkpoints_by_step, replay_step)
    fork = harness.clone_tree_search_checkpoint_to_fork(
        source,
        checkpoint.checkpoint_id,
        f"tree-fork-{source_index}-{replay_step}",
    )
    harness.set_snapshot_metadata(fork, tree_search_is_fork=True)
    return PreparedReplayFork(
        source_index=source_index,
        replay_step=replay_step,
        checkpoint_step=checkpoint_step,
        checkpoint=checkpoint,
        fork=fork,
    )


def restore_prepared_replay_fork(harness, prepared: PreparedReplayFork) -> RestoredReplayFork:
    restore_started = time.perf_counter()
    restore_fn = getattr(harness, "restore_tree_search_fork", harness.restore_once)
    restore_result = restore_fn(prepared.fork, prepared.checkpoint.checkpoint_id)
    if restore_result.status.value != "succeeded":
        raise RuntimeError(
            f"tree-search restore failed for step {prepared.replay_step}: {restore_result.message}"
        )
    recovery_finished = time.perf_counter()
    assert prepared.fork.task_run is not None
    prepared.fork.last_status = prepared.fork.task_run.poll_status()
    ready_at = time.perf_counter()
    return RestoredReplayFork(
        prepared=prepared,
        restore_started_at=restore_started,
        recovery_finished_at=recovery_finished,
        ready_at=ready_at,
        restore_ms=(restore_result.finished_at - restore_result.started_at).total_seconds() * 1000.0,
        recovery_ms=(recovery_finished - restore_started) * 1000.0,
        readiness_ms=(ready_at - recovery_finished) * 1000.0,
        end_to_end_recovery_ms=(ready_at - restore_started) * 1000.0,
    )


def run_replay_progress(
    config: BenchmarkConfig,
    harness,
    restored: RestoredReplayFork,
    *,
    source_sandbox_id: str,
    retained_source_checkpoints: int,
    fork_steps: int,
    run_to_completion: bool,
) -> dict[str, object]:
    fork = restored.prepared.fork
    if fork.task_description is None or fork.task_config is None or fork.agent_type is None:
        raise RuntimeError(f"tree-search fork {fork.sandbox_id} is missing task launch metadata")
    harness.launch_task(
        fork.agent_type,
        fork.task_description,
        fork.task_config,
        str(fork.sandbox_id),
    )
    if run_to_completion:
        harness.wait_for_task_completion(
            fork,
            timeout_s=task_completion_timeout_seconds(fork),
        )
    else:
        wait_for_task_completion_or_stop(
            harness,
            fork,
            action_budget=fork_steps,
        )
    progress_finished = time.perf_counter()
    row = annotate_row(
        config,
        fork,
        iteration=restored.prepared.replay_step,
        success_ratio=1.0,
        row={
            "source_index": restored.prepared.source_index,
            "source_sandbox_id": source_sandbox_id,
            "fork_sandbox_id": str(fork.sandbox_id),
            "replay_step": restored.prepared.replay_step,
            "checkpoint_step": restored.prepared.checkpoint_step,
            "checkpoint_step_gap": restored.prepared.replay_step - restored.prepared.checkpoint_step,
            "checkpoint_ms": restored.prepared.checkpoint.checkpoint_ms,
            "restore_ms": restored.restore_ms,
            "recovery_ms": restored.recovery_ms,
            "readiness_ms": restored.readiness_ms,
            "end_to_end_recovery_ms": restored.end_to_end_recovery_ms,
            "replay_progress_ms": (progress_finished - restored.ready_at) * 1000.0,
            "fanout_ms": (progress_finished - restored.restore_started_at) * 1000.0,
            "replay_actions": restored.prepared.checkpoint.replay_actions,
            "retained_source_checkpoints": retained_source_checkpoints,
        },
    )
    emit_row_telemetry(harness, fork, row, iteration=restored.prepared.replay_step)
    return row


def cleanup_replay_fork(harness, prepared: PreparedReplayFork) -> None:
    harness.deactivate_sandbox_runtime(prepared.fork)
    harness.storage.delete_all_checkpoints(prepared.fork.sandbox_id)
    harness.destroy_sandbox_dataset(prepared.fork)


def cleanup_replay_forks(harness, prepared_forks: list[PreparedReplayFork]) -> None:
    first_error: Exception | None = None
    while prepared_forks:
        prepared = prepared_forks.pop()
        try:
            cleanup_replay_fork(harness, prepared)
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def run_source_replays(
    config: BenchmarkConfig,
    harness,
    *,
    options: TreeOptions,
    task_record: BenchmarkTaskRecord,
    source: SandboxHandle,
    source_index: int,
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord],
    replay_steps: list[int],
    source_run_to_completion: bool,
    fork_run_to_completion: bool,
) -> list[dict[str, object]]:
    source_sandbox_id = str(source.sandbox_id)
    retained_source_checkpoints = len(checkpoints_by_step)
    rows: list[dict[str, object]] = []
    prepared_forks: list[PreparedReplayFork] = []
    if source_run_to_completion:
        harness.wait_for_task_completion(
            source,
            timeout_s=task_completion_timeout_seconds(source),
        )
    else:
        wait_for_task_completion_or_stop(harness, source, action_budget=options.source_steps)
    if source_run_to_completion or task_is_running(source):
        harness.deactivate_sandbox_runtime(source)
    try:
        if options.replay_mode == "concurrent":
            prepared_forks = [
                prepare_replay_fork(
                    harness,
                    source,
                    checkpoints_by_step,
                    source_index=source_index,
                    replay_step=replay_step,
                )
                for replay_step in replay_steps
            ]
            if prepared_forks:
                with ThreadPoolExecutor(max_workers=len(prepared_forks)) as executor:
                    restored_forks = list(executor.map(lambda prepared: restore_prepared_replay_fork(harness, prepared), prepared_forks))
                with ThreadPoolExecutor(max_workers=len(restored_forks)) as executor:
                    rows.extend(
                        executor.map(
                            lambda restored: run_replay_progress(
                                config,
                                harness,
                                restored,
                                source_sandbox_id=source_sandbox_id,
                                retained_source_checkpoints=retained_source_checkpoints,
                                fork_steps=options.fork_steps,
                                run_to_completion=fork_run_to_completion,
                            ),
                            restored_forks,
                        )
                    )
            return rows

        for replay_step in replay_steps:
            prepared = prepare_replay_fork(
                harness,
                source,
                checkpoints_by_step,
                source_index=source_index,
                replay_step=replay_step,
            )
            prepared_forks.append(prepared)
            restored = restore_prepared_replay_fork(harness, prepared)
            rows.append(
                run_replay_progress(
                    config,
                    harness,
                    restored,
                    source_sandbox_id=source_sandbox_id,
                    retained_source_checkpoints=retained_source_checkpoints,
                    fork_steps=options.fork_steps,
                    run_to_completion=fork_run_to_completion,
                )
            )
            cleanup_replay_forks(harness, prepared_forks)
        return rows
    finally:
        cleanup_replay_forks(harness, prepared_forks)


def run_manual_source(
    config: BenchmarkConfig,
    harness,
    *,
    options: TreeOptions,
    task_record: BenchmarkTaskRecord,
    source_index: int,
    source: SandboxHandle,
    replay_steps: list[int],
) -> list[dict[str, object]]:
    try:
        source_steps = effective_source_steps_for_task(config, options, task_record)
        selected_checkpoint_steps = set(replay_steps) if replay_total_steps_for_task(config, task_record) is not None else None
        checkpoints_by_step = collect_manual_checkpoint_indexes(
            harness,
            source=source,
            source_steps=source_steps,
            checkpoint_steps=selected_checkpoint_steps,
        )
        return run_source_replays(
            config,
            harness,
            options=options,
            task_record=task_record,
            source=source,
            source_index=source_index,
            checkpoints_by_step=checkpoints_by_step,
            replay_steps=replay_steps,
            source_run_to_completion=uses_trace_driven_tree_replay(config, task_record),
            fork_run_to_completion=uses_trace_driven_tree_replay(config, task_record),
        )
    finally:
        harness.deactivate_sandbox_runtime(source)
        harness.storage.delete_all_checkpoints(source.sandbox_id)
        harness.destroy_sandbox_dataset(source)


def run_auto_source(
    config: BenchmarkConfig,
    harness,
    *,
    options: TreeOptions,
    task_record: BenchmarkTaskRecord,
    source_index: int,
    source: SandboxHandle,
    replay_steps: list[int],
) -> list[dict[str, object]]:
    try:
        source_steps = effective_source_steps_for_task(config, options, task_record)
        checkpoints_by_step = collect_auto_checkpoint_indexes(
            harness,
            source=source,
            source_steps=source_steps,
            allow_sparse_steps=options.skip_if_no_meaningful_delta,
            wait_for_completion=uses_trace_driven_tree_replay(config, task_record),
        )
        return run_source_replays(
            config,
            harness,
            options=options,
            task_record=task_record,
            source=source,
            source_index=source_index,
            checkpoints_by_step=checkpoints_by_step,
            replay_steps=replay_steps,
            source_run_to_completion=uses_trace_driven_tree_replay(config, task_record),
            fork_run_to_completion=uses_trace_driven_tree_replay(config, task_record),
        )
    finally:
        harness.deactivate_sandbox_runtime(source)
        harness.storage.delete_all_checkpoints(source.sandbox_id)
        harness.destroy_sandbox_dataset(source)


def _source_specs(config: BenchmarkConfig):
    options = parse_tree_options(config)
    if is_replay_llm_service_type(config.llm_service) and config.task_dataset is not None:
        dataset_records = load_task_dataset(config.task_dataset)
        compatible_records = [
            record
            for record in dataset_records
            if effective_trace_replay_progress_count(record) is None
            or options.branch_points <= 0
            or effective_trace_replay_progress_count(record) > 1
        ]
        if len(compatible_records) < config.sandboxes:
            incompatible = [
                (
                    record.task_id or f"task-{index}",
                    effective_trace_replay_progress_count(record),
                )
                for index, record in enumerate(dataset_records)
                if effective_trace_replay_progress_count(record) is not None
                and options.branch_points > 0
                and effective_trace_replay_progress_count(record) <= 1
            ]
            raise ValueError(
                "tree scenario branch_points exceeds replay progress available in the dataset: "
                f"branch_points={options.branch_points}, "
                f"compatible_tasks={len(compatible_records)}, requested_sandboxes={config.sandboxes}, "
                f"incompatible={incompatible[: min(len(incompatible), 10)]}"
            )
        records = compatible_records[: config.sandboxes]
        if config.llm_service_options:
            from benchmarks.core import _apply_llm_service_options  # local import to avoid broad module churn
            records = [_apply_llm_service_options(r, config.llm_service_options) for r in records]
        if (
            config.max_agent_timeout_scale != 1.0
            or config.max_test_timeout_scale != 1.0
            or config.max_agent_timeout_scale_overrides
            or config.max_test_timeout_scale_overrides
        ):
            from benchmarks.core import _apply_task_timeout_scales  # local import to avoid broad module churn
            records = [
                _apply_task_timeout_scales(
                    r,
                    max_agent_timeout_scale=config.max_agent_timeout_scale,
                    max_test_timeout_scale=config.max_test_timeout_scale,
                    max_agent_timeout_scale_overrides=config.max_agent_timeout_scale_overrides,
                    max_test_timeout_scale_overrides=config.max_test_timeout_scale_overrides,
                )
                for r in records
            ]
    else:
        records = resolve_task_records(
            config,
            default_task_description=benchmark_task_description(),
            default_task_config=default_task_config(),
        )
    if is_replay_llm_service_type(config.llm_service) and config.task_dataset is None:
        compatible_records = [
            record
            for record in records
            if effective_trace_replay_progress_count(record) is None
            or options.branch_points <= 0
            or effective_trace_replay_progress_count(record) > 1
        ]
        if len(compatible_records) < config.sandboxes:
            incompatible = [
                (
                    record.task_id or f"task-{index}",
                    effective_trace_replay_progress_count(record),
                )
                for index, record in enumerate(records)
                if effective_trace_replay_progress_count(record) is not None
                and options.branch_points > 0
                and effective_trace_replay_progress_count(record) <= 1
            ]
            raise ValueError(
                "tree scenario branch_points exceeds replay progress for the selected dataset tasks: "
                f"branch_points={options.branch_points}, "
                f"compatible_tasks={len(compatible_records)}, requested_sandboxes={config.sandboxes}, "
                f"incompatible={incompatible[: min(len(incompatible), 10)]}"
            )
        records = compatible_records[: config.sandboxes]
    return make_benchmark_sandbox_specs(
        sandbox_name_prefix="tree-source",
        records=records,
    )


def _setup_source_specs(config: BenchmarkConfig, harness):
    specs = _source_specs(config)
    return setup_task_records_phase(
        harness,
        specs=specs,
        max_workers=config.effective_phase_workers.setup,
    )


def run_manual(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    options = parse_tree_options(config)
    specs = _source_specs(config)
    run_worker_count = min(config.effective_phase_workers.run, max(1, len(specs)))
    if config.phase_merging.setup_and_run:
        indexed_specs = list(enumerate(specs))
        row_groups = benchmark_setup_run_pipeline(
            indexed_specs,
            setup_fn=lambda item: harness.setup_task_record(item[1].sandbox_name, item[1].task_record),
            run_fn=lambda item, prepared: run_manual_source(
                config,
                harness,
                options=options,
                task_record=item[1].task_record,
                source_index=item[0],
                source=start_prepared_task_record(harness, prepared),
                replay_steps=choose_replay_steps_for_task(
                    config,
                    options,
                    item[1].task_record,
                    sandbox_name=item[1].sandbox_name,
                ),
            ),
            setup_max_workers=config.effective_phase_workers.setup,
            run_max_workers=run_worker_count,
            harness=harness,
            setup_item_attributes=lambda item: benchmark_phase_item_attributes(
                harness,
                phase="setup",
                sandbox_name=item[1].sandbox_name,
                task_record=item[1].task_record,
            ),
            run_item_attributes=lambda _item, prepared: benchmark_phase_item_attributes(
                harness,
                phase="run",
                sandbox_name=prepared.sandbox_name,
                sandbox=prepared.handle,
            ),
            executor_pool=config.phase_merging.setup_and_run_executor_pool,
        )
    else:
        prepared_sources = _setup_source_specs(config, harness)
        indexed_sources = list(enumerate(prepared_sources))
        row_groups = benchmark_phase_map(
            indexed_sources,
            lambda item: run_manual_source(
                config,
                harness,
                options=options,
                task_record=item[1].task_record,
                source_index=item[0],
                source=start_prepared_task_record(harness, item[1]),
                replay_steps=choose_replay_steps_for_task(
                    config,
                    options,
                    item[1].task_record,
                    sandbox_name=item[1].sandbox_name,
                ),
            ),
            phase="run",
            max_workers=run_worker_count,
            harness=harness,
            item_attributes=lambda item: benchmark_phase_item_attributes(
                harness,
                phase="run",
                sandbox_name=item[1].sandbox_name,
                sandbox=item[1].handle,
            ),
        )
    if config.verification_enabled:
        row_groups = benchmark_phase_map(
            row_groups,
            lambda group: group,
            phase="verification",
            max_workers=min(config.effective_phase_workers.verification, max(1, len(row_groups))),
            harness=harness,
        )
    else:
        emit_benchmark_phase_skipped(
            phase="verification",
            sandbox_count=len(row_groups),
            configured_max_workers=min(config.effective_phase_workers.verification, max(1, len(row_groups))),
        )
    rows = [row for group in row_groups for row in group]
    return sorted(rows, key=lambda row: (int(row["source_index"]), int(row["replay_step"]), str(row["fork_sandbox_id"])))


def run_auto(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    options = parse_tree_options(config)
    specs = _source_specs(config)
    run_worker_count = min(config.effective_phase_workers.run, max(1, len(specs)))
    if hasattr(harness, "drain_request_state_changes"):
        harness.drain_request_state_changes()
    try:
        if config.phase_merging.setup_and_run:
            indexed_specs = list(enumerate(specs))
            row_groups = benchmark_setup_run_pipeline(
                indexed_specs,
                setup_fn=lambda item: harness.setup_task_record(item[1].sandbox_name, item[1].task_record),
                run_fn=lambda item, prepared: run_auto_source(
                    config,
                    harness,
                    options=options,
                    task_record=item[1].task_record,
                    source_index=item[0],
                    source=start_prepared_task_record(harness, prepared),
                    replay_steps=choose_replay_steps_for_task(
                        config,
                        options,
                        item[1].task_record,
                        sandbox_name=item[1].sandbox_name,
                    ),
                ),
                setup_max_workers=config.effective_phase_workers.setup,
                run_max_workers=run_worker_count,
                harness=harness,
                setup_item_attributes=lambda item: benchmark_phase_item_attributes(
                    harness,
                    phase="setup",
                    sandbox_name=item[1].sandbox_name,
                    task_record=item[1].task_record,
                ),
                run_item_attributes=lambda _item, prepared: benchmark_phase_item_attributes(
                    harness,
                    phase="run",
                    sandbox_name=prepared.sandbox_name,
                    sandbox=prepared.handle,
                ),
                executor_pool=config.phase_merging.setup_and_run_executor_pool,
            )
        else:
            prepared_sources = _setup_source_specs(config, harness)
            indexed_sources = list(enumerate(prepared_sources))
            row_groups = benchmark_phase_map(
                indexed_sources,
                lambda item: run_auto_source(
                    config,
                    harness,
                    options=options,
                    task_record=item[1].task_record,
                    source_index=item[0],
                    source=start_prepared_task_record(harness, item[1]),
                    replay_steps=choose_replay_steps_for_task(
                        config,
                        options,
                        item[1].task_record,
                        sandbox_name=item[1].sandbox_name,
                    ),
                ),
                phase="run",
                max_workers=run_worker_count,
                harness=harness,
                item_attributes=lambda item: benchmark_phase_item_attributes(
                    harness,
                    phase="run",
                    sandbox_name=item[1].sandbox_name,
                    sandbox=item[1].handle,
                ),
            )
    finally:
        if hasattr(harness, "drain_request_state_changes"):
            harness.drain_request_state_changes()
    if config.verification_enabled:
        row_groups = benchmark_phase_map(
            row_groups,
            lambda group: group,
            phase="verification",
            max_workers=min(config.effective_phase_workers.verification, max(1, len(row_groups))),
            harness=harness,
        )
    else:
        emit_benchmark_phase_skipped(
            phase="verification",
            sandbox_count=len(row_groups),
            configured_max_workers=min(config.effective_phase_workers.verification, max(1, len(row_groups))),
        )
    rows = [row for group in row_groups for row in group]
    return sorted(rows, key=lambda row: (int(row["source_index"]), int(row["replay_step"]), str(row["fork_sandbox_id"])))


def prepare_harness(config: BenchmarkConfig, harness) -> None:
    if config.mode == "auto":
        harness.add_interceptor_hook(TreeSearchStepHook(harness))


def summarize(config: BenchmarkConfig, rows: list[dict[str, object]]) -> dict[str, float]:
    del config
    return compute_summary(
        rows,
        [
            "checkpoint_ms",
            "restore_ms",
            "recovery_ms",
            "readiness_ms",
            "end_to_end_recovery_ms",
            "replay_progress_ms",
            "fanout_ms",
            "retained_source_checkpoints",
        ],
    )


SCENARIO = ScenarioDefinition(
    name="tree",
    supported_modes=frozenset({"manual", "auto"}),
    build_harness_settings=build_harness_settings,
    run_manual=run_manual,
    run_auto=run_auto,
    summarize=summarize,
    prepare_harness=prepare_harness,
)

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time

from agent_cr import (
    KeepAllCheckpointManager,
    RequestContext,
    RequestInterceptorHook,
    SchedulerConfig,
    TreeSearchCheckpointingPolicy,
)
from integrations.agents import SandboxHandle, TaskConfig, TaskDescription

from benchmarks.config import BenchmarkConfig
from benchmarks.core import annotate_row, launch_task_records, parallel_map, resolve_task_records
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from benchmarks.support import (
    TreeSearchCheckpointRecord,
    build_tree_search_checkpoint_index,
    compute_summary,
)


_TASK_STOP_REQUEST_GRACE_SECONDS = 1.0
_TASK_FORCE_STOP_GRACE_SECONDS = 5.0


@dataclass(frozen=True)
class TreeOptions:
    source_steps: int = 6
    branch_points: int = 2
    fork_steps: int = 3
    replay_mode: str = "sequential"


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
    )


def build_harness_settings(config: BenchmarkConfig) -> HarnessSettings:
    options = parse_tree_options(config)
    scheduler_config = SchedulerConfig(
        min_checkpoint_interval_seconds=0.0,
        force_checkpoint_after_seconds=0.0,
        require_change_signal=False,
    )
    return HarnessSettings(
        scheduler_config=scheduler_config,
        scheduler_policy=TreeSearchCheckpointingPolicy(),
        checkpoint_manager_factory=lambda base: KeepAllCheckpointManager(base),
        max_workers=max(1, config.sandboxes * max(1, options.branch_points + 1)),
    )


def choose_replay_steps(source_steps: int, branch_points: int) -> list[int]:
    if source_steps <= 1 or branch_points <= 0:
        return []
    candidates = list(range(1, source_steps))
    if branch_points >= len(candidates):
        return candidates
    stride = max(1, len(candidates) // branch_points)
    return candidates[::stride][:branch_points]


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
) -> dict[int, TreeSearchCheckpointRecord]:
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord] = {}
    assert source.task_run is not None
    for step in range(1, source_steps + 1):
        source.task_run.wait_for_action_delta(delta=1)
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
) -> dict[int, TreeSearchCheckpointRecord]:
    assert source.task_run is not None
    source.task_run.wait_for_action_delta(delta=source_steps)
    manifests = harness.wait_for_tree_search_checkpoints(
        source.sandbox_id,
        initial_steps=source_steps,
    )
    if isinstance(manifests, dict):
        return manifests
    return build_tree_search_checkpoint_index(manifests, initial_steps=source_steps, require_complete=True)


def prepare_replay_fork(
    harness,
    source: SandboxHandle,
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord],
    *,
    source_index: int,
    replay_step: int,
) -> PreparedReplayFork:
    checkpoint = checkpoints_by_step[replay_step]
    fork = harness.clone_tree_search_checkpoint_to_fork(
        source,
        checkpoint.checkpoint_id,
        f"tree-fork-{source_index}-{replay_step}",
    )
    return PreparedReplayFork(
        source_index=source_index,
        replay_step=replay_step,
        checkpoint=checkpoint,
        fork=fork,
    )


def restore_prepared_replay_fork(harness, prepared: PreparedReplayFork) -> RestoredReplayFork:
    restore_started = time.perf_counter()
    restore_result = harness.restore_once(prepared.fork, prepared.checkpoint.checkpoint_id)
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
    wait_for_task_completion_or_stop(
        harness,
        fork,
        action_budget=fork_steps,
    )
    progress_finished = time.perf_counter()
    return annotate_row(
        config,
        fork,
        iteration=restored.prepared.replay_step,
        success_ratio=1.0,
        row={
            "source_index": restored.prepared.source_index,
            "source_sandbox_id": source_sandbox_id,
            "fork_sandbox_id": str(fork.sandbox_id),
            "replay_step": restored.prepared.replay_step,
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


def cleanup_replay_fork(harness, prepared: PreparedReplayFork) -> None:
    harness.deactivate_sandbox_runtime(prepared.fork)
    harness.storage.delete_all_checkpoints(prepared.fork.sandbox_id)
    harness.destroy_sandbox_dataset(prepared.fork)


def run_source_replays(
    config: BenchmarkConfig,
    harness,
    *,
    options: TreeOptions,
    source: SandboxHandle,
    source_index: int,
    checkpoints_by_step: dict[int, TreeSearchCheckpointRecord],
    replay_steps: list[int],
) -> list[dict[str, object]]:
    source_sandbox_id = str(source.sandbox_id)
    retained_source_checkpoints = len(checkpoints_by_step)
    rows: list[dict[str, object]] = []
    prepared_forks: list[PreparedReplayFork] = []
    wait_for_task_completion_or_stop(harness, source, action_budget=options.source_steps)
    if task_is_running(source):
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
                )
            )
            cleanup_replay_fork(harness, prepared)
            prepared_forks.remove(prepared)
        return rows
    finally:
        for prepared in list(prepared_forks):
            cleanup_replay_fork(harness, prepared)


def run_manual_source(
    config: BenchmarkConfig,
    harness,
    *,
    options: TreeOptions,
    source_index: int,
    source: SandboxHandle,
    replay_steps: list[int],
) -> list[dict[str, object]]:
    try:
        checkpoints_by_step = collect_manual_checkpoint_indexes(
            harness,
            source=source,
            source_steps=options.source_steps,
        )
        return run_source_replays(
            config,
            harness,
            options=options,
            source=source,
            source_index=source_index,
            checkpoints_by_step=checkpoints_by_step,
            replay_steps=replay_steps,
        )
    finally:
        harness.storage.delete_all_checkpoints(source.sandbox_id)
        harness.destroy_sandbox_dataset(source)


def run_auto_source(
    config: BenchmarkConfig,
    harness,
    *,
    options: TreeOptions,
    source_index: int,
    source: SandboxHandle,
    replay_steps: list[int],
) -> list[dict[str, object]]:
    try:
        checkpoints_by_step = collect_auto_checkpoint_indexes(
            harness,
            source=source,
            source_steps=options.source_steps,
        )
        return run_source_replays(
            config,
            harness,
            options=options,
            source=source,
            source_index=source_index,
            checkpoints_by_step=checkpoints_by_step,
            replay_steps=replay_steps,
        )
    finally:
        harness.storage.delete_all_checkpoints(source.sandbox_id)
        harness.destroy_sandbox_dataset(source)


def _launch_sources(config: BenchmarkConfig, harness) -> list[SandboxHandle]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    return launch_task_records(
        harness,
        sandbox_name_prefix="tree-source",
        records=records,
        max_workers=config.sandboxes,
    )


def run_manual(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    options = parse_tree_options(config)
    replay_steps = choose_replay_steps(options.source_steps, options.branch_points)
    sources = _launch_sources(config, harness)
    indexed_sources = list(enumerate(sources))
    try:
        row_groups = parallel_map(
            indexed_sources,
            lambda item: run_manual_source(
                config,
                harness,
                options=options,
                source_index=item[0],
                source=item[1],
                replay_steps=replay_steps,
            ),
            max_workers=max(1, len(indexed_sources)),
        )
    finally:
        for source in sources:
            harness.storage.delete_all_checkpoints(source.sandbox_id)
            harness.destroy_sandbox_dataset(source)
    rows = [row for group in row_groups for row in group]
    return sorted(rows, key=lambda row: (int(row["source_index"]), int(row["replay_step"]), str(row["fork_sandbox_id"])))


def run_auto(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    options = parse_tree_options(config)
    replay_steps = choose_replay_steps(options.source_steps, options.branch_points)
    sources = _launch_sources(config, harness)
    indexed_sources = list(enumerate(sources))
    if hasattr(harness, "drain_request_state_changes"):
        harness.drain_request_state_changes()
    try:
        row_groups = parallel_map(
            indexed_sources,
            lambda item: run_auto_source(
                config,
                harness,
                options=options,
                source_index=item[0],
                source=item[1],
                replay_steps=replay_steps,
            ),
            max_workers=max(1, len(indexed_sources)),
        )
    finally:
        if hasattr(harness, "drain_request_state_changes"):
            harness.drain_request_state_changes()
        for source in sources:
            harness.storage.delete_all_checkpoints(source.sandbox_id)
            harness.destroy_sandbox_dataset(source)
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

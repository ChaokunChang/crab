from __future__ import annotations

import math
import random
import time

from agent_cr import CheckpointId
from integrations.agents import SandboxHandle, TaskConfig

from benchmarks.config import BenchmarkConfig
from benchmarks.core import annotate_row, emit_row_telemetry, poll_sandbox_status, trace_response_count_for_sandbox, verify_task_accuracy
from benchmarks.support import average, task_timeout_seconds, total_actions


def should_inject_event(
    *,
    chunk_index: int,
    sandbox_index: int,
    rate: float,
    first_forced_event_chunk: int,
    rng: random.Random,
) -> bool:
    _ = sandbox_index
    if first_forced_event_chunk > 0 and chunk_index < first_forced_event_chunk:
        return False
    if first_forced_event_chunk > 0 and chunk_index == first_forced_event_chunk:
        return True
    return rng.random() < rate


def required_event_was_missed(
    *,
    chunk_index: int,
    events_injected: int,
    first_forced_event_chunk: int,
) -> bool:
    return first_forced_event_chunk > 0 and chunk_index >= first_forced_event_chunk and events_injected == 0


def wait_for_iteration_progress(
    sandbox: SandboxHandle,
    *,
    iteration: int,
    initial_actions: int = 6,
    later_action_delta: int = 1,
) -> dict[str, object]:
    assert sandbox.task_run is not None
    if iteration == 1:
        return sandbox.task_run.wait_for_progress(minimum_actions=initial_actions)
    return sandbox.task_run.wait_for_action_delta(delta=later_action_delta)


def replay_status_is_complete(status: dict[str, object], *, trace_response_count: int) -> bool:
    if bool(status.get("replay_is_complete", False)):
        return True
    if trace_response_count <= 0:
        return False
    return total_actions(status) >= trace_response_count


def manifest_replay_next_response_index(manifest) -> int:
    metadata = getattr(manifest, "metadata", {})
    if not isinstance(metadata, dict):
        return 0
    raw_value = metadata.get("benchmark_replay_action_count", 0)
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def wait_for_auto_replay_checkpoint(
    harness,
    sandbox: SandboxHandle,
    *,
    minimum_actions: int,
    trace_response_count: int,
) -> tuple[object | None, int]:
    timeout_s = max(30.0, task_timeout_seconds(sandbox.task_config or TaskConfig()))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        manifests = harness.list_checkpoint_manifests(sandbox.sandbox_id)
        for manifest in reversed(manifests):
            checkpoint_actions = manifest_replay_next_response_index(manifest)
            if checkpoint_actions < minimum_actions:
                continue
            return manifest, checkpoint_actions

        current = poll_sandbox_status(sandbox)
        if replay_status_is_complete(current, trace_response_count=trace_response_count):
            return None, total_actions(current)
        if str(current.get("state", "")) == "finished":
            return None, total_actions(current)
        time.sleep(0.2)

    current = poll_sandbox_status(sandbox)
    if replay_status_is_complete(current, trace_response_count=trace_response_count):
        return None, total_actions(current)
    raise RuntimeError(f"timed out waiting for auto replay checkpoint at action count {minimum_actions}")


def finalize_replay_row(
    config: BenchmarkConfig,
    harness,
    sandbox: SandboxHandle,
    *,
    row: dict[str, object],
    task_error: str = "",
    iteration: int = 1,
    verify_task_accuracy_result: bool = True,
    success_ratio: float | None = None,
) -> dict[str, object]:
    trace_response_count = trace_response_count_for_sandbox(sandbox)
    if verify_task_accuracy_result:
        verification = {
            "verification_status": "task_failed" if task_error else "verification_skipped",
            "verification_exit_code": -1,
            "verification_ms": 0.0,
        }
        if not task_error:
            task_error, verification = verify_task_accuracy(harness, sandbox)
    else:
        verification = {}
    status = poll_sandbox_status(sandbox)
    replay_is_complete = replay_status_is_complete(status, trace_response_count=trace_response_count)
    raw_replay_final_index = int(status.get("replay_next_response_index", status.get("total_actions", 0)))
    replay_final_index = raw_replay_final_index
    if replay_is_complete and trace_response_count > 0:
        replay_final_index = min(replay_final_index, trace_response_count)
    if not task_error and str(status.get("state", "")) == "finished" and not replay_is_complete:
        task_error = (
            "replay task finished before reaching the end of the trace "
            f"(replay_final_index={raw_replay_final_index}, trace_response_count={trace_response_count})"
        )
        if verify_task_accuracy_result:
            verification = {
                **verification,
                "verification_status": "task_failed",
                "verification_exit_code": -1,
            }
    row_payload = {
        **row,
        "trace_response_count": trace_response_count,
        "replay_final_index": replay_final_index,
        "replay_is_complete": replay_is_complete,
        **verification,
    }
    if verify_task_accuracy_result:
        resolved_success_ratio = 1.0 if not task_error and verification["verification_status"] == "passed" else 0.0
    else:
        resolved_success_ratio = 0.0 if task_error else (1.0 if success_ratio is None else float(success_ratio))
    annotated = annotate_row(
        config,
        sandbox,
        iteration=iteration,
        success_ratio=resolved_success_ratio,
        task_error=task_error,
        row=row_payload,
    )
    emit_row_telemetry(harness, sandbox, annotated, iteration=iteration)
    return annotated


def choose_replay_chunk_targets(
    sandbox: SandboxHandle,
    chunk_count: int,
) -> list[int]:
    total_responses = trace_response_count_for_sandbox(sandbox)
    if total_responses <= 0 or chunk_count <= 0:
        return []

    targets: list[int] = []
    for chunk_index in range(1, chunk_count + 1):
        target = min(total_responses, max(1, math.ceil((total_responses * chunk_index) / chunk_count)))
        if targets and target <= targets[-1]:
            continue
        targets.append(target)
    return targets


def summarize_metric_averages(metric_lists: dict[str, list[float]]) -> dict[str, float]:
    return {key: average(values) for key, values in metric_lists.items()}

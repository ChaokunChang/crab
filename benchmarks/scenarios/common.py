from __future__ import annotations

import random
import time

from agent_cr import CheckpointId
from integrations.agents import SandboxHandle, TaskConfig

from benchmarks.config import BenchmarkConfig
from benchmarks.core import annotate_row, poll_sandbox_status, trace_response_count_for_sandbox, verify_task_accuracy
from benchmarks.support import average, choose_replay_points, task_timeout_seconds, total_actions


def should_inject_event(
    *,
    iteration: int,
    sandbox_index: int,
    rate: float,
    first_injection_iteration: int,
    rng: random.Random,
) -> bool:
    if first_injection_iteration > 0 and iteration < first_injection_iteration:
        return False
    if first_injection_iteration > 0 and iteration == first_injection_iteration and sandbox_index == 0:
        return True
    return rng.random() < rate


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
    status = poll_sandbox_status(sandbox)
    row_payload = {
        **row,
        "trace_response_count": trace_response_count_for_sandbox(sandbox),
        "replay_final_index": int(status.get("replay_next_response_index", status.get("total_actions", 0))),
        "replay_is_complete": replay_status_is_complete(
            status,
            trace_response_count=trace_response_count_for_sandbox(sandbox),
        ),
    }
    if verify_task_accuracy_result:
        verification = {
            "verification_status": "task_failed" if task_error else "verification_skipped",
            "verification_exit_code": -1,
            "verification_ms": 0.0,
        }
        if not task_error:
            task_error, verification = verify_task_accuracy(harness, sandbox)
        resolved_success_ratio = 1.0 if verification["verification_status"] == "passed" else 0.0
        row_payload = {
            **row_payload,
            **verification,
        }
    else:
        resolved_success_ratio = 0.0 if task_error else (1.0 if success_ratio is None else float(success_ratio))
    return annotate_row(
        config,
        sandbox,
        iteration=iteration,
        success_ratio=resolved_success_ratio,
        task_error=task_error,
        row=row_payload,
    )


def choose_replay_targets(sandbox: SandboxHandle, iterations: int) -> list[int]:
    return choose_replay_points(trace_response_count_for_sandbox(sandbox), iterations)


def summarize_metric_averages(metric_lists: dict[str, list[float]]) -> dict[str, float]:
    return {key: average(values) for key, values in metric_lists.items()}

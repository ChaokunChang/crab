from __future__ import annotations

import logging
import math
import random
import time

from agent_cr import CheckpointId
from agent_cr.models import SchedulerCheckpointDecision
from integrations.agents import SandboxHandle, TaskConfig

from benchmarks.config import BenchmarkConfig
from benchmarks.core import (
    annotate_row,
    emit_row_telemetry,
    poll_sandbox_status,
    replay_action_count_wait_error,
    replay_status_is_complete,
    replay_trace_cursor,
    task_completion_timeout_seconds,
    trace_response_count_for_sandbox,
    verify_task_accuracy,
)
from benchmarks.support import average, task_timeout_seconds, total_actions

logger = logging.getLogger(__name__)


class NoCheckpointingPolicy:
    def __init__(self, *, reason: str = "scheduler_policy_no_checkpointing") -> None:
        self._reason = reason

    @property
    def name(self) -> str:
        return "no-checkpointing"

    def evaluate(self, snapshot) -> SchedulerCheckpointDecision:
        return SchedulerCheckpointDecision(
            should_checkpoint=False,
            checkpoint_process=False,
            checkpoint_filesystem=False,
            leave_running=False,
            reason=self._reason,
            policy_name=self.name,
        )


def resolve_scheduler_policy_override(
    config: BenchmarkConfig,
    *,
    scenario_default_policy,
):
    policy = config.scheduler.policy
    if policy in (None, "scenario_default"):
        return scenario_default_policy
    if policy == "no_checkpointing":
        return NoCheckpointingPolicy()
    raise ValueError(f"unsupported scheduler.policy={policy!r} for scenario {config.scenario!r}")


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


def manifest_trace_cursor(manifest) -> int:
    metadata = getattr(manifest, "metadata", {})
    if not isinstance(metadata, dict):
        return 0
    raw_value = metadata.get("benchmark_trace_cursor", 0)
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
            checkpoint_actions = manifest_trace_cursor(manifest)
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


def cleanup_finished_replay_sandbox_after_run(
    config: BenchmarkConfig,
    harness,
    sandbox: SandboxHandle,
    *,
    task_error: str = "",
) -> str:
    # Bridges the run phase and the verification phase. When verification is
    # enabled the next step will `runc exec` inside the sandbox, so we must
    # enforce the handoff invariant: the agent's task_future has finished, no
    # executor jobs are pending or running, the scheduler has stopped
    # scheduling checkpoints, and the container is not paused. Violating any
    # of these collides verify's exec with an in-flight scheduler pause and
    # produces "cannot exec in a paused container" (seen in run
    # 20260420_123846, sandbox spec-91-spec-81).
    if not config.verification_enabled and not config.phase_merging.setup_and_run:
        return task_error

    completed = str(sandbox.last_status.get("state", "")) == "finished"
    if not completed:
        try:
            harness.wait_for_task_completion(
                sandbox,
                timeout_s=task_completion_timeout_seconds(sandbox),
            )
        except Exception as exc:
            if not task_error:
                task_error = str(exc) or exc.__class__.__name__
            task_future = sandbox.task_future
            completed = bool(task_future is not None and task_future.done())
        else:
            completed = True

    poll_sandbox_status(sandbox)

    if config.verification_enabled:
        system = getattr(harness, "system", None)
        if system is not None:
            try:
                system.quiesce_for_verification(sandbox.sandbox_id)
            except Exception:
                logger.exception(
                    "Failed to quiesce sandbox %s for verification handoff",
                    sandbox.sandbox_id,
                )
        return task_error

    if not completed:
        return task_error

    deactivate_sandbox_runtime = getattr(harness, "deactivate_sandbox_runtime", None)
    if callable(deactivate_sandbox_runtime):
        deactivate_sandbox_runtime(sandbox)
    return task_error


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
    pre_verification_status = poll_sandbox_status(sandbox)
    pre_verification_replay_is_complete = replay_status_is_complete(
        pre_verification_status,
        trace_response_count=trace_response_count,
    )
    normalized_task_error = task_error
    if pre_verification_replay_is_complete and replay_action_count_wait_error(task_error):
        logger.info(
            "Ignoring stale replay progress error after replay completed sandbox=%s task=%s error=%r "
            "replay_final_trace_cursor=%d trace_response_count=%d",
            sandbox.sandbox_id,
            getattr(sandbox.sandbox_id, "value", str(sandbox.sandbox_id)),
            task_error,
            replay_trace_cursor(pre_verification_status),
            trace_response_count,
        )
        normalized_task_error = ""
    should_verify = verify_task_accuracy_result and config.verification_enabled
    initial_task_error = task_error
    if should_verify:
        verification = {
            "verification_status": "task_failed" if normalized_task_error else "verification_skipped",
            "verification_exit_code": -1,
            "verification_ms": 0.0,
        }
        if not normalized_task_error:
            normalized_task_error, verification = verify_task_accuracy(harness, sandbox)
    else:
        verification = {}
    status = poll_sandbox_status(sandbox)
    post_verification_replay_is_complete = replay_status_is_complete(status, trace_response_count=trace_response_count)
    replay_is_complete = pre_verification_replay_is_complete or post_verification_replay_is_complete
    raw_replay_final_trace_cursor = max(
        replay_trace_cursor(pre_verification_status),
        replay_trace_cursor(status),
    )
    replay_final_trace_cursor = raw_replay_final_trace_cursor
    if replay_is_complete and trace_response_count > 0:
        replay_final_trace_cursor = trace_response_count
    if not normalized_task_error and str(status.get("state", "")) == "finished" and not replay_is_complete:
        normalized_task_error = (
            "replay task finished before reaching the end of the trace "
            f"(replay_final_trace_cursor={raw_replay_final_trace_cursor}, trace_response_count={trace_response_count})"
        )
        if should_verify:
            verification = {
                **verification,
                "verification_status": "task_failed",
                "verification_exit_code": -1,
            }
    row_payload = {
        **row,
        "trace_response_count": trace_response_count,
        "replay_final_trace_cursor": replay_final_trace_cursor,
        "replay_is_complete": replay_is_complete,
        **verification,
    }
    if should_verify:
        resolved_success_ratio = (
            1.0 if not normalized_task_error and verification["verification_status"] == "passed" else 0.0
        )
    else:
        resolved_success_ratio = (
            0.0 if normalized_task_error else (1.0 if success_ratio is None else float(success_ratio))
        )
    annotated = annotate_row(
        config,
        sandbox,
        iteration=iteration,
        success_ratio=resolved_success_ratio,
        task_error=normalized_task_error,
        row=row_payload,
    )
    emit_event = getattr(harness, "emit_benchmark_event", None)
    if callable(emit_event):
        emit_event(
            "benchmark.replay.row",
            sandbox,
            iteration=iteration,
            event_type=None if row.get("event_type") is None else str(row.get("event_type")),
            extra={
                "should_verify": bool(should_verify),
                "initial_task_error": str(initial_task_error),
                "task_error": str(normalized_task_error),
                "verification_status": str(row_payload.get("verification_status", "")),
                "replay_is_complete": bool(replay_is_complete),
                "replay_final_trace_cursor": int(replay_final_trace_cursor),
                "trace_response_count": int(trace_response_count),
            },
        )
    logger.info(
        "Replay row finalized sandbox=%s task=%s should_verify=%s initial_task_error=%r task_error=%r "
        "verification_status=%s replay_final_trace_cursor=%d trace_response_count=%d replay_is_complete=%s "
        "success_ratio=%.3f",
        sandbox.sandbox_id,
        annotated.get("task_id", ""),
        should_verify,
        initial_task_error,
        normalized_task_error,
        row_payload.get("verification_status", ""),
        replay_final_trace_cursor,
        trace_response_count,
        replay_is_complete,
        resolved_success_ratio,
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

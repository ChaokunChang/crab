from __future__ import annotations

import time

from agent_cr import SchedulerConfig
from integrations.agents import TaskConfig, TaskDescription

from benchmarks.config import BenchmarkConfig
from benchmarks.core import (
    annotate_row,
    launch_task_records,
    poll_sandbox_status,
    resolve_task_records,
    trace_response_count_for_sandbox,
)
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from benchmarks.support import (
    compute_summary,
    is_replay_llm_service_type,
    task_timeout_seconds,
    verification_timeout_seconds,
    wait_for,
)


def benchmark_task_description() -> TaskDescription:
    return TaskDescription(
        "Start the benchmark workload inside the sandbox and keep making progress through filesystem, "
        "process, network, and stateful actions."
    )


def default_task_config() -> TaskConfig:
    return TaskConfig()


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
        scheduler_policy=None,
        checkpoint_manager_factory=lambda base: base,
        max_workers=config.effective_max_workers,
    )


def _run_replay_accuracy_sandbox(
    config: BenchmarkConfig,
    harness,
    sandbox,
) -> dict[str, object]:
    task_started = time.perf_counter()
    task_error = ""
    verification = {
        "verification_status": "task_failed",
        "verification_exit_code": -1,
        "verification_ms": 0.0,
    }
    try:
        harness.wait_for_task_completion(
            sandbox,
            timeout_s=task_timeout_seconds(sandbox.task_config or TaskConfig()),
        )
    except Exception as exc:
        task_error = str(exc)
    task_completion_ms = (time.perf_counter() - task_started) * 1000.0

    if not task_error:
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

    status = poll_sandbox_status(sandbox)
    success_ratio = 1.0 if verification["verification_status"] == "passed" else 0.0
    return annotate_row(
        config,
        sandbox,
        iteration=1,
        success_ratio=success_ratio,
        task_error=task_error,
        row={
            "trace_response_count": trace_response_count_for_sandbox(sandbox),
            "task_completion_ms": task_completion_ms,
            "tool_actions": int(status.get("total_actions", 0)),
            "fs_actions": int(status.get("filesystem_actions", status.get("total_actions", 0))),
            "process_actions": int(status.get("process_actions", status.get("total_actions", 0))),
            "network_actions": int(status.get("network_actions", status.get("total_actions", 0))),
            "replay_final_index": int(status.get("replay_next_response_index", status.get("total_actions", 0))),
            **verification,
        },
    )


def run_manual(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    sandboxes = launch_task_records(
        harness,
        sandbox_name_prefix="sandbox",
        records=records,
        max_workers=config.effective_max_workers,
    )
    if sandboxes and any(is_replay_llm_service_type(sandbox.llm_service_type) for sandbox in sandboxes):
        rows = [
            _run_replay_accuracy_sandbox(config, harness, sandbox)
            for sandbox in sandboxes
        ]
        return sorted(rows, key=lambda row: str(row["sandbox_id"]))

    rows: list[dict[str, object]] = []
    for iteration in range(1, config.iterations + 1):
        checkpoint_results = []
        for sandbox in sandboxes:
            assert sandbox.task_run is not None
            sandbox.task_run.wait_for_progress(minimum_actions=6)
            if harness.request_state_store is not None and not wait_for(
                lambda sid=sandbox.sandbox_id: harness.request_state_store.get(sid).llm_request_in_flight,
                timeout_s=20.0,
                raise_on_timeout=False,
            ):
                continue
        t0 = time.perf_counter()
        for sandbox in sandboxes:
            result = harness.checkpoint_if_due(sandbox)
            if result is not None:
                checkpoint_results.append(result)
        t1 = time.perf_counter()
        restore_results = [
            harness.restore_once(harness.get_sandbox_handle(str(result.sandbox_id)), result.checkpoint_id)
            for result in checkpoint_results
        ]
        t2 = time.perf_counter()
        for sandbox in sandboxes:
            harness.set_snapshot_metadata(sandbox)
        rows.append(
            {
                "scenario": config.scenario,
                "mode": config.mode,
                "provider": config.provider,
                "agent": config.agent,
                "llm_service": config.llm_service or "",
                "sandbox_id": "batch",
                "task_id": "batch",
                "iteration": iteration,
                "success_ratio": (
                    sum(1 for item in restore_results if item.status.value == "succeeded")
                    / max(1, len(restore_results))
                ),
                "task_error": "",
                "sandbox_count": config.sandboxes,
                "tool_actions": sum(int(sandbox.last_status["total_actions"]) for sandbox in sandboxes),
                "fs_actions": sum(int(sandbox.last_status["filesystem_actions"]) for sandbox in sandboxes),
                "process_actions": sum(int(sandbox.last_status["process_actions"]) for sandbox in sandboxes),
                "network_actions": sum(int(sandbox.last_status["network_actions"]) for sandbox in sandboxes),
                "checkpoint_count": len(checkpoint_results),
                "restore_count": len(restore_results),
                "checkpoint_batch_ms": (t1 - t0) * 1000.0,
                "restore_batch_ms": (t2 - t1) * 1000.0,
            }
        )
    return rows


def summary_metrics(config: BenchmarkConfig, rows: list[dict[str, object]]) -> dict[str, float]:
    del config
    metric_keys = ["checkpoint_batch_ms", "restore_batch_ms", "success_ratio"]
    if rows and "verification_status" in rows[0]:
        metric_keys = ["task_completion_ms", "verification_ms", "success_ratio"]
    return compute_summary(rows, metric_keys)


SCENARIO = ScenarioDefinition(
    name="e2e",
    supported_modes=frozenset({"manual"}),
    build_harness_settings=build_harness_settings,
    run_manual=run_manual,
    run_auto=None,
    summarize=summary_metrics,
)

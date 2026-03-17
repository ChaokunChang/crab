#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import SchedulerConfig
from integrations.agents import TaskConfig, TaskDescription
from benchmarks.support import (
    add_common_args,
    compute_summary,
    configure_logging,
    is_replay_llm_service_type,
    task_timeout_seconds,
    verification_timeout_seconds,
    wait_for,
    write_rows,
)
from benchmarks.real_host_scenario_base import (
    RealHostScenarioHarness,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-CR sandboxed benchmark")
    parser.add_argument("--sandboxes", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    add_common_args(parser)
    return parser.parse_args()


def benchmark_task_description() -> TaskDescription:
    return TaskDescription(
        "Start the benchmark workload inside the sandbox and keep making progress through filesystem, "
        "process, network, and stateful actions."
    )


def default_task_config() -> TaskConfig:
    return TaskConfig()


def _sandbox_benchmark_metadata(sandbox) -> dict[str, object]:
    metadata = sandbox.launch_metadata.get("benchmark", {})
    return metadata if isinstance(metadata, dict) else {}


def _task_id_for_sandbox(sandbox) -> str:
    metadata = _sandbox_benchmark_metadata(sandbox)
    raw_task_id = metadata.get("task_id")
    if isinstance(raw_task_id, str) and raw_task_id:
        return raw_task_id
    if sandbox.task_config is not None:
        raw_task_id = sandbox.task_config.options.get("task_id")
        if isinstance(raw_task_id, str) and raw_task_id:
            return raw_task_id
    return str(sandbox.sandbox_id)


def _trace_response_count_for_sandbox(sandbox) -> int:
    metadata = _sandbox_benchmark_metadata(sandbox)
    raw_value = metadata.get("trace_response_count")
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def _run_replay_accuracy_sandbox(
    args: argparse.Namespace,
    harness: RealHostScenarioHarness,
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
    status = dict(sandbox.last_status)
    if sandbox.task_run is not None:
        try:
            status = sandbox.task_run.poll_status()
        except Exception:
            status = dict(sandbox.last_status)
    sandbox.last_status = dict(status)
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
    return {
        "iter": 1,
        "provider": args.provider,
        "agent_type": args.agent_type,
        "sandbox_id": str(sandbox.sandbox_id),
        "task_id": _task_id_for_sandbox(sandbox),
        "trace_response_count": _trace_response_count_for_sandbox(sandbox),
        "task_completion_ms": task_completion_ms,
        "tool_actions": int(status.get("total_actions", 0)),
        "fs_actions": int(status.get("filesystem_actions", status.get("total_actions", 0))),
        "process_actions": int(status.get("process_actions", status.get("total_actions", 0))),
        "network_actions": int(status.get("network_actions", status.get("total_actions", 0))),
        "replay_final_index": int(status.get("replay_next_response_index", status.get("total_actions", 0))),
        "success_ratio": 1.0 if verification["verification_status"] == "passed" else 0.0,
        "task_error": task_error,
        **verification,
    }


def run_benchmark(args: argparse.Namespace, harness: RealHostScenarioHarness) -> list[dict[str, object]]:
    dataset_path = getattr(args, "dataset", None)
    dataset = harness.load_dataset(dataset_path) if dataset_path is not None else None
    with ThreadPoolExecutor(max_workers=max(1, args.sandboxes)) as launcher:
        sandboxes = list(
            launcher.map(
                lambda index: harness.launch_task_record(
                    f"sandbox-{index}",
                    harness.select_task_record(
                        dataset,
                        sandbox_index=index,
                        default_agent_type=args.agent_type,
                        default_llm_service_type=args.llm_service_type,
                        default_task_description=benchmark_task_description(),
                        default_task_config=default_task_config(),
                    ),
                ),
                range(args.sandboxes),
            )
        )
    if sandboxes and any(is_replay_llm_service_type(sandbox.llm_service_type) for sandbox in sandboxes):
        with ThreadPoolExecutor(max_workers=max(1, args.sandboxes)) as executor:
            rows = list(
                executor.map(
                    lambda sandbox: _run_replay_accuracy_sandbox(args, harness, sandbox),
                    sandboxes,
                )
            )
        return sorted(rows, key=lambda row: str(row["sandbox_id"]))
    rows: list[dict[str, object]] = []

    for iteration in range(args.iters):
        checkpoint_results = []
        for sandbox in sandboxes:
            assert sandbox.task_run is not None
            sandbox.task_run.wait_for_progress(minimum_actions=6)
            if harness.request_state_store is not None and not wait_for(
                lambda sid=sandbox.sandbox_id: harness.request_state_store.get(sid).llm_request_in_flight,
                timeout_s=20.0,
                raise_on_timeout=False,
            ):
                logger.warning(
                    "sandbox %s did not enter an in-flight LLM request window before checkpoint; continuing",
                    sandbox.sandbox_id,
                )
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
                "iter": iteration,
                "provider": args.provider,
                "agent_type": args.agent_type,
                "sandboxes": args.sandboxes,
                "tool_actions": sum(int(sandbox.last_status["total_actions"]) for sandbox in sandboxes),
                "fs_actions": sum(int(sandbox.last_status["filesystem_actions"]) for sandbox in sandboxes),
                "process_actions": sum(int(sandbox.last_status["process_actions"]) for sandbox in sandboxes),
                "network_actions": sum(int(sandbox.last_status["network_actions"]) for sandbox in sandboxes),
                "checkpoints": len(checkpoint_results),
                "restores": len(restore_results),
                "checkpoint_batch_ms": (t1 - t0) * 1000.0,
                "restore_batch_ms": (t2 - t1) * 1000.0,
                "success_ratio": (
                    sum(1 for item in restore_results if item.status.value == "succeeded")
                    / max(1, len(restore_results))
                ),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)
    scheduler_config = SchedulerConfig(
        min_checkpoint_interval_seconds=0.0,
        force_checkpoint_after_seconds=0.0,
        require_change_signal=True,
        prefer_checkpoint_during_llm_request=True,
        require_llm_request_for_checkpoint=False,
    )
    with RealHostScenarioHarness(
        provider=args.provider,
        transfer_delay_ms=args.transfer_delay_ms,
        scheduler_config=scheduler_config,
        scheduler_policy=None,
        checkpoint_manager_factory=lambda base: base,
        max_workers=args.sandboxes,
        work_dir_host_root=args.work_dir_host_root,
    ) as harness:
        rows = run_benchmark(args, harness)
    write_rows(args.out, rows)
    metric_keys = ["checkpoint_batch_ms", "restore_batch_ms", "success_ratio"]
    if rows and "verification_status" in rows[0]:
        metric_keys = ["task_completion_ms", "verification_ms", "success_ratio"]
    summary = compute_summary(rows, metric_keys)
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")


if __name__ == "__main__":
    main()

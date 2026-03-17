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
from benchmarks.support import add_common_args, compute_summary, configure_logging, wait_for, write_rows
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
    return TaskConfig(minimum_actions=0)


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
    summary = compute_summary(rows, ["checkpoint_batch_ms", "restore_batch_ms", "success_ratio"])
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")


if __name__ == "__main__":
    main()

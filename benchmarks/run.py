#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from benchmarks.config import BenchmarkConfig, load_config
from benchmarks.real_host_scenario_base import RealHostScenarioHarness
from benchmarks.scenarios.e2e import SCENARIO as E2E_SCENARIO
from benchmarks.scenarios.fault import SCENARIO as FAULT_SCENARIO
from benchmarks.scenarios.spot import SCENARIO as SPOT_SCENARIO
from benchmarks.scenarios.tree import SCENARIO as TREE_SCENARIO
from benchmarks.support import (
    benchmark_run_context,
    benchmark_run_duration_seconds,
    compute_telemetry_summary,
    configure_logging,
    write_rows,
)


SCENARIOS = {
    E2E_SCENARIO.name: E2E_SCENARIO,
    FAULT_SCENARIO.name: FAULT_SCENARIO,
    SPOT_SCENARIO.name: SPOT_SCENARIO,
    TREE_SCENARIO.name: TREE_SCENARIO,
}

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Agent-CR benchmark runner")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _summary_lines(summary: dict[str, float]) -> list[str]:
    return [f"{key}_avg: {value:.3f}" for key, value in summary.items()]


def _telemetry_metric_aliases(config: BenchmarkConfig, rows: list[dict[str, object]]) -> tuple[dict[str, str | tuple[str, ...]], dict[str, dict[str, object]]]:
    aliases: dict[str, str | tuple[str, ...]] = {}
    attribute_filters: dict[str, dict[str, object]] = {}
    if config.scenario == "fault":
        aliases.update(
            {
                "checkpoint_ms": "benchmark.checkpoint_ms",
                "restore_ms": "benchmark.restore_ms",
                "recovery_ms": "benchmark.recovery_ms",
                "readiness_ms": "benchmark.readiness_ms",
                "end_to_end_recovery_ms": "benchmark.end_to_end_recovery_ms",
                "workload_resume_ms": "benchmark.workload_resume_ms",
            }
        )
        if config.mode == "auto" and not (rows and ("verification_status" in rows[0] or "chunks_planned" in rows[0] or "iterations_planned" in rows[0])):
            for key in ("checkpoint_ms", "restore_ms", "recovery_ms", "readiness_ms", "end_to_end_recovery_ms", "workload_resume_ms"):
                attribute_filters[key] = {"event_injected": 1}
    elif config.scenario == "spot":
        aliases.update(
            {
                "checkpoint_ms": "benchmark.checkpoint_ms",
                "restore_ms": "benchmark.restore_ms",
                "recovery_ms": "benchmark.recovery_ms",
                "readiness_ms": "benchmark.readiness_ms",
                "end_to_end_recovery_ms": "benchmark.end_to_end_recovery_ms",
                "migration_ms": "benchmark.migration_ms",
                "budget_slack_ms": "benchmark.budget_slack_ms",
            }
        )
        if config.mode == "auto" and not (rows and ("verification_status" in rows[0] or "chunks_planned" in rows[0] or "iterations_planned" in rows[0])):
            for key in aliases:
                attribute_filters[key] = {"event_injected": 1}
    elif config.scenario == "e2e":
        if rows and "verification_status" in rows[0]:
            aliases.update(
                {
                    "task_completion_ms": "benchmark.task.duration_ms",
                    "verification_ms": "benchmark.task.verify.duration_ms",
                }
            )
        else:
            aliases.update(
                {
                    "checkpoint_batch_ms": "benchmark.checkpoint_batch_ms",
                    "restore_batch_ms": "benchmark.restore_batch_ms",
                }
            )
    elif config.scenario == "tree":
        aliases.update(
            {
                "checkpoint_ms": "benchmark.checkpoint_ms",
                "restore_ms": "benchmark.restore_ms",
                "recovery_ms": "benchmark.recovery_ms",
                "readiness_ms": "benchmark.readiness_ms",
                "end_to_end_recovery_ms": "benchmark.end_to_end_recovery_ms",
                "replay_progress_ms": "benchmark.replay_progress_ms",
                "fanout_ms": "benchmark.fanout_ms",
            }
        )
    return aliases, attribute_filters


def _artifact_lines(*, log_file: Path | None, output: Path | None, telemetry_output: Path | None) -> list[str]:
    lines: list[str] = []
    for label, path in (
        ("log_file", log_file),
        ("output", output),
        ("telemetry_output", telemetry_output),
    ):
        if path is not None:
            lines.append(f"{label}: {path.resolve()}")
    return lines


def _emit_lines(lines: list[str]) -> None:
    for line in lines:
        logger.info(line)
        print(line)


def run_benchmark_config(config: BenchmarkConfig) -> list[dict[str, object]]:
    scenario = SCENARIOS[config.scenario]
    settings = scenario.build_harness_settings(config)
    configure_logging(
        config.log_level,
        log_file=config.log_file,
        log_file_mode="a" if config.log_file_mode == "append" else "w",
    )
    run_context = benchmark_run_context(config.config_path)
    logger.info(
        "========== benchmark.run start config=%s scenario=%s mode=%s provider=%s agent=%s llm_service=%s sandboxes=%d max_workers=%d iterations=%d pid=%s ==========",
        config.config_path.resolve(),
        config.scenario,
        config.mode,
        config.provider,
        config.agent,
        config.llm_service or "",
        config.sandboxes,
        config.effective_max_workers,
        config.iterations,
        run_context["pid"],
    )
    telemetry_output = config.telemetry_output
    if telemetry_output is None:
        telemetry_output = (
            config.output.with_suffix(".telemetry.jsonl")
            if config.output is not None
            else config.config_path.with_suffix(".telemetry.jsonl")
        )
    try:
        with RealHostScenarioHarness(
            provider=config.provider,
            transfer_delay_ms=config.transfer_delay_ms,
            scheduler_config=settings.scheduler_config,
            scheduler_policy=settings.scheduler_policy,
            checkpoint_manager_factory=settings.checkpoint_manager_factory,
            max_workers=settings.max_workers,
            auto_cr=config.mode == "auto",
            work_dir_host_root=config.work_dir_host_root,
            telemetry_output=telemetry_output,
            telemetry_detail_level=config.telemetry_detail_level,
            telemetry_capture_command_output=config.telemetry_capture_command_output,
            telemetry_max_text_attribute_bytes=config.telemetry_max_text_attribute_bytes,
            benchmark_root=config.benchmark_root,
            zpool_size=config.zpool_size,
            zpool_name=config.zpool_name,
            zpool_image=config.zpool_image,
            reuse_zpool=config.reuse_zpool,
            image_cache_root=config.image_cache_root,
            run_id=str(run_context["run_id"]),
        ) as harness:
            if scenario.prepare_harness is not None:
                scenario.prepare_harness(config, harness)
            rows = scenario.runner_for_mode(config.mode)(config, harness)
        if config.output is not None:
            config.output.parent.mkdir(parents=True, exist_ok=True)
            write_rows(str(config.output), rows)
        summary = scenario.summarize(config, rows)
        telemetry_aliases, telemetry_filters = _telemetry_metric_aliases(config, rows)
        if telemetry_aliases:
            telemetry_summary = compute_telemetry_summary(
                telemetry_output,
                telemetry_aliases,
                run_id=str(run_context["run_id"]),
                attribute_filters=telemetry_filters,
            )
            if telemetry_summary:
                summary = {**summary, **telemetry_summary}
        _emit_lines(_summary_lines(summary))
        _emit_lines(
            _artifact_lines(
                log_file=config.log_file,
                output=config.output,
                telemetry_output=telemetry_output,
            )
        )
        logger.info(
            "========== benchmark.run end status=completed config=%s duration_s=%.3f ==========",
            config.config_path.resolve(),
            benchmark_run_duration_seconds(run_context),
        )
        return rows
    except Exception:
        logger.exception(
            "========== benchmark.run end status=failed config=%s duration_s=%.3f ==========",
            config.config_path.resolve(),
            benchmark_run_duration_seconds(run_context),
        )
        raise


def main() -> None:
    args = parse_args()
    run_benchmark_config(load_config(args.config))


if __name__ == "__main__":
    main()

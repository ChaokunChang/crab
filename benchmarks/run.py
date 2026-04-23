#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import shutil
import time

from benchmarks.config import BenchmarkConfig, load_config
from benchmarks.core import emit_benchmark_phase_progress
from benchmarks.real_host_scenario_base import RealHostScenarioHarness
from benchmarks.scenarios.e2e import SCENARIO as E2E_SCENARIO
from benchmarks.scenarios.fault import SCENARIO as FAULT_SCENARIO
from benchmarks.scenarios.spec import SCENARIO as SPEC_SCENARIO
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
    SPEC_SCENARIO.name: SPEC_SCENARIO,
    SPOT_SCENARIO.name: SPOT_SCENARIO,
    TREE_SCENARIO.name: TREE_SCENARIO,
}

logger = logging.getLogger(__name__)


def _default_report_output_dir(telemetry_output: Path) -> Path:
    return telemetry_output.with_suffix(".report")


def _prepare_telemetry_output_path(path: Path, *, file_mode: str) -> None:
    if file_mode != "write":
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)


def _clear_benchmark_root_if_needed(config: BenchmarkConfig, harness_context: object | None) -> None:
    if not config.clear_benchmark_root_after_run or harness_context is None:
        return
    if bool(getattr(harness_context, "uses_temporary_root", False)):
        return
    benchmark_run_root = getattr(harness_context, "root", None)
    if benchmark_run_root is None:
        return
    root_path = Path(benchmark_run_root).expanduser().resolve()
    if not root_path.exists():
        return
    logger.info("Clearing benchmark run root %s", root_path)
    try:
        shutil.rmtree(root_path)
    except Exception:
        logger.exception("Failed to clear benchmark run root %s", root_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Agent-CR benchmark runner")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def _summary_lines(summary: dict[str, float]) -> list[str]:
    return [f"{key}_avg: {value:.3f}" for key, value in summary.items()]


def _failed_sandbox_lines(rows: list[dict[str, object]]) -> list[str]:
    seen: set[str] = set()
    lines: list[str] = []
    for row in rows:
        sandbox_id = str(row.get("sandbox_id", ""))
        if sandbox_id in seen:
            continue
        task_error = str(row.get("task_error", ""))
        success = float(row.get("success_ratio", 1.0))
        if success >= 1.0 and not task_error:
            continue
        seen.add(sandbox_id)
        task_id = str(row.get("task_id", ""))
        lines.append(f"FAILED sandbox={sandbox_id} task={task_id} error={task_error}")
    if lines:
        lines.insert(0, f"--- Failed sandboxes ({len(lines)}) ---")
    return lines


def _telemetry_metric_aliases(config: BenchmarkConfig, rows: list[dict[str, object]]) -> tuple[dict[str, str | tuple[str, ...]], dict[str, dict[str, object]], set[str]]:
    """Map summary keys to telemetry metric names for post-run aggregation.

    The telemetry summary intentionally *overrides* the row-based summary
    (via ``{**summary, **telemetry_summary}``) so there is no double-counting.
    This allows worker-level metrics (e.g. ``checkpoint.total_ms``) from the
    telemetry stream to replace row-based placeholders when the scenario does
    not measure them directly (e.g. auto mode).  The tuple form means
    "try the first metric name, fall back to the second".
    """
    aliases: dict[str, str | tuple[str, ...]] = {}
    attribute_filters: dict[str, dict[str, object]] = {}
    last_value_keys: set[str] = set()
    if config.scenario == "fault":
        aliases.update(
            {
                "checkpoint_ms": ("checkpoint.total_ms", "benchmark.checkpoint_ms"),
                "restore_ms": ("restore.total_ms", "benchmark.restore_ms"),
                "recovery_ms": "benchmark.recovery_ms",
                "readiness_ms": "benchmark.readiness_ms",
                "end_to_end_recovery_ms": "benchmark.end_to_end_recovery_ms",
                "workload_resume_ms": "benchmark.workload_resume_ms",
            }
        )
        if config.mode == "auto" and not (rows and ("verification_status" in rows[0] or "chunks_planned" in rows[0] or "iterations_planned" in rows[0])):
            for key in ("recovery_ms", "readiness_ms", "end_to_end_recovery_ms", "workload_resume_ms"):
                attribute_filters[key] = {"event_injected": 1}
    elif config.scenario == "spot":
        aliases.update(
            {
                "checkpoint_ms": ("checkpoint.total_ms", "benchmark.checkpoint_ms"),
                "restore_ms": ("restore.total_ms", "benchmark.restore_ms"),
                "recovery_ms": "benchmark.recovery_ms",
                "readiness_ms": "benchmark.readiness_ms",
                "end_to_end_recovery_ms": "benchmark.end_to_end_recovery_ms",
                "migration_ms": "benchmark.migration_ms",
                "budget_slack_ms": "benchmark.budget_slack_ms",
            }
        )
        if config.mode == "auto" and not (rows and ("verification_status" in rows[0] or "chunks_planned" in rows[0] or "iterations_planned" in rows[0])):
            for key in ("recovery_ms", "readiness_ms", "end_to_end_recovery_ms", "migration_ms", "budget_slack_ms"):
                attribute_filters[key] = {"event_injected": 1}
    elif config.scenario == "e2e":
        if rows and "task_completion_ms" in rows[0]:
            aliases["task_completion_ms"] = "benchmark.task.duration_ms"
            if "verification_status" in rows[0]:
                aliases["verification_ms"] = "benchmark.task.verify.duration_ms"
        else:
            aliases.update(
                {
                    "checkpoint_batch_ms": "benchmark.checkpoint_batch_ms",
                    "restore_batch_ms": "benchmark.restore_batch_ms",
                }
            )
    elif config.scenario == "spec":
        aliases.update(
            {
                "task_completion_ms": "benchmark.task.duration_ms",
                "verification_ms": "benchmark.task.verify.duration_ms",
                "spec_saved_ms": "benchmark.spec.saved_ms",
                "spec_penalty_ms": "benchmark.spec.penalty_ms",
                "spec_hidden_penalty_ms": "benchmark.spec.hidden_penalty_ms",
                "spec_net_gain_ms": "benchmark.spec.net_gain_ms",
                "spec_accept_rate": "benchmark.spec.accept_rate",
            }
        )
        last_value_keys.add("spec_accept_rate")
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
    return aliases, attribute_filters, last_value_keys


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


def _flush_logging_handlers() -> None:
    for handler in logging.getLogger().handlers:
        handler.flush()


def _copy_artifact_replica(source: Path, benchmark_run_root: Path) -> Path | None:
    source_path = source.expanduser().resolve()
    if not source_path.exists():
        return None
    replica_path = (benchmark_run_root / source_path.name).expanduser().resolve()
    if source_path == replica_path:
        return replica_path
    replica_path.parent.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        if replica_path.exists():
            if replica_path.is_dir():
                shutil.rmtree(replica_path)
            else:
                replica_path.unlink()
        shutil.copytree(source_path, replica_path)
    else:
        if replica_path.exists() and replica_path.is_dir():
            shutil.rmtree(replica_path)
        shutil.copy2(source_path, replica_path)
    return replica_path


def _replicate_artifacts_to_benchmark_run_root(
    *,
    harness_context: object | None,
    log_file: Path | None,
    output: Path | None,
    telemetry_output: Path | None,
    telemetry_report_output_dir: Path | None,
) -> None:
    if harness_context is None or bool(getattr(harness_context, "uses_temporary_root", False)):
        return
    root = getattr(harness_context, "root", None)
    if root is None:
        return
    benchmark_run_root = Path(root).expanduser().resolve()
    _flush_logging_handlers()
    for label, path in (
        ("log_file", log_file),
        ("output", output),
        ("telemetry_output", telemetry_output),
        ("telemetry_report", telemetry_report_output_dir),
    ):
        if path is None:
            continue
        try:
            _copy_artifact_replica(Path(path), benchmark_run_root)
        except Exception:
            logger.exception(
                "Failed to replicate benchmark artifact label=%s source=%s benchmark_run_root=%s",
                label,
                path,
                benchmark_run_root,
            )
            continue
    logger.info("artificats replicated to %s", benchmark_run_root)
    print("artificats replicated to %s", benchmark_run_root, flush=True)

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
        "========== benchmark.run start config=%s scenario=%s mode=%s provider=%s agent=%s llm_service=%s sandboxes=%d max_workers=%d phase_workers=%s iterations=%d pid=%s ==========",
        config.config_path.resolve(),
        config.scenario,
        config.mode,
        config.provider,
        config.agent,
        config.llm_service or "",
        config.sandboxes,
        config.effective_max_workers,
        config.effective_phase_workers.as_dict(),
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
    _prepare_telemetry_output_path(telemetry_output, file_mode=config.telemetry_file_mode)
    executor_default_workers = max(settings.max_workers, config.effective_phase_workers.run)
    executor_config = config.executor.resolve(default_workers=executor_default_workers)
    harness_context: object | None = None
    report_dir: Path | None = None
    try:
        teardown_started_at: float | None = None
        harness_context = RealHostScenarioHarness(
            provider=config.provider,
            transfer_delay_ms=config.transfer_delay_ms,
            scheduler_config=settings.scheduler_config,
            scheduler_policy=settings.scheduler_policy,
            checkpoint_manager_factory=settings.checkpoint_manager_factory,
            executor_config=executor_config,
            max_workers=executor_default_workers,
            auto_cr=config.mode == "auto",
            work_dir_host_root=config.work_dir_host_root,
            telemetry_output=telemetry_output,
            telemetry_detail_level=config.telemetry_detail_level,
            telemetry_capture_command_output=config.telemetry_capture_command_output,
            telemetry_max_text_attribute_bytes=config.telemetry_max_text_attribute_bytes,
            telemetry_keep_in_memory_copy=config.telemetry_keep_in_memory_copy,
            telemetry_writer_mode=config.telemetry_writer_mode,
            telemetry_queue_capacity=config.telemetry_queue_capacity,
            telemetry_batch_max_records=config.telemetry_batch_max_records,
            telemetry_flush_interval_ms=config.telemetry_flush_interval_ms,
            telemetry_overflow_policy=config.telemetry_overflow_policy,
            telemetry_serializer=config.telemetry_serializer,
            benchmark_root_home=config.benchmark_root_home,
            benchmark_run_name=config.benchmark_run_name,
            zpool_size=config.zpool_size,
            zpool_name=config.zpool_name,
            zpool_image=config.zpool_image,
            reuse_zpool=config.reuse_zpool,
            runtime_command_timeout_seconds=config.runtime_command_timeout_seconds,
            runtime_zfs_prepare_timeout_seconds=config.runtime_zfs_prepare_timeout_seconds,
            image_cache_root=config.image_cache_root,
            run_id=str(run_context["run_id"]),
            monitoring_enabled=config.monitoring.enabled,
            monitoring_sample_interval_ms=config.monitoring.sample_interval_ms,
            monitoring_include_host=config.monitoring.include_host,
            monitoring_include_sandboxes=config.monitoring.include_sandboxes,
            llm_server_launch_mode=config.llm_server.launch_mode,
            host_inspector_launch_mode=config.host_inspector.launch_mode,
            host_inspector_log_level=config.host_inspector.log_level,
            host_inspector_log_file=config.host_inspector.log_file,
            runtime_root=config.storage_planes.runtime_root,
            storage_root=config.storage_planes.storage_root,
            agent_host_root=config.storage_planes.agent_host_root,
            expected_sandboxes=settings.expected_sandboxes or config.sandboxes,
            rootfs_reuse_enabled=config.rootfs_reuse.enabled,
            sandbox_resource_limits=config.sandbox_resource_limits.to_runtime_limits(),
            fork_reuse_enabled=bool(config.scenario_options.get("enable_fork_reuse", False)),
            eager_fork_cleanup_on_reject=bool(
                config.scenario_options.get("eager_fork_cleanup_on_reject", False)
            ),
        )
        with harness_context as harness:
            if scenario.prepare_harness is not None:
                scenario.prepare_harness(config, harness)
            rows = scenario.runner_for_mode(config.mode)(config, harness)
            teardown_started_at = time.perf_counter()
            emit_benchmark_phase_progress(
                phase="teardown",
                status="start",
                sandbox_count=config.sandboxes,
                configured_max_workers=1,
            )
        if teardown_started_at is not None:
            emit_benchmark_phase_progress(
                phase="teardown",
                status="end",
                sandbox_count=config.sandboxes,
                configured_max_workers=1,
                duration_seconds=max(0.0, time.perf_counter() - teardown_started_at),
            )
        postprocess_started_at = time.perf_counter()
        emit_benchmark_phase_progress(
            phase="postprocess",
            status="start",
            sandbox_count=config.sandboxes,
            configured_max_workers=1,
        )
        postprocess_completed = False
        try:
            if config.output is not None:
                config.output.parent.mkdir(parents=True, exist_ok=True)
                write_rows(str(config.output), rows)
            summary = scenario.summarize(config, rows)
            telemetry_aliases, telemetry_filters, telemetry_last_value_keys = _telemetry_metric_aliases(config, rows)
            if telemetry_aliases:
                telemetry_summary = compute_telemetry_summary(
                    telemetry_output,
                    telemetry_aliases,
                    run_id=str(run_context["run_id"]),
                    attribute_filters=telemetry_filters,
                    last_value_keys=telemetry_last_value_keys,
                )
                if telemetry_summary:
                    summary = {**summary, **telemetry_summary}
            if config.telemetry_report.enabled and telemetry_output.exists():
                report_dir = config.telemetry_report.output_dir or _default_report_output_dir(telemetry_output)
                try:
                    from benchmarks.telemetry_analysis import generate_report_bundle

                    generate_report_bundle(
                        telemetry_output,
                        output_dir=report_dir,
                        top_k=max(5, int(config.telemetry_report.top_k)),
                        log_scale=config.telemetry_report.log_scale_charts,
                        export_svg=config.telemetry_report.export_svg,
                    )
                except Exception:
                    logger.exception(
                        "Telemetry report generation failed for telemetry_output=%s output_dir=%s",
                        telemetry_output,
                        report_dir,
                    )
            _emit_lines(_summary_lines(summary))
            failed_lines = _failed_sandbox_lines(rows)
            if failed_lines:
                _emit_lines(failed_lines)
            artifact_lines = _artifact_lines(
                log_file=config.log_file,
                output=config.output,
                telemetry_output=telemetry_output,
            )
            if report_dir is not None:
                artifact_lines.append(f"telemetry_report: {report_dir.resolve()}")
            _emit_lines(artifact_lines)
            postprocess_completed = True
        finally:
            emit_benchmark_phase_progress(
                phase="postprocess",
                status="end" if postprocess_completed else "failed",
                sandbox_count=config.sandboxes,
                configured_max_workers=1,
                duration_seconds=max(0.0, time.perf_counter() - postprocess_started_at),
            )
        logger.info(
            "========== benchmark.run end status=completed config=%s duration_s=%.3f ==========",
            config.config_path.resolve(),
            benchmark_run_duration_seconds(run_context),
        )
        _replicate_artifacts_to_benchmark_run_root(
            harness_context=harness_context,
            log_file=config.log_file,
            output=config.output,
            telemetry_output=telemetry_output,
            telemetry_report_output_dir=report_dir,
        )
        return rows
    except Exception:
        logger.exception(
            "========== benchmark.run end status=failed config=%s duration_s=%.3f ==========",
            config.config_path.resolve(),
            benchmark_run_duration_seconds(run_context),
        )
        _replicate_artifacts_to_benchmark_run_root(
            harness_context=harness_context,
            log_file=config.log_file,
            output=config.output,
            telemetry_output=telemetry_output,
            telemetry_report_output_dir=report_dir,
        )
        raise
    finally:
        _clear_benchmark_root_if_needed(config, harness_context)


def main() -> None:
    args = parse_args()
    run_benchmark_config(load_config(args.config))


if __name__ == "__main__":
    main()

from __future__ import annotations

import dataclasses

from crab import FaultToleranceCheckpointingPolicy, LatestOnlyCheckpointManager, SchedulerConfig

from benchmarks.config import BenchmarkConfig
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from benchmarks.scenarios.common import resolve_scheduler_policy_override
from benchmarks.scenarios.e2e import (
    _finalize_replay_accuracy_sandbox,
    _run_replay_accuracy_sandbox_run,
    benchmark_task_description,
    default_task_config,
)
from benchmarks.core import (
    benchmark_phase_item_attributes,
    benchmark_phase_map,
    benchmark_setup_run_pipeline,
    emit_benchmark_phase_skipped,
    make_benchmark_sandbox_specs,
    resolve_task_records,
    setup_task_records_phase,
    start_prepared_task_record,
)
from benchmarks.support import compute_summary


def _apply_spec_scenario_options(records, config: BenchmarkConfig):
    from benchmarks.core import _apply_llm_service_options  # local import to avoid broad module churn

    normalized = [
        dataclasses.replace(
            record,
            agent_type=config.agent,
            llm_service_type=config.llm_service,
        )
        for record in records
    ]
    if not config.scenario_options:
        return normalized
    return [_apply_llm_service_options(record, config.scenario_options) for record in normalized]


def build_harness_settings(config: BenchmarkConfig) -> HarnessSettings:
    scheduler_config = config.scheduler.apply(
        SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
            prefer_checkpoint_during_llm_request=True,
            require_llm_request_for_checkpoint=False,
        )
    )
    return HarnessSettings(
        scheduler_config=scheduler_config,
        scheduler_policy=resolve_scheduler_policy_override(
            config,
            scenario_default_policy=FaultToleranceCheckpointingPolicy(scheduler_config),
        ),
        checkpoint_manager_factory=lambda base: LatestOnlyCheckpointManager(
            base,
            delete_filesystem_checkpoints=True,
        ),
        max_workers=config.effective_max_workers,
        # Speculative execution can keep one fork live next to each active
        # sandbox, so size shared resources for the peak live sandbox count.
        expected_sandboxes=config.sandboxes * 2,
    )


_SPEC_AGENT_TO_LLM_SERVICE = {
    "mini_swe": "mini_swe_spec_trace_replay",
    "terminus": "terminus_spec_trace_replay",
}


def prepare_harness(config: BenchmarkConfig, harness) -> None:
    del harness
    if config.agent not in _SPEC_AGENT_TO_LLM_SERVICE:
        raise ValueError(
            f"scenario='spec' requires agent in {sorted(_SPEC_AGENT_TO_LLM_SERVICE)}, got {config.agent!r}"
        )
    expected_llm = _SPEC_AGENT_TO_LLM_SERVICE[config.agent]
    if config.llm_service != expected_llm:
        raise ValueError(
            f"scenario='spec' with agent='{config.agent}' requires llm_service='{expected_llm}'"
        )
    if config.iterations != 0:
        raise ValueError("scenario='spec' requires iterations=0")


def _finalize_spec_replay_sandbox(run_result: dict[str, object]) -> dict[str, object]:
    row = _finalize_replay_accuracy_sandbox(run_result)
    status = dict(run_result["status"])
    row.update(
        {
            "spec_total_turns": int(status.get("spec_total_turns", row.get("tool_actions", 0))),
            "spec_accept_count": int(status.get("spec_accept_count", 0)),
            "spec_reject_count": int(status.get("spec_reject_count", 0)),
            "spec_accept_rate": float(status.get("spec_accept_rate", 0.0)),
            "spec_saved_ms": float(status.get("spec_saved_ms", 0.0)),
            "spec_penalty_ms": float(status.get("spec_penalty_ms", 0.0)),
            "spec_hidden_penalty_ms": float(status.get("spec_hidden_penalty_ms", 0.0)),
            "spec_net_gain_ms": float(
                status.get(
                    "spec_net_gain_ms",
                    float(status.get("spec_saved_ms", 0.0)) - float(status.get("spec_penalty_ms", 0.0)),
                )
            ),
            "spec_fork_create_count": int(status.get("spec_fork_create_count", 0)),
            "spec_fork_reuse_count": int(status.get("spec_fork_reuse_count", 0)),
        }
    )
    return row


def run_manual(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    records = resolve_task_records(
        config,
        default_task_description=benchmark_task_description(),
        default_task_config=default_task_config(),
    )
    records = _apply_spec_scenario_options(records, config)
    specs = make_benchmark_sandbox_specs(
        sandbox_name_prefix="spec",
        records=records,
    )
    if config.phase_merging.setup_and_run:
        indexed_specs = list(enumerate(specs))
        run_results = benchmark_setup_run_pipeline(
            indexed_specs,
            setup_fn=lambda item: harness.setup_task_record(item[1].sandbox_name, item[1].task_record),
            run_fn=lambda _item, prepared: _run_replay_accuracy_sandbox_run(
                config,
                harness,
                start_prepared_task_record(harness, prepared),
            ),
            setup_max_workers=config.effective_phase_workers.setup,
            run_max_workers=config.effective_phase_workers.run,
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
        prepared = setup_task_records_phase(
            harness,
            specs=specs,
            max_workers=config.effective_phase_workers.setup,
        )
        run_results = benchmark_phase_map(
            prepared,
            lambda item: _run_replay_accuracy_sandbox_run(
                config,
                harness,
                start_prepared_task_record(harness, item),
            ),
            phase="run",
            max_workers=config.effective_phase_workers.run,
            harness=harness,
            item_attributes=lambda item: benchmark_phase_item_attributes(
                harness,
                phase="run",
                sandbox_name=item.sandbox_name,
                sandbox=item.handle,
            ),
        )
    if config.verification_enabled:
        rows = benchmark_phase_map(
            run_results,
            _finalize_spec_replay_sandbox,
            phase="verification",
            max_workers=config.effective_phase_workers.verification,
            harness=harness,
            item_attributes=lambda item: benchmark_phase_item_attributes(
                harness,
                phase="verification",
                sandbox_name=str(item["sandbox"].sandbox_id),
                sandbox=item["sandbox"],
            ),
        )
    else:
        emit_benchmark_phase_skipped(
            phase="verification",
            sandbox_count=len(run_results),
            configured_max_workers=config.effective_phase_workers.verification,
        )
        rows = [_finalize_spec_replay_sandbox(item) for item in run_results]
    return sorted(rows, key=lambda row: str(row["sandbox_id"]))


def run_auto(config: BenchmarkConfig, harness) -> list[dict[str, object]]:
    return run_manual(config, harness)


def summary_metrics(config: BenchmarkConfig, rows: list[dict[str, object]]) -> dict[str, float]:
    del config
    metric_keys = [
        "task_completion_ms",
        "success_ratio",
        "spec_accept_rate",
        "spec_saved_ms",
        "spec_penalty_ms",
        "spec_hidden_penalty_ms",
        "spec_net_gain_ms",
    ]
    if rows and "verification_status" in rows[0]:
        metric_keys.insert(1, "verification_ms")
    return compute_summary(rows, metric_keys)


SCENARIO = ScenarioDefinition(
    name="spec",
    supported_modes=frozenset({"manual", "auto"}),
    build_harness_settings=build_harness_settings,
    run_manual=run_manual,
    run_auto=run_auto,
    summarize=summary_metrics,
    prepare_harness=prepare_harness,
)

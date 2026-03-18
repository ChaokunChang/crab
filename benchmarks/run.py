#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.config import BenchmarkConfig, load_config
from benchmarks.real_host_scenario_base import RealHostScenarioHarness
from benchmarks.scenarios.e2e import SCENARIO as E2E_SCENARIO
from benchmarks.scenarios.fault import SCENARIO as FAULT_SCENARIO
from benchmarks.scenarios.spot import SCENARIO as SPOT_SCENARIO
from benchmarks.scenarios.tree import SCENARIO as TREE_SCENARIO
from benchmarks.support import configure_logging, write_rows


SCENARIOS = {
    E2E_SCENARIO.name: E2E_SCENARIO,
    FAULT_SCENARIO.name: FAULT_SCENARIO,
    SPOT_SCENARIO.name: SPOT_SCENARIO,
    TREE_SCENARIO.name: TREE_SCENARIO,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified Agent-CR benchmark runner")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def run_benchmark_config(config: BenchmarkConfig) -> list[dict[str, object]]:
    scenario = SCENARIOS[config.scenario]
    settings = scenario.build_harness_settings(config)
    configure_logging(config.log_level)
    telemetry_output = config.telemetry_output
    if telemetry_output is None:
        telemetry_output = (
            config.output.with_suffix(".telemetry.jsonl")
            if config.output is not None
            else config.config_path.with_suffix(".telemetry.jsonl")
        )
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
        image_cache_root=config.image_cache_root,
    ) as harness:
        if scenario.prepare_harness is not None:
            scenario.prepare_harness(config, harness)
        rows = scenario.runner_for_mode(config.mode)(config, harness)
    if config.output is not None:
        config.output.parent.mkdir(parents=True, exist_ok=True)
        write_rows(str(config.output), rows)
    summary = scenario.summarize(config, rows)
    for key, value in summary.items():
        print(f"{key}_avg: {value:.3f}")
    return rows


def main() -> None:
    args = parse_args()
    run_benchmark_config(load_config(args.config))


if __name__ == "__main__":
    main()

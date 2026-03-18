from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from benchmarks.config import BenchmarkConfig


@dataclass(frozen=True)
class HarnessSettings:
    scheduler_config: object
    scheduler_policy: object
    checkpoint_manager_factory: Callable[[object], object]
    max_workers: int


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    supported_modes: frozenset[str]
    build_harness_settings: Callable[[BenchmarkConfig], HarnessSettings]
    run_manual: Callable[[BenchmarkConfig, object], list[dict[str, object]]]
    run_auto: Callable[[BenchmarkConfig, object], list[dict[str, object]]] | None
    summarize: Callable[[BenchmarkConfig, list[dict[str, object]]], dict[str, float]]
    prepare_harness: Callable[[BenchmarkConfig, object], None] | None = None

    def runner_for_mode(self, mode: str) -> Callable[[BenchmarkConfig, object], list[dict[str, object]]]:
        if mode == "manual":
            return self.run_manual
        if mode == "auto" and self.run_auto is not None:
            return self.run_auto
        raise ValueError(f"scenario={self.name!r} does not support mode={mode!r}")

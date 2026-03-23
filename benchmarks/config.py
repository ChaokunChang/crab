from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent_cr.telemetry import (
    DEFAULT_TELEMETRY_CAPTURE_COMMAND_OUTPUT,
    DEFAULT_TELEMETRY_DETAIL_LEVEL,
    DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES,
)


_SCENARIOS = {"e2e", "fault", "spot", "tree"}
_MODES = {"manual", "auto"}
_PROVIDERS = {"openai", "anthropic"}
_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}
_LOG_FILE_MODES = {"append": "a", "write": "w"}
_TELEMETRY_DETAIL_LEVELS = {"basic", "detailed"}
_BENCHMARK_PHASES = ("build", "prepare", "run", "verification")
_SUPPORTED_MODES = {
    "e2e": {"manual"},
    "fault": {"manual", "auto"},
    "spot": {"manual", "auto"},
    "tree": {"manual", "auto"},
}
_DEFAULT_ITERATIONS = {
    "e2e": 5,
    "fault": 3,
    "spot": 3,
    "tree": 1,
}


@dataclass(frozen=True)
class BenchmarkPhaseWorkers:
    build: int | None = None
    prepare: int | None = None
    run: int | None = None
    verification: int | None = None

    def resolve(self, *, default: int) -> "ResolvedBenchmarkPhaseWorkers":
        return ResolvedBenchmarkPhaseWorkers(
            build=default if self.build is None else int(self.build),
            prepare=default if self.prepare is None else int(self.prepare),
            run=default if self.run is None else int(self.run),
            verification=default if self.verification is None else int(self.verification),
        )


@dataclass(frozen=True)
class ResolvedBenchmarkPhaseWorkers:
    build: int
    prepare: int
    run: int
    verification: int

    def for_phase(self, phase: str) -> int:
        if phase not in _BENCHMARK_PHASES:
            raise KeyError(f"unknown benchmark phase {phase!r}")
        return int(getattr(self, phase))

    def as_dict(self) -> dict[str, int]:
        return {phase: self.for_phase(phase) for phase in _BENCHMARK_PHASES}


@dataclass(frozen=True)
class BenchmarkConfig:
    config_path: Path
    scenario: str
    mode: str
    provider: str = "openai"
    agent: str = "simulated"
    llm_service: str | None = None
    task_dataset: Path | None = None
    sandboxes: int = 1
    max_workers: int | None = None
    phase_workers: BenchmarkPhaseWorkers | None = None
    iterations: int = 1
    output: Path | None = None
    telemetry_output: Path | None = None
    telemetry_detail_level: str = DEFAULT_TELEMETRY_DETAIL_LEVEL
    telemetry_capture_command_output: bool = DEFAULT_TELEMETRY_CAPTURE_COMMAND_OUTPUT
    telemetry_max_text_attribute_bytes: int = DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES
    log_file: Path | None = None
    log_file_mode: str = "append"
    benchmark_root: Path | None = None
    zpool_size: str = "10G"
    zpool_name: str | None = None
    zpool_image: Path | None = None
    reuse_zpool: bool = False
    image_cache_root: Path | None = None
    log_level: str = "info"
    transfer_delay_ms: float = 0.0
    work_dir_host_root: Path | None = None
    scenario_options: dict[str, object] = field(default_factory=dict)
    llm_service_options: dict[str, object] = field(default_factory=dict)

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent

    @property
    def effective_max_workers(self) -> int:
        configured = self.max_workers if self.max_workers is not None else self.sandboxes
        return max(1, min(self.sandboxes, configured))

    @property
    def effective_phase_workers(self) -> ResolvedBenchmarkPhaseWorkers:
        overrides = self.phase_workers or BenchmarkPhaseWorkers()
        return overrides.resolve(default=self.effective_max_workers)


def _resolve_optional_path(base_dir: Path, raw_value: object) -> Path | None:
    if raw_value is None:
        return None
    path = Path(str(raw_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return path


def _require_object(payload: object, *, label: str) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(payload)


def _load_phase_workers(payload: object) -> BenchmarkPhaseWorkers | None:
    if payload is None:
        return None
    data = _require_object(payload, label="phase_workers")
    unknown = sorted(set(data) - set(_BENCHMARK_PHASES))
    if unknown:
        raise ValueError(
            f"phase_workers contains unknown phases {unknown}; expected only {list(_BENCHMARK_PHASES)}"
        )
    values: dict[str, int | None] = {}
    for phase in _BENCHMARK_PHASES:
        raw_value = data.get(phase)
        if raw_value is None:
            values[phase] = None
            continue
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"phase_workers.{phase} must be positive, got {value}")
        values[phase] = value
    return BenchmarkPhaseWorkers(**values)


def load_config(path: Path) -> BenchmarkConfig:
    config_path = path.expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data = _require_object(payload, label=f"benchmark config {config_path}")

    scenario = str(data.get("scenario", "")).strip()
    if scenario not in _SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(_SCENARIOS)}, got {scenario!r}")

    mode = str(data.get("mode", "")).strip()
    if mode not in _MODES:
        raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
    if mode not in _SUPPORTED_MODES[scenario]:
        raise ValueError(f"scenario={scenario!r} does not support mode={mode!r}")

    provider = str(data.get("provider", "openai")).strip()
    if provider not in _PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(_PROVIDERS)}, got {provider!r}")

    log_level = str(data.get("log_level", "info")).strip().lower()
    if log_level not in _LOG_LEVELS:
        raise ValueError(f"log_level must be one of {sorted(_LOG_LEVELS)}, got {log_level!r}")

    log_file_mode = str(data.get("log_file_mode", "append")).strip().lower()
    if log_file_mode not in _LOG_FILE_MODES:
        raise ValueError(f"log_file_mode must be one of {sorted(_LOG_FILE_MODES)}, got {log_file_mode!r}")

    sandboxes = int(data.get("sandboxes", 1))
    if sandboxes <= 0:
        raise ValueError(f"sandboxes must be positive, got {sandboxes}")
    raw_max_workers = data.get("max_workers")
    max_workers = None if raw_max_workers is None else int(raw_max_workers)
    if max_workers is not None and max_workers <= 0:
        raise ValueError(f"max_workers must be positive when provided, got {max_workers}")
    phase_workers = _load_phase_workers(data.get("phase_workers"))

    iterations = int(data.get("iterations", _DEFAULT_ITERATIONS[scenario]))
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    scenario_options = data.get("scenario_options", {})
    if scenario_options is None:
        scenario_options = {}
    scenario_options = _require_object(scenario_options, label="scenario_options")

    llm_service_options = data.get("llm_service_options", {})
    if llm_service_options is None:
        llm_service_options = {}
    llm_service_options = _require_object(llm_service_options, label="llm_service_options")

    base_dir = config_path.parent
    telemetry_options = data.get("telemetry", {})
    if telemetry_options is None:
        telemetry_options = {}
    telemetry_options = _require_object(telemetry_options, label="telemetry")
    telemetry_output = _resolve_optional_path(
        base_dir,
        telemetry_options.get("output", data.get("telemetry_output")),
    )
    telemetry_detail_level = str(
        telemetry_options.get("detail_level", DEFAULT_TELEMETRY_DETAIL_LEVEL)
    ).strip().lower()
    if telemetry_detail_level not in _TELEMETRY_DETAIL_LEVELS:
        raise ValueError(
            f"telemetry.detail_level must be one of {sorted(_TELEMETRY_DETAIL_LEVELS)}, "
            f"got {telemetry_detail_level!r}"
        )
    telemetry_capture_command_output = bool(
        telemetry_options.get(
            "capture_command_output",
            DEFAULT_TELEMETRY_CAPTURE_COMMAND_OUTPUT,
        )
    )
    telemetry_max_text_attribute_bytes = int(
        telemetry_options.get(
            "max_text_attribute_bytes",
            DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES,
        )
    )
    if telemetry_max_text_attribute_bytes <= 0:
        raise ValueError("telemetry.max_text_attribute_bytes must be positive")

    return BenchmarkConfig(
        config_path=config_path,
        scenario=scenario,
        mode=mode,
        provider=provider,
        agent=str(data.get("agent", "simulated")),
        llm_service=None if data.get("llm_service") is None else str(data["llm_service"]),
        task_dataset=_resolve_optional_path(base_dir, data.get("task_dataset")),
        sandboxes=sandboxes,
        max_workers=max_workers,
        phase_workers=phase_workers,
        iterations=iterations,
        output=_resolve_optional_path(base_dir, data.get("output")),
        telemetry_output=telemetry_output,
        telemetry_detail_level=telemetry_detail_level,
        telemetry_capture_command_output=telemetry_capture_command_output,
        telemetry_max_text_attribute_bytes=telemetry_max_text_attribute_bytes,
        log_file=_resolve_optional_path(base_dir, data.get("log_file")),
        log_file_mode=log_file_mode,
        benchmark_root=_resolve_optional_path(base_dir, data.get("benchmark_root")),
        zpool_size=str(data.get("zpool_size", "10G")),
        zpool_name=None if data.get("zpool_name") is None else str(data.get("zpool_name")),
        zpool_image=_resolve_optional_path(base_dir, data.get("zpool_image")),
        reuse_zpool=bool(data.get("reuse_zpool", False)),
        image_cache_root=_resolve_optional_path(base_dir, data.get("image_cache_root")),
        log_level=log_level,
        transfer_delay_ms=float(data.get("transfer_delay_ms", 0.0)),
        work_dir_host_root=_resolve_optional_path(base_dir, data.get("work_dir_host_root")),
        scenario_options=scenario_options,
        llm_service_options=llm_service_options,
    )

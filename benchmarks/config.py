from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


_SCENARIOS = {"e2e", "fault", "spot", "tree"}
_MODES = {"manual", "auto"}
_PROVIDERS = {"openai", "anthropic"}
_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}
_LOG_FILE_MODES = {"append": "a", "write": "w"}
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
    iterations: int = 1
    output: Path | None = None
    telemetry_output: Path | None = None
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

    @property
    def config_dir(self) -> Path:
        return self.config_path.parent

    @property
    def effective_max_workers(self) -> int:
        configured = self.max_workers if self.max_workers is not None else self.sandboxes
        return max(1, min(self.sandboxes, configured))


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

    iterations = int(data.get("iterations", _DEFAULT_ITERATIONS[scenario]))
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")

    scenario_options = data.get("scenario_options", {})
    if scenario_options is None:
        scenario_options = {}
    scenario_options = _require_object(scenario_options, label="scenario_options")

    base_dir = config_path.parent
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
        iterations=iterations,
        output=_resolve_optional_path(base_dir, data.get("output")),
        telemetry_output=_resolve_optional_path(base_dir, data.get("telemetry_output")),
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
    )

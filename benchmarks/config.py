from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from agent_cr import ExecutorConfig, SchedulerConfig
from integrations.sandboxes.runtime.bundle import (
    DEFAULT_CPU_PERIOD_US,
    SandboxResourceLimits,
)
from agent_cr.telemetry import (
    DEFAULT_TELEMETRY_CAPTURE_COMMAND_OUTPUT,
    DEFAULT_TELEMETRY_BATCH_MAX_RECORDS,
    DEFAULT_TELEMETRY_DETAIL_LEVEL,
    DEFAULT_TELEMETRY_FLUSH_INTERVAL_MS,
    DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES,
    DEFAULT_TELEMETRY_OVERFLOW_POLICY,
    DEFAULT_TELEMETRY_QUEUE_CAPACITY,
    DEFAULT_TELEMETRY_SERIALIZER,
    DEFAULT_TELEMETRY_WRITER_MODE,
)


_SCENARIOS = {"e2e", "fault", "spot", "tree", "spec"}
_MODES = {"manual", "auto"}
_PROVIDERS = {"openai", "anthropic"}
_LOG_LEVELS = {"debug", "info", "warning", "error", "critical"}
_FILE_MODES = {"append": "a", "write": "w"}
_TELEMETRY_DETAIL_LEVELS = {"basic", "detailed"}
_BENCHMARK_PHASES = ("setup", "run", "verification")
_LLM_SERVER_LAUNCH_MODES = {"process", "thread"}
_HOST_INSPECTOR_LAUNCH_MODES = {"process", "thread"}
_HOST_INSPECTOR_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_TELEMETRY_WRITER_MODES = {"sync", "async"}
_TELEMETRY_OVERFLOW_POLICIES = {"drop_new", "block"}
_TELEMETRY_SERIALIZERS = {"auto", "stdlib", "orjson"}
_SCHEDULER_POLICIES = {"scenario_default", "no_checkpointing"}
_CHECKPOINT_SCHEDULING_POLICIES = {"fifo", "reactive"}
_SETUP_RUN_EXECUTOR_POOLS = {"separate", "shared"}
_SUPPORTED_MODES = {
    "e2e": {"manual"},
    "fault": {"manual", "auto"},
    "spot": {"manual", "auto"},
    "tree": {"manual", "auto"},
    "spec": {"manual", "auto"},
}
_DEFAULT_ITERATIONS = {
    "e2e": 5,
    "fault": 3,
    "spot": 3,
    "tree": 1,
    "spec": 0,
}


@dataclass(frozen=True)
class TelemetryReportConfig:
    enabled: bool = True
    output_dir: Path | None = None
    top_k: int = 25
    log_scale_charts: bool = False
    export_svg: bool = True


@dataclass(frozen=True)
class MonitoringConfig:
    enabled: bool = True
    sample_interval_ms: int = 1000
    include_host: bool = True
    include_sandboxes: bool = True


@dataclass(frozen=True)
class BenchmarkPhaseWorkers:
    setup: int | None = None
    run: int | None = None
    verification: int | None = None

    def resolve(self, *, default: int) -> "ResolvedBenchmarkPhaseWorkers":
        return ResolvedBenchmarkPhaseWorkers(
            setup=default if self.setup is None else int(self.setup),
            run=default if self.run is None else int(self.run),
            verification=default if self.verification is None else int(self.verification),
        )


@dataclass(frozen=True)
class ResolvedBenchmarkPhaseWorkers:
    setup: int
    run: int
    verification: int

    def for_phase(self, phase: str) -> int:
        if phase not in _BENCHMARK_PHASES:
            raise KeyError(f"unknown benchmark phase {phase!r}")
        return int(getattr(self, phase))

    def as_dict(self) -> dict[str, int]:
        return {phase: self.for_phase(phase) for phase in _BENCHMARK_PHASES}


@dataclass(frozen=True)
class BenchmarkExecutorConfig:
    checkpoint_workers: int | None = None
    restore_workers: int | None = None
    coordination_workers: int | None = None
    composite_step_workers: int | None = None
    checkpoint_queue_size: int = 10000
    checkpoint_scheduling_policy: str = "fifo"
    reactive_checkpoint_urgent_quota: int = 4
    max_retries: int = 0
    retry_backoff_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.checkpoint_workers is not None and self.checkpoint_workers <= 0:
            raise ValueError("executor.checkpoint_workers must be positive when provided")
        if self.restore_workers is not None and self.restore_workers <= 0:
            raise ValueError("executor.restore_workers must be positive when provided")
        if self.coordination_workers is not None and self.coordination_workers <= 0:
            raise ValueError("executor.coordination_workers must be positive when provided")
        if self.composite_step_workers is not None and self.composite_step_workers <= 0:
            raise ValueError("executor.composite_step_workers must be positive when provided")
        if self.checkpoint_queue_size <= 0:
            raise ValueError("executor.checkpoint_queue_size must be positive")
        if self.checkpoint_scheduling_policy not in _CHECKPOINT_SCHEDULING_POLICIES:
            raise ValueError(
                "executor.checkpoint_scheduling_policy must be one of "
                f"{sorted(_CHECKPOINT_SCHEDULING_POLICIES)}, got {self.checkpoint_scheduling_policy!r}"
            )
        if self.reactive_checkpoint_urgent_quota <= 0:
            raise ValueError("executor.reactive_checkpoint_urgent_quota must be positive")
        if self.max_retries < 0:
            raise ValueError("executor.max_retries must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError("executor.retry_backoff_seconds must be >= 0")

    def resolve(self, *, default_workers: int) -> ExecutorConfig:
        return ExecutorConfig(
            max_workers=max(1, int(default_workers)),
            checkpoint_workers=self.checkpoint_workers,
            restore_workers=self.restore_workers,
            coordination_workers=self.coordination_workers,
            composite_step_workers=self.composite_step_workers,
            max_checkpoint_queue_size=self.checkpoint_queue_size,
            checkpoint_scheduling_policy=self.checkpoint_scheduling_policy,
            reactive_checkpoint_urgent_quota=self.reactive_checkpoint_urgent_quota,
            max_retries=self.max_retries,
            retry_backoff_seconds=self.retry_backoff_seconds,
        )


@dataclass(frozen=True)
class BenchmarkSchedulerConfig:
    policy: str | None = None
    min_checkpoint_interval_seconds: float | None = None
    force_checkpoint_after_seconds: float | None = None
    require_change_signal: bool | None = None
    checkpoint_full_baseline_on_first_checkpoint: bool | None = None
    prefer_checkpoint_during_llm_request: bool | None = None
    require_llm_request_for_checkpoint: bool | None = None
    inspect_without_pause: bool | None = None
    incremental_process_enabled: bool | None = None
    full_process_checkpoint_interval: int | None = None
    max_process_chain_length: int | None = None

    def __post_init__(self) -> None:
        if self.full_process_checkpoint_interval is not None and self.full_process_checkpoint_interval < 1:
            raise ValueError("scheduler.full_process_checkpoint_interval must be >= 1")
        if self.max_process_chain_length is not None and self.max_process_chain_length < 1:
            raise ValueError("scheduler.max_process_chain_length must be >= 1")
        if self.policy is None:
            return
        if self.policy not in _SCHEDULER_POLICIES:
            raise ValueError(
                f"scheduler.policy must be one of {sorted(_SCHEDULER_POLICIES)}, got {self.policy!r}"
            )

    def apply(self, base: SchedulerConfig) -> SchedulerConfig:
        return SchedulerConfig(
            min_checkpoint_interval_seconds=(
                base.min_checkpoint_interval_seconds
                if self.min_checkpoint_interval_seconds is None
                else self.min_checkpoint_interval_seconds
            ),
            force_checkpoint_after_seconds=(
                base.force_checkpoint_after_seconds
                if self.force_checkpoint_after_seconds is None
                else self.force_checkpoint_after_seconds
            ),
            require_change_signal=(
                base.require_change_signal
                if self.require_change_signal is None
                else self.require_change_signal
            ),
            checkpoint_full_baseline_on_first_checkpoint=(
                base.checkpoint_full_baseline_on_first_checkpoint
                if self.checkpoint_full_baseline_on_first_checkpoint is None
                else self.checkpoint_full_baseline_on_first_checkpoint
            ),
            prefer_checkpoint_during_llm_request=(
                base.prefer_checkpoint_during_llm_request
                if self.prefer_checkpoint_during_llm_request is None
                else self.prefer_checkpoint_during_llm_request
            ),
            require_llm_request_for_checkpoint=(
                base.require_llm_request_for_checkpoint
                if self.require_llm_request_for_checkpoint is None
                else self.require_llm_request_for_checkpoint
            ),
            inspect_without_pause=(
                base.inspect_without_pause
                if self.inspect_without_pause is None
                else self.inspect_without_pause
            ),
            incremental_process_enabled=(
                base.incremental_process_enabled
                if self.incremental_process_enabled is None
                else self.incremental_process_enabled
            ),
            full_process_checkpoint_interval=(
                base.full_process_checkpoint_interval
                if self.full_process_checkpoint_interval is None
                else self.full_process_checkpoint_interval
            ),
            max_process_chain_length=(
                base.max_process_chain_length
                if self.max_process_chain_length is None
                else self.max_process_chain_length
            ),
        )


@dataclass(frozen=True)
class BenchmarkLLMServerConfig:
    launch_mode: str = "process"

    def __post_init__(self) -> None:
        if self.launch_mode not in _LLM_SERVER_LAUNCH_MODES:
            raise ValueError(
                f"llm_server.launch_mode must be one of {sorted(_LLM_SERVER_LAUNCH_MODES)}, "
                f"got {self.launch_mode!r}"
            )


@dataclass(frozen=True)
class BenchmarkHostInspectorConfig:
    launch_mode: str = "process"
    log_level: str = "INFO"
    log_file: bool = False

    def __post_init__(self) -> None:
        if self.launch_mode not in _HOST_INSPECTOR_LAUNCH_MODES:
            raise ValueError(
                f"host_inspector.launch_mode must be one of {sorted(_HOST_INSPECTOR_LAUNCH_MODES)}, "
                f"got {self.launch_mode!r}"
            )
        if self.log_level not in _HOST_INSPECTOR_LOG_LEVELS:
            raise ValueError(
                f"host_inspector.log_level must be one of {sorted(_HOST_INSPECTOR_LOG_LEVELS)}, "
                f"got {self.log_level!r}"
            )


@dataclass(frozen=True)
class BenchmarkStoragePlanesConfig:
    runtime_root: Path | None = None
    storage_root: Path | None = None
    agent_host_root: Path | None = None


@dataclass(frozen=True)
class BenchmarkRootfsReuseConfig:
    enabled: bool = True


@dataclass(frozen=True)
class SandboxResourceLimitsConfig:
    cpus: int | None = None
    memory_bytes: int | None = None
    pids_limit: int | None = None
    cpu_period_us: int = DEFAULT_CPU_PERIOD_US

    def to_runtime_limits(self) -> SandboxResourceLimits | None:
        if self.cpus is None and self.memory_bytes is None and self.pids_limit is None:
            return None
        return SandboxResourceLimits(
            cpus=self.cpus,
            memory_bytes=self.memory_bytes,
            pids_limit=self.pids_limit,
            cpu_period_us=self.cpu_period_us,
        )


@dataclass(frozen=True)
class BenchmarkPhaseMergingConfig:
    setup_and_run: bool = False
    setup_and_run_executor_pool: str = "separate"

    def __post_init__(self) -> None:
        if self.setup_and_run_executor_pool not in _SETUP_RUN_EXECUTOR_POOLS:
            raise ValueError(
                "phase_merging.setup_and_run_executor_pool must be one of "
                f"{sorted(_SETUP_RUN_EXECUTOR_POOLS)}, got {self.setup_and_run_executor_pool!r}"
            )


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
    telemetry_keep_in_memory_copy: bool | None = None
    telemetry_writer_mode: str = DEFAULT_TELEMETRY_WRITER_MODE
    telemetry_queue_capacity: int = DEFAULT_TELEMETRY_QUEUE_CAPACITY
    telemetry_batch_max_records: int = DEFAULT_TELEMETRY_BATCH_MAX_RECORDS
    telemetry_flush_interval_ms: int = DEFAULT_TELEMETRY_FLUSH_INTERVAL_MS
    telemetry_overflow_policy: str = DEFAULT_TELEMETRY_OVERFLOW_POLICY
    telemetry_serializer: str = DEFAULT_TELEMETRY_SERIALIZER
    telemetry_file_mode: str = "append"
    log_file: Path | None = None
    log_file_mode: str = "append"
    verification_enabled: bool = True
    benchmark_root: Path | None = None
    clear_benchmark_root_after_run: bool = False
    zpool_size: str = "10G"
    zpool_name: str | None = None
    zpool_image: Path | None = None
    reuse_zpool: bool = False
    runtime_command_timeout_seconds: float = 60.0
    runtime_zfs_prepare_timeout_seconds: float = 300.0
    image_cache_root: Path | None = None
    log_level: str = "info"
    transfer_delay_ms: float = 0.0
    work_dir_host_root: Path | None = None
    scenario_options: dict[str, object] = field(default_factory=dict)
    llm_service_options: dict[str, object] = field(default_factory=dict)
    max_agent_timeout_scale: float = 1.0
    max_test_timeout_scale: float = 1.0
    # Per-task scale overrides keyed by task_id. Take precedence over the
    # scalar `max_*_timeout_scale` for matching tasks. Useful for traces
    # whose recorded `max_test_timeout_sec` is calibrated against an
    # uncapped baseline and exceeds the global scale on our 4-CPU
    # cgroup (e.g. pytorch-model-cli still needs >180s even at 8 CPUs).
    max_agent_timeout_scale_overrides: dict[str, float] = field(default_factory=dict)
    max_test_timeout_scale_overrides: dict[str, float] = field(default_factory=dict)
    telemetry_report: TelemetryReportConfig = field(default_factory=TelemetryReportConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    executor: BenchmarkExecutorConfig = field(default_factory=BenchmarkExecutorConfig)
    scheduler: BenchmarkSchedulerConfig = field(default_factory=BenchmarkSchedulerConfig)
    llm_server: BenchmarkLLMServerConfig = field(default_factory=BenchmarkLLMServerConfig)
    host_inspector: BenchmarkHostInspectorConfig = field(default_factory=BenchmarkHostInspectorConfig)
    storage_planes: BenchmarkStoragePlanesConfig = field(default_factory=BenchmarkStoragePlanesConfig)
    rootfs_reuse: BenchmarkRootfsReuseConfig = field(default_factory=BenchmarkRootfsReuseConfig)
    phase_merging: BenchmarkPhaseMergingConfig = field(default_factory=BenchmarkPhaseMergingConfig)
    sandbox_resource_limits: SandboxResourceLimitsConfig = field(default_factory=SandboxResourceLimitsConfig)
    benchmark_root_home: Path | None = None
    benchmark_run_name: str | None = None
    relaunch_on_restore_failure: bool = False

    def __post_init__(self) -> None:
        benchmark_root = _coerce_optional_path(self.benchmark_root)
        benchmark_root_home = _coerce_optional_path(self.benchmark_root_home)
        if (
            benchmark_root is not None
            and benchmark_root_home is not None
            and benchmark_root != benchmark_root_home
        ):
            raise ValueError(
                "benchmark_root and benchmark_root_home are aliases; specify only one "
                "or give them the same value"
            )
        resolved_home = benchmark_root_home or benchmark_root
        object.__setattr__(self, "benchmark_root", resolved_home)
        object.__setattr__(self, "benchmark_root_home", resolved_home)
        object.__setattr__(
            self,
            "benchmark_run_name",
            _normalize_benchmark_run_name(self.benchmark_run_name),
        )

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


def _coerce_optional_path(raw_value: object) -> Path | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, Path):
        return raw_value
    return Path(str(raw_value))


def _normalize_benchmark_run_name(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip()
    path = Path(value)
    if not value or path.is_absolute() or len(path.parts) != 1 or value in {".", ".."}:
        raise ValueError("benchmark_run_name must be a single non-empty directory name")
    return value


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


def _load_executor_config(payload: object) -> BenchmarkExecutorConfig:
    if payload is None:
        return BenchmarkExecutorConfig()
    data = _require_object(payload, label="executor")
    return BenchmarkExecutorConfig(
        checkpoint_workers=(
            None if data.get("checkpoint_workers") is None else int(data["checkpoint_workers"])
        ),
        restore_workers=(
            None if data.get("restore_workers") is None else int(data["restore_workers"])
        ),
        coordination_workers=(
            None if data.get("coordination_workers") is None else int(data["coordination_workers"])
        ),
        composite_step_workers=(
            None if data.get("composite_step_workers") is None else int(data["composite_step_workers"])
        ),
        checkpoint_queue_size=int(data.get("checkpoint_queue_size", 10000)),
        checkpoint_scheduling_policy=str(data.get("checkpoint_scheduling_policy", "fifo")),
        reactive_checkpoint_urgent_quota=int(data.get("reactive_checkpoint_urgent_quota", 4)),
        max_retries=int(data.get("max_retries", 0)),
        retry_backoff_seconds=float(data.get("retry_backoff_seconds", 0.05)),
    )


def _load_scheduler_config(payload: object) -> BenchmarkSchedulerConfig:
    if payload is None:
        return BenchmarkSchedulerConfig()
    data = _require_object(payload, label="scheduler")
    return BenchmarkSchedulerConfig(
        policy=(
            None
            if data.get("policy") is None
            else str(data["policy"]).strip().lower()
        ),
        min_checkpoint_interval_seconds=(
            None
            if data.get("min_checkpoint_interval_seconds") is None
            else float(data["min_checkpoint_interval_seconds"])
        ),
        force_checkpoint_after_seconds=(
            None
            if data.get("force_checkpoint_after_seconds") is None
            else float(data["force_checkpoint_after_seconds"])
        ),
        require_change_signal=(
            None if data.get("require_change_signal") is None else bool(data["require_change_signal"])
        ),
        checkpoint_full_baseline_on_first_checkpoint=(
            None
            if data.get("checkpoint_full_baseline_on_first_checkpoint") is None
            else bool(data["checkpoint_full_baseline_on_first_checkpoint"])
        ),
        prefer_checkpoint_during_llm_request=(
            None
            if data.get("prefer_checkpoint_during_llm_request") is None
            else bool(data["prefer_checkpoint_during_llm_request"])
        ),
        require_llm_request_for_checkpoint=(
            None
            if data.get("require_llm_request_for_checkpoint") is None
            else bool(data["require_llm_request_for_checkpoint"])
        ),
        inspect_without_pause=(
            None if data.get("inspect_without_pause") is None else bool(data["inspect_without_pause"])
        ),
        incremental_process_enabled=(
            None
            if data.get("incremental_process_enabled") is None
            else bool(data["incremental_process_enabled"])
        ),
        full_process_checkpoint_interval=(
            None
            if data.get("full_process_checkpoint_interval") is None
            else int(data["full_process_checkpoint_interval"])
        ),
        max_process_chain_length=(
            None
            if data.get("max_process_chain_length") is None
            else int(data["max_process_chain_length"])
        ),
    )


def _load_llm_server_config(payload: object) -> BenchmarkLLMServerConfig:
    if payload is None:
        return BenchmarkLLMServerConfig()
    data = _require_object(payload, label="llm_server")
    return BenchmarkLLMServerConfig(
        launch_mode=str(data.get("launch_mode", "process")).strip().lower(),
    )


def _load_host_inspector_config(payload: object) -> BenchmarkHostInspectorConfig:
    if payload is None:
        return BenchmarkHostInspectorConfig()
    data = _require_object(payload, label="host_inspector")
    return BenchmarkHostInspectorConfig(
        launch_mode=str(data.get("launch_mode", "process")).strip().lower(),
        log_level=str(data.get("log_level", "INFO")).strip().upper(),
        log_file=bool(data.get("log_file", False)),
    )


def _load_storage_planes_config(base_dir: Path, payload: object) -> BenchmarkStoragePlanesConfig:
    if payload is None:
        return BenchmarkStoragePlanesConfig()
    data = _require_object(payload, label="storage_planes")
    return BenchmarkStoragePlanesConfig(
        runtime_root=_resolve_optional_path(base_dir, data.get("runtime_root")),
        storage_root=_resolve_optional_path(base_dir, data.get("storage_root")),
        agent_host_root=_resolve_optional_path(base_dir, data.get("agent_host_root")),
    )


def _load_rootfs_reuse_config(payload: object) -> BenchmarkRootfsReuseConfig:
    if payload is None:
        return BenchmarkRootfsReuseConfig()
    data = _require_object(payload, label="rootfs_reuse")
    return BenchmarkRootfsReuseConfig(
        enabled=bool(data.get("enabled", True)),
    )


def _load_phase_merging_config(payload: object) -> BenchmarkPhaseMergingConfig:
    if payload is None:
        return BenchmarkPhaseMergingConfig()
    data = _require_object(payload, label="phase_merging")
    return BenchmarkPhaseMergingConfig(
        setup_and_run=bool(data.get("setup_and_run", False)),
        setup_and_run_executor_pool=str(data.get("setup_and_run_executor_pool", "separate")).strip().lower(),
    )


def _optional_positive_int(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    result = int(value)
    if result <= 0:
        raise ValueError(f"{label} must be positive when set, got {result}")
    return result


def _load_sandbox_resource_limits_config(payload: object) -> SandboxResourceLimitsConfig:
    if payload is None:
        return SandboxResourceLimitsConfig()
    data = _require_object(payload, label="sandbox_resource_limits")
    known = {"cpus", "memory_bytes", "pids_limit", "cpu_period_us"}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(
            f"sandbox_resource_limits contains unknown keys {unknown}; expected subset of {sorted(known)}"
        )
    return SandboxResourceLimitsConfig(
        cpus=_optional_positive_int(data.get("cpus"), label="sandbox_resource_limits.cpus"),
        memory_bytes=_optional_positive_int(
            data.get("memory_bytes"), label="sandbox_resource_limits.memory_bytes"
        ),
        pids_limit=_optional_positive_int(
            data.get("pids_limit"), label="sandbox_resource_limits.pids_limit"
        ),
        cpu_period_us=int(data.get("cpu_period_us", DEFAULT_CPU_PERIOD_US)),
    )


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
    if log_file_mode not in _FILE_MODES:
        raise ValueError(f"log_file_mode must be one of {sorted(_FILE_MODES)}, got {log_file_mode!r}")

    sandboxes = int(data.get("sandboxes", 1))
    if sandboxes <= 0:
        raise ValueError(f"sandboxes must be positive, got {sandboxes}")
    raw_max_workers = data.get("max_workers")
    max_workers = None if raw_max_workers is None else int(raw_max_workers)
    if max_workers is not None and max_workers <= 0:
        raise ValueError(f"max_workers must be positive when provided, got {max_workers}")
    phase_workers = _load_phase_workers(data.get("phase_workers"))

    iterations = int(data.get("iterations", _DEFAULT_ITERATIONS[scenario]))
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")
    if iterations == 0 and scenario not in {"fault", "spec"}:
        raise ValueError(f"iterations must be positive for scenario {scenario!r}, got {iterations}")
    if scenario == "spec" and iterations != 0:
        raise ValueError("iterations must be exactly 0 for scenario 'spec' because it replays full traces")

    agent = str(data.get("agent", "simulated"))
    llm_service = None if data.get("llm_service") is None else str(data["llm_service"])
    if scenario == "spec":
        spec_agent_to_llm = {
            "mini_swe": "mini_swe_spec_trace_replay",
            "terminus": "terminus_spec_trace_replay",
        }
        if agent not in spec_agent_to_llm:
            raise ValueError(
                f"scenario='spec' requires agent in {sorted(spec_agent_to_llm)}, got {agent!r}"
            )
        expected_llm_service = spec_agent_to_llm[agent]
        if llm_service != expected_llm_service:
            raise ValueError(
                f"scenario='spec' with agent='{agent}' requires llm_service='{expected_llm_service}'"
            )

    scenario_options = data.get("scenario_options", {})
    if scenario_options is None:
        scenario_options = {}
    scenario_options = _require_object(scenario_options, label="scenario_options")

    llm_service_options = data.get("llm_service_options", {})
    if llm_service_options is None:
        llm_service_options = {}
    llm_service_options = _require_object(llm_service_options, label="llm_service_options")
    max_agent_timeout_scale = float(data.get("max_agent_timeout_scale", 1.0))
    if max_agent_timeout_scale < 0:
        raise ValueError(f"max_agent_timeout_scale must be non-negative, got {max_agent_timeout_scale!r}")
    max_test_timeout_scale = float(data.get("max_test_timeout_scale", 1.0))
    if max_test_timeout_scale < 0:
        raise ValueError(f"max_test_timeout_scale must be non-negative, got {max_test_timeout_scale!r}")

    def _parse_scale_overrides(key: str) -> dict[str, float]:
        raw = data.get(key) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"{key} must be an object mapping task_id → scale")
        out: dict[str, float] = {}
        for task_id, value in raw.items():
            scale = float(value)
            if scale < 0:
                raise ValueError(f"{key}[{task_id!r}] must be non-negative, got {scale!r}")
            out[str(task_id)] = scale
        return out

    max_agent_timeout_scale_overrides = _parse_scale_overrides("max_agent_timeout_scale_overrides")
    max_test_timeout_scale_overrides = _parse_scale_overrides("max_test_timeout_scale_overrides")

    base_dir = config_path.parent
    telemetry_options = data.get("telemetry", {})
    if telemetry_options is None:
        telemetry_options = {}
    telemetry_options = _require_object(telemetry_options, label="telemetry")
    telemetry_output = _resolve_optional_path(
        base_dir,
        telemetry_options.get("output", data.get("telemetry_output")),
    )
    telemetry_file_mode = str(
        telemetry_options.get("file_mode", data.get("telemetry_file_mode", "append"))
    ).strip().lower()
    if telemetry_file_mode not in _FILE_MODES:
        raise ValueError(
            f"telemetry.file_mode must be one of {sorted(_FILE_MODES)}, got {telemetry_file_mode!r}"
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
    telemetry_keep_in_memory_copy = (
        None if telemetry_options.get("keep_in_memory_copy") is None else bool(telemetry_options["keep_in_memory_copy"])
    )
    telemetry_writer_mode = str(
        telemetry_options.get("writer_mode", DEFAULT_TELEMETRY_WRITER_MODE)
    ).strip().lower()
    if telemetry_writer_mode not in _TELEMETRY_WRITER_MODES:
        raise ValueError(
            f"telemetry.writer_mode must be one of {sorted(_TELEMETRY_WRITER_MODES)}, got {telemetry_writer_mode!r}"
        )
    telemetry_queue_capacity = int(
        telemetry_options.get("queue_capacity", DEFAULT_TELEMETRY_QUEUE_CAPACITY)
    )
    if telemetry_queue_capacity <= 0:
        raise ValueError("telemetry.queue_capacity must be positive")
    telemetry_batch_max_records = int(
        telemetry_options.get("batch_max_records", DEFAULT_TELEMETRY_BATCH_MAX_RECORDS)
    )
    if telemetry_batch_max_records <= 0:
        raise ValueError("telemetry.batch_max_records must be positive")
    telemetry_flush_interval_ms = int(
        telemetry_options.get("flush_interval_ms", DEFAULT_TELEMETRY_FLUSH_INTERVAL_MS)
    )
    if telemetry_flush_interval_ms <= 0:
        raise ValueError("telemetry.flush_interval_ms must be positive")
    telemetry_overflow_policy = str(
        telemetry_options.get("overflow_policy", DEFAULT_TELEMETRY_OVERFLOW_POLICY)
    ).strip().lower()
    if telemetry_overflow_policy not in _TELEMETRY_OVERFLOW_POLICIES:
        raise ValueError(
            f"telemetry.overflow_policy must be one of {sorted(_TELEMETRY_OVERFLOW_POLICIES)}, got {telemetry_overflow_policy!r}"
        )
    telemetry_serializer = str(
        telemetry_options.get("serializer", DEFAULT_TELEMETRY_SERIALIZER)
    ).strip().lower()
    if telemetry_serializer not in _TELEMETRY_SERIALIZERS:
        raise ValueError(
            f"telemetry.serializer must be one of {sorted(_TELEMETRY_SERIALIZERS)}, got {telemetry_serializer!r}"
        )
    report_options = telemetry_options.get("report", {})
    if report_options is None:
        report_options = {}
    report_options = _require_object(report_options, label="telemetry.report")
    report_top_k = int(report_options.get("top_k", 25))
    if report_top_k <= 0:
        raise ValueError("telemetry.report.top_k must be positive")
    telemetry_report = TelemetryReportConfig(
        enabled=bool(report_options.get("enabled", True)),
        output_dir=_resolve_optional_path(base_dir, report_options.get("output_dir")),
        top_k=report_top_k,
        log_scale_charts=bool(report_options.get("log_scale_charts", False)),
        export_svg=bool(report_options.get("export_svg", True)),
    )

    monitoring_options = data.get("monitoring", {})
    if monitoring_options is None:
        monitoring_options = {}
    monitoring_options = _require_object(monitoring_options, label="monitoring")
    monitoring_sample_interval_ms = int(monitoring_options.get("sample_interval_ms", 1000))
    if monitoring_sample_interval_ms <= 0:
        raise ValueError("monitoring.sample_interval_ms must be positive")
    monitoring = MonitoringConfig(
        enabled=bool(monitoring_options.get("enabled", True)),
        sample_interval_ms=monitoring_sample_interval_ms,
        include_host=bool(monitoring_options.get("include_host", True)),
        include_sandboxes=bool(monitoring_options.get("include_sandboxes", True)),
    )
    executor = _load_executor_config(data.get("executor"))
    scheduler = _load_scheduler_config(data.get("scheduler"))
    llm_server = _load_llm_server_config(data.get("llm_server"))
    host_inspector = _load_host_inspector_config(data.get("host_inspector"))
    storage_planes = _load_storage_planes_config(base_dir, data.get("storage_planes"))
    rootfs_reuse = _load_rootfs_reuse_config(data.get("rootfs_reuse"))
    phase_merging = _load_phase_merging_config(data.get("phase_merging"))
    sandbox_resource_limits = _load_sandbox_resource_limits_config(
        data.get("sandbox_resource_limits")
    )
    verification_options = data.get("verification", {})
    if verification_options is None:
        verification_options = {}
    verification_options = _require_object(verification_options, label="verification")
    verification_enabled = bool(
        verification_options.get(
            "enabled",
            data.get("verification_enabled", True),
        )
    )
    benchmark_root = _resolve_optional_path(base_dir, data.get("benchmark_root"))
    benchmark_root_home = _resolve_optional_path(base_dir, data.get("benchmark_root_home"))
    benchmark_run_name = _normalize_benchmark_run_name(data.get("benchmark_run_name"))

    return BenchmarkConfig(
        config_path=config_path,
        scenario=scenario,
        mode=mode,
        provider=provider,
        agent=agent,
        llm_service=llm_service,
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
        telemetry_keep_in_memory_copy=telemetry_keep_in_memory_copy,
        telemetry_writer_mode=telemetry_writer_mode,
        telemetry_queue_capacity=telemetry_queue_capacity,
        telemetry_batch_max_records=telemetry_batch_max_records,
        telemetry_flush_interval_ms=telemetry_flush_interval_ms,
        telemetry_overflow_policy=telemetry_overflow_policy,
        telemetry_serializer=telemetry_serializer,
        telemetry_file_mode=telemetry_file_mode,
        log_file=_resolve_optional_path(base_dir, data.get("log_file")),
        log_file_mode=log_file_mode,
        verification_enabled=verification_enabled,
        benchmark_root=benchmark_root,
        clear_benchmark_root_after_run=bool(data.get("clear_benchmark_root_after_run", False)),
        zpool_size=str(data.get("zpool_size", "10G")),
        zpool_name=None if data.get("zpool_name") is None else str(data.get("zpool_name")),
        zpool_image=_resolve_optional_path(base_dir, data.get("zpool_image")),
        reuse_zpool=bool(data.get("reuse_zpool", False)),
        runtime_command_timeout_seconds=float(data.get("runtime_command_timeout_seconds", 60.0)),
        runtime_zfs_prepare_timeout_seconds=float(data.get("runtime_zfs_prepare_timeout_seconds", 300.0)),
        image_cache_root=_resolve_optional_path(base_dir, data.get("image_cache_root")),
        log_level=log_level,
        transfer_delay_ms=float(data.get("transfer_delay_ms", 0.0)),
        work_dir_host_root=_resolve_optional_path(base_dir, data.get("work_dir_host_root")),
        scenario_options=scenario_options,
        llm_service_options=llm_service_options,
        max_agent_timeout_scale=max_agent_timeout_scale,
        max_test_timeout_scale=max_test_timeout_scale,
        max_agent_timeout_scale_overrides=max_agent_timeout_scale_overrides,
        max_test_timeout_scale_overrides=max_test_timeout_scale_overrides,
        telemetry_report=telemetry_report,
        monitoring=monitoring,
        executor=executor,
        scheduler=scheduler,
        llm_server=llm_server,
        host_inspector=host_inspector,
        storage_planes=storage_planes,
        rootfs_reuse=rootfs_reuse,
        phase_merging=phase_merging,
        sandbox_resource_limits=sandbox_resource_limits,
        benchmark_root_home=benchmark_root_home,
        benchmark_run_name=benchmark_run_name,
        relaunch_on_restore_failure=bool(data.get("relaunch_on_restore_failure", False)),
    )

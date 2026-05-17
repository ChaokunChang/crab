"""Agent-CR Engine — the host-side runtime manager for the SDK.

The Engine is what users would call a "daemon," conceptually similar to
`dockerd`. In this first cut the Engine runs in-process when no daemon is
available; `Engine.connect()` is reserved for the future daemon mode and
raises `NotImplementedError` today.

A single Engine owns:
  - the underlying `Runtime` (in-memory for testing, `RuncRuntime` for prod)
  - the `AgentCRSystem` (scheduler, executor, checkpoint manager, retention)
  - the `AgentCRRequestInterceptorServer` (semantic-aware C/R via LLM traffic)
  - per-sandbox upstream URL bookkeeping

Users typically don't touch the Engine. `Sandbox(...)` calls
`get_default_engine()` which lazily starts an in-process engine. The
sysadmin/cloud-operator path that starts a daemon will swap that default
to a `connect()` instance once the daemon mode lands.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .config import ExecutorConfig, SchedulerConfig, StorageConfig, TelemetryConfig
from .contracts import Runtime
from .executor import CRExecutor
from .ids import SandboxId
from .inspector import EBPFSandboxInspector
from .interceptor import (
    AgentCRRequestInterceptorServer,
    CompositeRequestInterceptorHook,
    InMemoryRequestStateStore,
    RequestAwareSandboxInspector,
    SandboxResponseGateRegistry,
)
from .runtime import (
    RuncCheckpointOptions,
    RuncRestoreOptions,
    RuncRuntime,
    RuncRuntimeOptions,
    RuncRuntimePaths,
)
from .scheduler import CRScheduler, FaultToleranceCheckpointingPolicy, InMemorySchedulerStateStore
from .sdk_llm_forwarder import SdkLLMForwarder, serve_sdk_llm_forwarder
from .storage import LocalCheckpointManager
from .system import AgentCRSystem, build_default_system
from .telemetry import build_configured_telemetry_sink
from .workers import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    DefaultCWorker,
    DefaultRWorker,
)

if TYPE_CHECKING:
    from .sandbox import Sandbox

logger = logging.getLogger(__name__)


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _optional_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _require_mapping(value, label=label)


def _resolve_config_path(value: object, *, base_dir: Path) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def _as_int(value: object, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    return int(value)


def _as_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    return float(value)


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is part of the benchmark env.
        raise RuntimeError("EngineConfig.from_file() requires PyYAML to load YAML config files") from exc
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _require_mapping(payload, label=f"engine config {path}")


def _telemetry_config_from_mapping(data: Mapping[str, Any], *, base_dir: Path) -> TelemetryConfig:
    base = TelemetryConfig()
    jsonl_path = _resolve_config_path(
        data.get("jsonl_path", data.get("output")),
        base_dir=base_dir,
    )
    return TelemetryConfig(
        enabled=_as_bool(data.get("enabled"), default=base.enabled),
        jsonl_path=jsonl_path,
        keep_in_memory_copy=(
            None if data.get("keep_in_memory_copy") is None else _as_bool(data.get("keep_in_memory_copy"), default=False)
        ),
        detail_level=str(data.get("detail_level", base.detail_level)),
        capture_command_output=_as_bool(
            data.get("capture_command_output"),
            default=base.capture_command_output,
        ),
        max_text_attribute_bytes=int(data.get("max_text_attribute_bytes", base.max_text_attribute_bytes)),
        writer_mode=str(data.get("writer_mode", base.writer_mode)),
        queue_capacity=int(data.get("queue_capacity", base.queue_capacity)),
        batch_max_records=int(data.get("batch_max_records", base.batch_max_records)),
        flush_interval_ms=int(data.get("flush_interval_ms", base.flush_interval_ms)),
        overflow_policy=str(data.get("overflow_policy", base.overflow_policy)),
        serializer=str(data.get("serializer", base.serializer)),
    )


def _executor_config_from_mapping(data: Mapping[str, Any], *, default_workers: int | None) -> ExecutorConfig:
    base = ExecutorConfig(max_workers=max(1, int(default_workers or ExecutorConfig().max_workers)))
    return ExecutorConfig(
        max_workers=int(data.get("max_workers", base.max_workers)),
        checkpoint_workers=_as_int(data.get("checkpoint_workers")),
        restore_workers=_as_int(data.get("restore_workers")),
        coordination_workers=_as_int(data.get("coordination_workers")),
        composite_step_workers=_as_int(data.get("composite_step_workers")),
        max_checkpoint_queue_size=int(
            data.get("max_checkpoint_queue_size", data.get("checkpoint_queue_size", base.max_checkpoint_queue_size))
        ),
        checkpoint_scheduling_policy=str(data.get("checkpoint_scheduling_policy", base.checkpoint_scheduling_policy)),
        reactive_checkpoint_urgent_quota=int(
            data.get("reactive_checkpoint_urgent_quota", base.reactive_checkpoint_urgent_quota)
        ),
        max_retries=int(data.get("max_retries", base.max_retries)),
        retry_backoff_seconds=float(data.get("retry_backoff_seconds", base.retry_backoff_seconds)),
    )


def _scheduler_config_from_mapping(data: Mapping[str, Any]) -> SchedulerConfig:
    base = SchedulerConfig()
    return SchedulerConfig(
        min_checkpoint_interval_seconds=_as_float(
            data.get("min_checkpoint_interval_seconds"),
            default=base.min_checkpoint_interval_seconds,
        ),
        force_checkpoint_after_seconds=_as_float(
            data.get("force_checkpoint_after_seconds"),
            default=base.force_checkpoint_after_seconds,
        ),
        require_change_signal=_as_bool(data.get("require_change_signal"), default=base.require_change_signal),
        checkpoint_full_baseline_on_first_checkpoint=_as_bool(
            data.get("checkpoint_full_baseline_on_first_checkpoint"),
            default=base.checkpoint_full_baseline_on_first_checkpoint,
        ),
        prefer_checkpoint_during_llm_request=_as_bool(
            data.get("prefer_checkpoint_during_llm_request"),
            default=base.prefer_checkpoint_during_llm_request,
        ),
        require_llm_request_for_checkpoint=_as_bool(
            data.get("require_llm_request_for_checkpoint"),
            default=base.require_llm_request_for_checkpoint,
        ),
        inspect_without_pause=_as_bool(data.get("inspect_without_pause"), default=base.inspect_without_pause),
        incremental_process_enabled=_as_bool(
            data.get("incremental_process_enabled"),
            default=base.incremental_process_enabled,
        ),
        full_process_checkpoint_interval=int(
            data.get("full_process_checkpoint_interval", base.full_process_checkpoint_interval)
        ),
        max_process_chain_length=int(data.get("max_process_chain_length", base.max_process_chain_length)),
    )


def _runc_options_from_mapping(data: Mapping[str, Any]) -> RuncRuntimeOptions:
    base = RuncRuntimeOptions()
    checkpoint_data = _optional_mapping(data.get("checkpoint"), label="runc.checkpoint")
    restore_data = _optional_mapping(data.get("restore"), label="runc.restore")
    checkpoint = RuncCheckpointOptions(
        tcp_established=_as_bool(
            checkpoint_data.get("tcp_established"),
            default=base.checkpoint.tcp_established,
        ),
        shell_job=_as_bool(checkpoint_data.get("shell_job"), default=base.checkpoint.shell_job),
        tcp_skip_in_flight=_as_bool(
            checkpoint_data.get("tcp_skip_in_flight"),
            default=base.checkpoint.tcp_skip_in_flight,
        ),
        ext_unix_sk=_as_bool(checkpoint_data.get("ext_unix_sk"), default=base.checkpoint.ext_unix_sk),
        extra_args=tuple(str(item) for item in checkpoint_data.get("extra_args", base.checkpoint.extra_args)),
    )
    restore = RuncRestoreOptions(
        detach=_as_bool(restore_data.get("detach"), default=base.restore.detach),
        tcp_established=_as_bool(restore_data.get("tcp_established"), default=base.restore.tcp_established),
        shell_job=_as_bool(restore_data.get("shell_job"), default=base.restore.shell_job),
        ext_unix_sk=_as_bool(restore_data.get("ext_unix_sk"), default=base.restore.ext_unix_sk),
        lazy_pages=_as_bool(restore_data.get("lazy_pages"), default=base.restore.lazy_pages),
        extra_args=tuple(str(item) for item in restore_data.get("extra_args", base.restore.extra_args)),
    )
    return RuncRuntimeOptions(
        checkpoint=checkpoint,
        restore=restore,
        command_timeout_seconds=float(data.get("command_timeout_seconds", base.command_timeout_seconds)),
        zfs_prepare_timeout_seconds=float(data.get("zfs_prepare_timeout_seconds", base.zfs_prepare_timeout_seconds)),
    )


def _runc_paths_from_mapping(data: Mapping[str, Any], *, base_dir: Path) -> RuncRuntimePaths:
    base = RuncRuntimePaths()
    return RuncRuntimePaths(
        state_root=_resolve_config_path(data.get("state_root"), base_dir=base_dir) or base.state_root,
        bundle_root=_resolve_config_path(data.get("bundle_root"), base_dir=base_dir) or base.bundle_root,
        checkpoint_root=_resolve_config_path(data.get("checkpoint_root"), base_dir=base_dir) or base.checkpoint_root,
        metadata_root=_resolve_config_path(data.get("metadata_root"), base_dir=base_dir) or base.metadata_root,
        zfs_dataset_prefix=str(data.get("zfs_dataset_prefix", base.zfs_dataset_prefix)).rstrip("/"),
    )


def _coerce_engine_config(config: object | None) -> "EngineConfig":
    if config is None:
        return EngineConfig()
    if isinstance(config, EngineConfig):
        return config
    if isinstance(config, (str, os.PathLike)):
        return EngineConfig.from_file(config)
    if isinstance(config, Mapping):
        return EngineConfig.from_mapping(config)
    raise TypeError(f"unsupported engine config type: {type(config).__name__}")


@dataclass
class EngineConfig:
    """Engine configuration.

    Operators tune this in `/etc/agentcr/config.yaml` (planned). For the SDK,
    `EngineConfig()` produces sensible defaults for in-process use with the
    real runc backend. Unit tests can still pass `runtime="docker"` to use
    the in-memory runtime.

    Telemetry is intentionally not exposed here for user-facing use — see
    `agent_cr.telemetry` for sysadmin-side configuration.
    """

    runtime: str = "runc"
    """`runc` selects the real Agent-CR runtime. `docker` selects the
    in-memory runtime used by lightweight unit tests."""

    run_id: str | None = None
    """Optional run identifier stamped onto engine telemetry records."""

    storage_root: Path | None = None
    """Root directory for checkpoint manifests. Defaults to a temp dir under
    `/tmp/agentcr-<random>` for the in-process case."""

    interceptor_host: str = "127.0.0.1"
    interceptor_port: int = 0
    """0 means pick a free port automatically."""

    forwarder_host: str = "127.0.0.1"
    forwarder_port: int = 0
    """Host + port for the per-sandbox LLM forwarder. The interceptor's
    upstream is set to this forwarder's URL; the forwarder dispatches each
    sandbox's traffic to the real LLM URL the sandbox registered. Same
    architectural pattern as the benchmark harness, which points the
    interceptor at a single BenchmarkLLMRouter URL."""

    enable_interceptor: bool = True
    """Disable the LLM interceptor when the host does not need outbound
    LLM tagging (e.g. bare-image experiments). Disables the forwarder too."""

    enable_sandbox_network: bool = True
    """For runc, create a small bridge/veth network when in-sandbox agents
    need deterministic LLM request attribution across multiple sandboxes."""

    network_expected_sandboxes: int | None = None
    """Optional capacity hint for the bridge network allocator."""

    runc_options: RuncRuntimeOptions | None = None
    """Override knobs for `RuncRuntime`. Only used when `runtime == "runc"`."""

    runc_paths: RuncRuntimePaths | None = None
    """Override runc state/bundle/checkpoint/metadata paths."""

    zfs_dataset_prefix: str | None = None
    """ZFS dataset prefix for sandbox rootfs datasets, e.g.
    `agentcr-300/agent-cr-sdk`. If unset, the engine uses
    `$AGENT_CR_ZFS_DATASET_PREFIX`, `$AGENT_CR_ZPOOL_NAME/agent-cr-sdk`, or
    an installed zpool whose name starts with `agentcr`."""

    runtime_root: Path | None = None
    """Host root for runc state, bundles, metadata, image cache, work dirs,
    and per-agent state. Defaults under `storage_root`."""

    image_cache_root: Path | None = None
    work_dir_host_root: Path | None = None
    agent_state_root: Path | None = None
    default_image: str = "ubuntu:22.04"

    scheduler_config: SchedulerConfig | None = None
    executor_config: ExecutorConfig | None = None
    storage_config: StorageConfig | None = None
    telemetry_config: TelemetryConfig | None = None

    log_file: Path | None = None
    """Optional SDK engine log file. If unset, the engine does not install
    a file logger and regular Python logging configuration applies."""

    log_level: str = "INFO"
    log_file_mode: str = "append"
    """`append` or `write` when `log_file` is set."""

    agent_worker_threads: int = 4
    """Max concurrent `agent.run()` invocations across all sandboxes."""

    extra: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | os.PathLike[str]) -> "EngineConfig":
        config_path = Path(path).expanduser().resolve()
        return cls.from_mapping(_load_yaml_mapping(config_path), base_dir=config_path.parent)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: Path | None = None,
    ) -> "EngineConfig":
        data = _require_mapping(payload.get("engine", payload), label="engine config")
        config_base_dir = Path.cwd() if base_dir is None else Path(base_dir).expanduser().resolve()
        storage_planes = _optional_mapping(data.get("storage_planes"), label="storage_planes")
        network = _optional_mapping(data.get("network"), label="network")
        interceptor = _optional_mapping(data.get("interceptor"), label="interceptor")
        forwarder = _optional_mapping(data.get("forwarder"), label="forwarder")
        logging_config = _optional_mapping(data.get("logging"), label="logging")

        default_workers = _as_int(data.get("max_workers"), default=None)
        executor_data = _optional_mapping(data.get("executor"), label="executor")
        scheduler_data = _optional_mapping(data.get("scheduler"), label="scheduler")
        telemetry_data = _optional_mapping(data.get("telemetry"), label="telemetry")
        runc_data = _optional_mapping(data.get("runc"), label="runc")
        runc_paths_data = _optional_mapping(data.get("runc_paths"), label="runc_paths")

        storage_root = _resolve_config_path(
            data.get("storage_root", storage_planes.get("storage_root")),
            base_dir=config_base_dir,
        )
        runtime_root = _resolve_config_path(
            data.get("runtime_root", storage_planes.get("runtime_root")),
            base_dir=config_base_dir,
        )
        image_cache_root = _resolve_config_path(
            data.get("image_cache_root", storage_planes.get("image_cache_root")),
            base_dir=config_base_dir,
        )
        work_dir_host_root = _resolve_config_path(
            data.get("work_dir_host_root", storage_planes.get("work_dir_host_root")),
            base_dir=config_base_dir,
        )
        agent_state_root = _resolve_config_path(
            data.get(
                "agent_state_root",
                storage_planes.get("agent_state_root", storage_planes.get("agent_host_root")),
            ),
            base_dir=config_base_dir,
        )

        telemetry_config = _telemetry_config_from_mapping(telemetry_data, base_dir=config_base_dir) if telemetry_data else None
        executor_config = _executor_config_from_mapping(executor_data, default_workers=default_workers) if executor_data or default_workers else None
        scheduler_config = _scheduler_config_from_mapping(scheduler_data) if scheduler_data else None
        runc_options = _runc_options_from_mapping(runc_data) if runc_data else None
        runc_paths = _runc_paths_from_mapping(runc_paths_data, base_dir=config_base_dir) if runc_paths_data else None

        log_file = _resolve_config_path(
            data.get("log_file", logging_config.get("file", logging_config.get("log_file"))),
            base_dir=config_base_dir,
        )
        log_level = str(data.get("log_level", logging_config.get("level", logging_config.get("log_level", "INFO"))))
        log_file_mode = str(
            data.get("log_file_mode", logging_config.get("file_mode", logging_config.get("log_file_mode", "append")))
        ).strip().lower()

        return cls(
            runtime=str(data.get("runtime", "runc")),
            run_id=None if data.get("run_id") is None else str(data.get("run_id")),
            storage_root=storage_root,
            interceptor_host=str(interceptor.get("host", data.get("interceptor_host", "127.0.0.1"))),
            interceptor_port=int(interceptor.get("port", data.get("interceptor_port", 0))),
            forwarder_host=str(forwarder.get("host", data.get("forwarder_host", "127.0.0.1"))),
            forwarder_port=int(forwarder.get("port", data.get("forwarder_port", 0))),
            enable_interceptor=_as_bool(
                interceptor.get("enabled", data.get("enable_interceptor")),
                default=True,
            ),
            enable_sandbox_network=_as_bool(
                network.get("enable_sandbox_network", data.get("enable_sandbox_network")),
                default=True,
            ),
            network_expected_sandboxes=_as_int(
                network.get("expected_sandboxes", data.get("network_expected_sandboxes")),
                default=None,
            ),
            runc_options=runc_options,
            runc_paths=runc_paths,
            zfs_dataset_prefix=(
                None
                if data.get("zfs_dataset_prefix") is None
                else str(data.get("zfs_dataset_prefix")).rstrip("/")
            ),
            runtime_root=runtime_root,
            image_cache_root=image_cache_root,
            work_dir_host_root=work_dir_host_root,
            agent_state_root=agent_state_root,
            default_image=str(data.get("default_image", "ubuntu:22.04")),
            scheduler_config=scheduler_config,
            executor_config=executor_config,
            telemetry_config=telemetry_config,
            log_file=log_file,
            log_level=log_level,
            log_file_mode=log_file_mode,
            agent_worker_threads=int(data.get("agent_worker_threads", default_workers or 4)),
            extra={
                key: value
                for key, value in data.items()
                if key
                not in {
                    "agent_state_root",
                    "agent_worker_threads",
                    "default_image",
                    "enable_interceptor",
                    "enable_sandbox_network",
                    "executor",
                    "forwarder",
                    "image_cache_root",
                    "interceptor",
                    "interceptor_host",
                    "interceptor_port",
                    "log_file",
                    "log_file_mode",
                    "log_level",
                    "logging",
                    "max_workers",
                    "network",
                    "network_expected_sandboxes",
                    "runc",
                    "runc_paths",
                    "runtime",
                    "run_id",
                    "runtime_root",
                    "scheduler",
                    "storage_planes",
                    "storage_root",
                    "telemetry",
                    "work_dir_host_root",
                    "zfs_dataset_prefix",
                }
            },
        )


class Engine:
    """In-process Agent-CR engine.

    Use `Engine.start(config)` to construct; the engine context-manages
    cleanly so `with Engine.start() as eng: ...` works. For SDK convenience,
    `get_default_engine()` returns a lazily-started singleton.
    """

    def __init__(self, config: EngineConfig | str | os.PathLike[str] | Mapping[str, Any] | None = None) -> None:
        self._config = _coerce_engine_config(config)
        self._lock = threading.Lock()
        self._started = False
        self._owns_storage_dir = False
        self._tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._storage_root: Path | None = None
        self._runtime_root: Path | None = None
        self._image_cache_root: Path | None = None
        self._work_dir_host_root: Path | None = None
        self._agent_state_root: Path | None = None
        self._runtime: Runtime | None = None
        self._system: AgentCRSystem | None = None
        self._interceptor: AgentCRRequestInterceptorServer | None = None
        self._request_state_store: InMemoryRequestStateStore | None = None
        self._forwarder: SdkLLMForwarder | None = None
        self._forwarder_server = None
        self._forwarder_thread: threading.Thread | None = None
        self._forwarder_base_url: str | None = None
        self._agent_pool: ThreadPoolExecutor | None = None
        self._network_manager = None
        self._sandboxes: "dict[SandboxId, Sandbox]" = {}
        self._sandbox_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def start(cls, config: EngineConfig | str | os.PathLike[str] | Mapping[str, Any] | None = None) -> "Engine":
        engine = cls(config)
        engine._start()
        return engine

    @classmethod
    def connect(cls, *_, **__) -> "Engine":
        """Reserved for the future daemon mode. Raises today."""
        raise NotImplementedError(
            "Engine.connect() is not yet implemented; use Engine.start() for "
            "in-process use. Daemon mode will land in a follow-up PR."
        )

    def _start(self) -> None:
        with self._lock:
            if self._started:
                return
            cfg = self._config
            self._configure_logging(cfg)
            storage_root = cfg.storage_root
            if storage_root is None:
                self._tempdir = tempfile.TemporaryDirectory(prefix="agentcr-engine-")
                storage_root = Path(self._tempdir.name)
                self._owns_storage_dir = True
            storage_root.mkdir(parents=True, exist_ok=True)
            self._storage_root = storage_root
            self._request_state_store = InMemoryRequestStateStore()

            self._runtime_root = (cfg.runtime_root or (storage_root / "runtime")).expanduser().resolve()
            self._image_cache_root = (cfg.image_cache_root or (self._runtime_root / "images")).expanduser().resolve()
            self._work_dir_host_root = (
                cfg.work_dir_host_root or (self._runtime_root / "work")
            ).expanduser().resolve()
            self._agent_state_root = (
                cfg.agent_state_root or (self._runtime_root / "agents")
            ).expanduser().resolve()
            for path in (
                self._runtime_root,
                self._image_cache_root,
                self._work_dir_host_root,
                self._agent_state_root,
            ):
                path.mkdir(parents=True, exist_ok=True)

            if cfg.runtime == "runc" and cfg.enable_sandbox_network:
                from integrations.sandboxes.runtime.network import BenchmarkNetworkManager

                network_manager = BenchmarkNetworkManager()
                network_manager.configure(expected_sandboxes=cfg.network_expected_sandboxes)
                if cfg.enable_interceptor:
                    network_manager.ensure_bridge()
                self._network_manager = network_manager

            if cfg.runtime == "runc":
                self._system = self._build_runc_system(storage_root)
            else:
                self._system = build_default_system(
                    storage_root=storage_root,
                    runtime=cfg.runtime,
                    scheduler_config=cfg.scheduler_config,
                    executor_config=cfg.executor_config,
                    storage_config=cfg.storage_config,
                    runc_runtime_options=cfg.runc_options,
                    telemetry_config=cfg.telemetry_config,
                    request_state_store=self._request_state_store,
                )
            self._runtime = self._system.sandbox_manager

            if cfg.enable_interceptor:
                interceptor_host = self._effective_interceptor_host(cfg.interceptor_host)
                # Same architectural pattern as the benchmark harness:
                #   sandbox → AgentCRRequestInterceptorServer → forwarder → real LLM
                # The interceptor sees a single upstream URL (the forwarder).
                # The forwarder reads `X-Agent-Sandbox-Id` (set by the
                # interceptor's sandbox_id_resolver) and dispatches to the
                # per-sandbox real LLM URL registered via
                # `Engine.register_upstream`. The interceptor itself is
                # unchanged from the harness path.
                forwarder_server, forwarder = serve_sdk_llm_forwarder(
                    host=cfg.forwarder_host,
                    port=cfg.forwarder_port,
                )
                self._forwarder = forwarder
                self._forwarder_server = forwarder_server
                forwarder_host, forwarder_port = forwarder_server.server_address
                self._forwarder_base_url = f"http://{forwarder_host}:{int(forwarder_port)}"
                self._forwarder_thread = threading.Thread(
                    target=forwarder_server.serve_forever,
                    daemon=True,
                    name="agentcr-sdk-forwarder",
                )
                self._forwarder_thread.start()

                self._interceptor = AgentCRRequestInterceptorServer(
                    upstream_url=self._forwarder_base_url,
                    request_state_store=self._request_state_store,
                    response_gate_registry=self._system.response_gate_registry,
                    on_state_change=self._system.notify_interceptor_state_change,
                    on_response_ready=self._system.notify_live_response_ready,
                    sandbox_id_resolver=self._resolve_sandbox_id_for_request,
                    host=interceptor_host,
                    port=cfg.interceptor_port,
                )
                self._interceptor.start()
                logger.info(
                    "Engine started: runtime=%s storage_root=%s interceptor=%s forwarder=%s",
                    cfg.runtime,
                    storage_root,
                    self._interceptor.base_url,
                    self._forwarder_base_url,
                )
            else:
                logger.info(
                    "Engine started without interceptor: runtime=%s storage_root=%s",
                    cfg.runtime,
                    storage_root,
                )

            self._system.start()
            self._agent_pool = ThreadPoolExecutor(
                max_workers=max(1, cfg.agent_worker_threads),
                thread_name_prefix="agentcr-agent",
            )
            self._started = True

    def _configure_logging(self, cfg: EngineConfig) -> None:
        if cfg.log_file is None and not cfg.log_level:
            return
        level = getattr(logging, str(cfg.log_level or "INFO").upper(), logging.INFO)
        mode_value = str(cfg.log_file_mode or "append").lower()
        if mode_value not in {"append", "write", "a", "w"}:
            raise ValueError("EngineConfig.log_file_mode must be 'append' or 'write'")
        for logger_name in ("agent_cr", "integrations.sandboxes", "integrations.agents"):
            logging.getLogger(logger_name).setLevel(level)
        if cfg.log_file is None:
            return
        log_path = cfg.log_file.expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        existing = [
            handler
            for handler in logging.getLogger("agent_cr").handlers
            if getattr(handler, "_agentcr_engine_log_file", None) == str(log_path)
        ]
        if existing:
            for handler in existing:
                handler.setLevel(level)
            return
        mode = "w" if mode_value in {"write", "w"} else "a"
        handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        setattr(handler, "_agentcr_engine_log_file", str(log_path))
        for logger_name in ("agent_cr", "integrations.sandboxes", "integrations.agents"):
            logging.getLogger(logger_name).addHandler(handler)

    def _build_runc_system(self, storage_root: Path) -> AgentCRSystem:
        cfg = self._config
        assert self._request_state_store is not None
        assert self._runtime_root is not None
        paths = cfg.runc_paths or RuncRuntimePaths(
            state_root=self._runtime_root / "runtime-state",
            bundle_root=self._runtime_root / "bundles",
            checkpoint_root=self._runtime_root / "checkpoints",
            metadata_root=self._runtime_root / "sandbox-meta",
            zfs_dataset_prefix=self._resolve_zfs_dataset_prefix(),
        )
        self._ensure_zfs_parent_dataset(paths.zfs_dataset_prefix)
        telemetry_cfg = cfg.telemetry_config or TelemetryConfig(enabled=True)
        telemetry = build_configured_telemetry_sink(
            telemetry_cfg,
            default_attributes={"run_id": cfg.run_id} if cfg.run_id else None,
            keep_in_memory_fallback=True,
        )
        runtime = RuncRuntime(
            paths=paths,
            options=cfg.runc_options,
            telemetry=telemetry,
        )
        storage_cfg = cfg.storage_config or StorageConfig(root_dir=storage_root)
        storage = LocalCheckpointManager(
            storage_cfg,
            runtime_image_path_in_use=runtime.runtime_image_path_in_use,
        )
        inspector = RequestAwareSandboxInspector(
            EBPFSandboxInspector(),
            self._request_state_store,
        )
        executor_cfg = cfg.executor_config or ExecutorConfig()
        executor = CRExecutor(
            executor_cfg,
            DefaultCWorker(
                AdapterProcessCWorker(runtime),
                AdapterFileSystemCWorker(runtime),
                storage,
                runtime,
                telemetry=telemetry,
                step_workers=executor_cfg.resolved_composite_step_workers,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(runtime),
                AdapterFileSystemRWorker(runtime),
                storage,
                telemetry=telemetry,
                runtime=runtime,
            ),
            telemetry,
        )
        response_gate_registry = SandboxResponseGateRegistry()
        scheduler_cfg = cfg.scheduler_config or SchedulerConfig()
        return AgentCRSystem(
            scheduler=CRScheduler(
                scheduler_cfg,
                inspector,
                runtime,
                InMemorySchedulerStateStore(),
                telemetry,
                FaultToleranceCheckpointingPolicy(scheduler_cfg),
            ),
            executor=executor,
            storage=storage,
            inspector=inspector,
            runtime=runtime,
            telemetry=telemetry,
            request_state_store=self._request_state_store,
            response_gate_registry=response_gate_registry,
        )

    def _effective_interceptor_host(self, configured_host: str) -> str:
        if configured_host not in {"127.0.0.1", "localhost"}:
            return configured_host
        manager = self._network_manager
        if manager is None:
            return configured_host
        bridge_ip = getattr(manager, "bridge_ip", None)
        return str(bridge_ip) if bridge_ip else configured_host

    def _resolve_zfs_dataset_prefix(self) -> str:
        cfg = self._config
        if cfg.zfs_dataset_prefix:
            return cfg.zfs_dataset_prefix.rstrip("/")
        env_prefix = os.environ.get("AGENT_CR_ZFS_DATASET_PREFIX", "").strip()
        if env_prefix:
            return env_prefix.rstrip("/")
        env_pool = os.environ.get("AGENT_CR_ZPOOL_NAME", "").strip()
        if env_pool:
            return f"{env_pool}/agent-cr-sdk"
        try:
            result = subprocess.run(
                ["zpool", "list", "-H", "-o", "name"],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            return "agentcr/agent-cr-sdk"
        pools = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        for pool in pools:
            if pool.startswith("agentcr"):
                return f"{pool}/agent-cr-sdk"
        return f"{pools[0]}/agent-cr-sdk" if pools else "agentcr/agent-cr-sdk"

    def _ensure_zfs_parent_dataset(self, dataset_prefix: str) -> None:
        parts = [part for part in dataset_prefix.strip("/").split("/") if part]
        if len(parts) < 2:
            return
        current = parts[0]
        for part in parts[1:]:
            current = f"{current}/{part}"
            exists = subprocess.run(
                ["zfs", "list", "-H", "-o", "name", current],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode == 0
            if exists:
                continue
            subprocess.run(["zfs", "create", current], check=True)

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            sandboxes = list(self._sandboxes.values())
        # Kill sandboxes outside the lock to avoid deadlocks with their
        # internal locking.
        for sbx in sandboxes:
            try:
                sbx.kill()
            except Exception:
                logger.debug("Best-effort sandbox kill failed during engine stop", exc_info=True)
        with self._lock:
            if self._agent_pool is not None:
                self._agent_pool.shutdown(wait=False)
                self._agent_pool = None
            if self._interceptor is not None:
                try:
                    self._interceptor.stop()
                except Exception:
                    logger.exception("Interceptor stop failed")
                self._interceptor = None
            if self._forwarder_server is not None:
                try:
                    self._forwarder_server.shutdown()
                    self._forwarder_server.server_close()
                except Exception:
                    logger.exception("Forwarder shutdown failed")
                self._forwarder_server = None
            if self._forwarder_thread is not None:
                self._forwarder_thread.join(timeout=5.0)
                self._forwarder_thread = None
            self._forwarder = None
            self._forwarder_base_url = None
            if self._network_manager is not None:
                try:
                    self._network_manager.cleanup()
                except Exception:
                    logger.exception("Sandbox network cleanup failed")
                self._network_manager = None
            if self._system is not None:
                try:
                    self._system.stop()
                except Exception:
                    logger.exception("AgentCRSystem stop failed")
                self._system = None
            if self._tempdir is not None and self._owns_storage_dir:
                try:
                    self._tempdir.cleanup()
                except Exception:
                    logger.debug("Tempdir cleanup failed", exc_info=True)
                self._tempdir = None
            self._started = False

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Accessors used by Sandbox
    # ------------------------------------------------------------------

    @property
    def started(self) -> bool:
        return self._started

    @property
    def config(self) -> EngineConfig:
        return self._config

    @property
    def runtime(self) -> Runtime:
        if self._runtime is None:
            raise RuntimeError("engine is not started")
        return self._runtime

    @property
    def system(self) -> AgentCRSystem:
        if self._system is None:
            raise RuntimeError("engine is not started")
        return self._system

    @property
    def storage_root(self) -> Path:
        if self._storage_root is None:
            raise RuntimeError("engine is not started")
        return self._storage_root

    @property
    def runtime_root(self) -> Path:
        if self._runtime_root is None:
            raise RuntimeError("engine is not started")
        return self._runtime_root

    @property
    def image_cache_root(self) -> Path:
        if self._image_cache_root is None:
            raise RuntimeError("engine is not started")
        return self._image_cache_root

    @property
    def work_dir_host_root(self) -> Path:
        if self._work_dir_host_root is None:
            raise RuntimeError("engine is not started")
        return self._work_dir_host_root

    @property
    def agent_state_root(self) -> Path:
        if self._agent_state_root is None:
            raise RuntimeError("engine is not started")
        return self._agent_state_root

    @property
    def interceptor_base_url(self) -> str | None:
        if self._interceptor is None:
            return None
        return self._interceptor.base_url

    @property
    def agent_pool(self) -> ThreadPoolExecutor:
        if self._agent_pool is None:
            raise RuntimeError("engine is not started")
        return self._agent_pool

    # ------------------------------------------------------------------
    # Per-sandbox upstream URL bookkeeping (delegates to the SDK forwarder
    # so the per-sandbox routing lives in one place — same architectural
    # pattern as the harness, which keeps per-sandbox dispatch in the
    # BenchmarkLLMRouter rather than in the interceptor).
    # ------------------------------------------------------------------

    def register_upstream(self, sandbox_id: SandboxId, url: str) -> None:
        if not url or self._forwarder is None:
            return
        self._forwarder.register(str(sandbox_id), url)

    def unregister_upstream(self, sandbox_id: SandboxId) -> None:
        if self._forwarder is None:
            return
        self._forwarder.unregister(str(sandbox_id))

    def _lookup_upstream(self, sandbox_id: SandboxId) -> str | None:
        if self._forwarder is None:
            return None
        return self._forwarder.resolve(str(sandbox_id))

    def _resolve_sandbox_id_for_request(
        self,
        client_host: str | None,
        headers: dict[str, str],
        body: bytes,
    ) -> str | None:
        header_value = headers.get("X-Agent-Sandbox-Id", "").strip()
        if header_value:
            return header_value
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            payload = {}
        if isinstance(payload, dict):
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                candidate = metadata.get("sandbox_id")
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()

        if self._network_manager is not None:
            try:
                sandbox_id = self._network_manager.resolve_sandbox_id(client_host)
            except Exception:
                sandbox_id = None
            if sandbox_id is not None:
                return str(sandbox_id)

        with self._sandbox_lock:
            sandboxes = list(self._sandboxes.values())
        for sandbox in sandboxes:
            try:
                runtime_state = self.runtime.inspect_runtime(sandbox.sandbox_id)
            except Exception:
                continue
            metadata = runtime_state.metadata or {}
            for key in ("guest_ip", "bridge_ip", "host_ip"):
                value = metadata.get(key)
                if isinstance(value, str) and value == client_host:
                    return str(sandbox.sandbox_id)

        # SDK runc sandboxes currently default to host networking unless the
        # caller supplies a network namespace. That makes the HTTP peer
        # 127.0.0.1 for all in-sandbox agents, so fall back to the single
        # registered upstream case. Multi-sandbox LLM routing should use an
        # explicit header or per-sandbox network identity.
        registered = [
            sandbox
            for sandbox in sandboxes
            if self._lookup_upstream(sandbox.sandbox_id) is not None
        ]
        if len(registered) == 1:
            return str(registered[0].sandbox_id)
        return None

    @property
    def forwarder_base_url(self) -> str | None:
        return self._forwarder_base_url

    @property
    def network_bridge_ip(self) -> str | None:
        manager = self._network_manager
        if manager is None:
            return None
        return str(manager.bridge_ip)

    def allocate_network_lease(self, sandbox_id: SandboxId):
        manager = self._network_manager
        if manager is None:
            raise RuntimeError(
                "sandbox network is not enabled; start Engine with "
                "EngineConfig(enable_sandbox_network=True)"
            )
        lease = manager.allocate_lease(sandbox_id)
        manager.register_guest_ip(lease.guest_ip, sandbox_id)
        return lease

    def release_network_lease(self, sandbox_id: SandboxId) -> None:
        manager = self._network_manager
        if manager is None:
            return
        try:
            manager.release_lease(sandbox_id)
        except Exception:
            logger.exception("Failed to release sandbox network lease: %s", sandbox_id)

    def repair_network_lease(self, sandbox_id: SandboxId) -> bool:
        manager = self._network_manager
        if manager is None:
            return False
        try:
            return bool(manager.repair_lease(sandbox_id))
        except Exception:
            logger.exception("Failed to repair sandbox network lease: %s", sandbox_id)
            return False

    # ------------------------------------------------------------------
    # Sandbox registry — used so engine.stop() can clean up.
    # ------------------------------------------------------------------

    def _register_sandbox(self, sandbox: "Sandbox") -> None:
        with self._sandbox_lock:
            self._sandboxes[sandbox.sandbox_id] = sandbox

    def _unregister_sandbox(self, sandbox: "Sandbox") -> None:
        with self._sandbox_lock:
            self._sandboxes.pop(sandbox.sandbox_id, None)


# ---------------------------------------------------------------------------
# Default in-process engine — used when the user calls `Sandbox(...)` without
# explicitly providing one. Operators replace this with a connection to the
# daemon by calling `set_default_engine()` once.
# ---------------------------------------------------------------------------


_DEFAULT_ENGINE: Engine | None = None
_DEFAULT_ENGINE_LOCK = threading.Lock()


def get_default_engine(config: EngineConfig | str | os.PathLike[str] | Mapping[str, Any] | None = None) -> Engine:
    """Return the process-wide default engine, lazily starting one if needed.

    The first caller's `config` wins. Subsequent calls return the same Engine
    regardless of their `config` argument. Use `set_default_engine` to replace
    the default explicitly (e.g. from a daemon connection in the future).
    """
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        if _DEFAULT_ENGINE is None:
            _DEFAULT_ENGINE = Engine.start(config)
        return _DEFAULT_ENGINE


def set_default_engine(engine: Engine | None) -> Engine | None:
    """Replace the process-wide default engine. Returns the previous one
    (caller is responsible for stopping it if needed)."""
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        previous = _DEFAULT_ENGINE
        _DEFAULT_ENGINE = engine
    return previous


def shutdown_default_engine() -> None:
    """Stop and clear the default engine. Safe to call multiple times."""
    global _DEFAULT_ENGINE
    with _DEFAULT_ENGINE_LOCK:
        engine = _DEFAULT_ENGINE
        _DEFAULT_ENGINE = None
    if engine is not None:
        engine.stop()


__all__ = [
    "Engine",
    "EngineConfig",
    "get_default_engine",
    "set_default_engine",
    "shutdown_default_engine",
]

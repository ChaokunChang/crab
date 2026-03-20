#!/usr/bin/env python3
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import errno
import json
import logging
import shutil
import subprocess
import os
import shlex
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import (
    AgentCRRequestInterceptorServer,
    AgentCRSystem,
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CheckpointId,
    CheckpointManager,
    CheckpointManifest,
    CompositeRequestInterceptorHook,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    HostInspectorServiceClient,
    RemoteSandboxInspector,
    ExecutorConfig,
    CompositeTelemetrySink,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    JsonlTelemetrySink,
    LocalCheckpointManager,
    RequestInterceptorHook,
    RequestAwareSandboxInspector,
    RuncRuntime,
    RuncRuntimePaths,
    SandboxDescription,
    SandboxExecResult,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
    TelemetrySink,
    TelemetryRequestInterceptorHook,
)
from agent_cr.models import ArtifactPayload, utc_now
from agent_cr.host_inspector.fs_helper import LibbpfFilesystemMonitor
from agent_cr.host_inspector.runtime_resolver import RuntimeResolver
from agent_cr.host_inspector.server import HostInspectorDaemon, HostInspectorServer
from integrations.agents import BaseAgent, SandboxHandle, TaskConfig, TaskDescription, build_agent_registry
from integrations.llm_services import (
    default_llm_service_type_for_agent,
    serve_benchmark_llm_router,
    validate_llm_service_type,
)
from integrations.sandboxes.runtime import launcher as sandbox_launcher
from integrations.sandboxes.runtime import network as sandbox_network
from integrations.sandboxes.runtime import bundle as sandbox_bundle
from integrations.sandboxes.runtime import compose as sandbox_compose
from integrations.sandboxes.runtime import image as sandbox_image
from integrations.sandboxes.iflow import DOCKERFILE_PATH as IFLOW_DOCKERFILE_PATH
from integrations.sandboxes.simulated import DOCKERFILE_PATH as SIMULATED_DOCKERFILE_PATH
from benchmarks import core as benchmark_core
from benchmarks import support as benchmark_support

logger = logging.getLogger(__name__)

_HOST_INSPECTOR_HOST = "127.0.0.1"
_HOST_INSPECTOR_PORT = 9782
_DEFAULT_IMAGE_CACHE_ROOT = ROOT / ".cache" / "agent-cr" / "images"
_TERMNIUS_PROCESS_CAPABILITIES = [
    "CAP_AUDIT_WRITE",
    "CAP_CHOWN",
    "CAP_DAC_OVERRIDE",
    "CAP_FOWNER",
    "CAP_FSETID",
    "CAP_KILL",
    "CAP_MKNOD",
    "CAP_NET_BIND_SERVICE",
    "CAP_NET_RAW",
    "CAP_SETFCAP",
    "CAP_SETGID",
    "CAP_SETPCAP",
    "CAP_SETUID",
    "CAP_SYS_CHROOT",
]


def checkpoint_guard_from_inspector(inspector):
    def guard(job):
        try:
            snapshot = inspector.inspect(job.sandbox_id)
        except Exception:
            return True, None
        if snapshot.is_running:
            return True, None
        return False, "sandbox_not_running"

    return guard


def require_binaries() -> None:
    required = ["docker", "runc", "criu", "zfs", "zpool", "ip", "iptables", "sysctl"]
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"missing required binaries: {', '.join(missing)}")


def wait_for_http_json(url: str, *, timeout_s: float = 30.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {url}: {last_exc}")


def _zpool_exists(pool_name: str) -> bool:
    result = subprocess.run(
        ["zpool", "list", "-H", "-o", "name", pool_name],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _zfs_dataset_exists(dataset: str) -> bool:
    result = subprocess.run(
        ["zfs", "list", "-H", "-o", "name", dataset],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0

@dataclass(frozen=True)
class AgentSandboxImage:
    agent_type: str
    image_tag: str
    exported_rootfs: Path
    image_defaults: sandbox_image.ImageRuntimeDefaults


class RealHostScenarioHarness:
    @property
    def _active_runtime(self):
        if self.runtime is not None:
            return self.runtime
        if self.system is None:
            return None
        runtime = getattr(self.system, "runtime", None)
        if runtime is not None:
            return runtime
        return getattr(self.system, "sandbox_manager", None)

    @property
    def sandbox_manager(self):
        return self.runtime

    @sandbox_manager.setter
    def sandbox_manager(self, value) -> None:
        self.runtime = value

    def __init__(
        self,
        *,
        provider: str,
        transfer_delay_ms: float,
        scheduler_config: SchedulerConfig,
        scheduler_policy,
        checkpoint_manager_factory,
        max_workers: int,
        auto_cr: bool = False,
        work_dir_host_root: Path | None = None,
        telemetry_output: Path | None = None,
        zpool_size: str = "10G",
        zpool_name: str | None = None,
        zpool_image: Path | None = None,
        reuse_zpool: bool = False,
        image_cache_root: Path | None = None,
    ) -> None:
        self.provider = provider
        self.transfer_delay_ms = transfer_delay_ms
        self.scheduler_config = scheduler_config
        self.scheduler_policy = scheduler_policy
        self.checkpoint_manager_factory = checkpoint_manager_factory
        self.max_workers = max_workers
        self.auto_cr = auto_cr
        self.work_dir_host_root = work_dir_host_root
        self.telemetry_output = telemetry_output
        self.zpool_size = zpool_size
        self.configured_zpool_name = zpool_name
        self.configured_zpool_image = zpool_image
        self.reuse_zpool = reuse_zpool
        self.image_cache_root = (image_cache_root or _DEFAULT_IMAGE_CACHE_ROOT).resolve()
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.root: Path | None = None
        self.pool_name = ""
        self.runtime_state_root: Path | None = None
        self.host_inspector_url: str = ""
        self._host_inspector_server: HostInspectorServer | None = None
        self.host_inspector_client: HostInspectorServiceClient = None
        self.telemetry: TelemetrySink | None = None
        self.request_state_store: InMemoryRequestStateStore | None = None
        self.base_inspector: EBPFSandboxInspector | None = None
        self.inspector: RequestAwareSandboxInspector | None = None
        self.runtime: RuncRuntime | None = None
        self.storage: CheckpointManager | None = None
        self.executor: CRExecutor | None = None
        self.system: AgentCRSystem | None = None
        self.interceptor: AgentCRRequestInterceptorServer | None = None
        self.interceptor_hook = CompositeRequestInterceptorHook()
        self.llm_server = None
        self.llm_thread: threading.Thread | None = None
        self.sandboxes: list[SandboxHandle] = []
        self._sandbox_by_id: dict[SandboxId, SandboxHandle] = {}
        self.network_manager = sandbox_network.BenchmarkNetworkManager()
        self._compose_image_tags: set[str] = set()
        self._agent_registry = build_agent_registry()
        self._sandbox_images: dict[str, AgentSandboxImage] = {}
        self._sandbox_image_lock = threading.Lock()
        self._task_executor = ThreadPoolExecutor(max_workers=max(1, self.max_workers))
        self._zpool_image_path: Path | None = None

    @property
    def benchmark_bridge_ip(self) -> str:
        return self.network_manager.bridge_ip

    def _start_host_inspector_server(self) -> str:
        assert self.runtime_state_root is not None
        if self._host_inspector_server is not None:
            return f"http://{_HOST_INSPECTOR_HOST}:{self._host_inspector_server.port}"

        self.runtime_state_root.mkdir(parents=True, exist_ok=True)
        daemon = HostInspectorDaemon(
            resolver=RuntimeResolver(runc_state_root=self.runtime_state_root),
            fs_monitor=LibbpfFilesystemMonitor(),
        )
        try:
            self._host_inspector_server = HostInspectorServer(
                host=_HOST_INSPECTOR_HOST,
                port=_HOST_INSPECTOR_PORT,
                daemon=daemon,
            )
        except OSError as exc:
            if exc.errno != errno.EADDRINUSE:
                raise
            logger.warning(
                "Host inspector port %d is already in use; falling back to an ephemeral port",
                _HOST_INSPECTOR_PORT,
            )
            self._host_inspector_server = HostInspectorServer(
                host=_HOST_INSPECTOR_HOST,
                port=0,
                daemon=daemon,
            )
        logger.info(
            "Starting host inspector server in-process host=%s port=%d runc_state_root=%s",
            _HOST_INSPECTOR_HOST,
            self._host_inspector_server.port,
            self.runtime_state_root,
        )
        self._host_inspector_server.start()
        url = f"http://{_HOST_INSPECTOR_HOST}:{self._host_inspector_server.port}"
        try:
            wait_for_http_json(f"{url}/healthz")
        except Exception:
            self._stop_host_inspector_server()
            raise
        return url

    def _stop_host_inspector_server(self) -> None:
        server = self._host_inspector_server
        self._host_inspector_server = None
        if server is None:
            return
        server.stop()

    def __enter__(self) -> "RealHostScenarioHarness":
        require_binaries()
        self.network_manager.configure()
        logger.info(
            "Selected benchmark network cidr=%s bridge_ip=%s",
            self.network_manager.network_cidr,
            self.network_manager.bridge_ip,
        )
        self._tmpdir = tempfile.TemporaryDirectory(prefix="agent_cr_scenario_bench_")
        # self.root = Path("/root/workspace/agent-cr/logs/tmp/agent_cr_bench")
        bench_dir = os.environ.get("AGENTCR_BENCH_DIR", None)
        if bench_dir and bench_dir.lower() not in ['tmpdir', 'tmp']:
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.root = (Path(bench_dir).expanduser().resolve() / suffix)
        else:
            self.root = Path(self._tmpdir.name)
        unique_suffix = uuid.uuid4().hex[:10]
        self.pool_name = self.configured_zpool_name or f"agentcrbench{unique_suffix}"
        self.runtime_state_root = self.root / "runtime-state"
        self.host_inspector_url = self._start_host_inspector_server()
        self.llm_server = serve_benchmark_llm_router(host="127.0.0.1", port=0)
        self.llm_thread = threading.Thread(target=self.llm_server.serve_forever, daemon=True)
        self.llm_thread.start()
        wait_for_http_json(f"http://127.0.0.1:{self.llm_server.server_address[1]}/healthz")
        self._ensure_zpool()

        telemetry_sinks: list[TelemetrySink] = [InMemoryTelemetrySink()]
        telemetry_path = self.telemetry_output or (self.root / "telemetry.jsonl")
        telemetry_sinks.append(JsonlTelemetrySink(telemetry_path))
        self.telemetry = CompositeTelemetrySink(telemetry_sinks)
        self.request_state_store = InMemoryRequestStateStore()

        self.host_inspector_client = HostInspectorServiceClient(self.host_inspector_url)
        self.base_inspector = RemoteSandboxInspector(self.host_inspector_client)
        self.inspector = RequestAwareSandboxInspector(self.base_inspector, self.request_state_store)
        self.runtime = RuncRuntime(
            paths=RuncRuntimePaths(
                state_root=self.runtime_state_root,
                bundle_root=self.root / "bundles",
                checkpoint_root=self.root / "checkpoints",
                metadata_root=self.root / "sandbox-meta",
                zfs_dataset_prefix=f"{self.pool_name}/agent-cr",
            ),
            host_inspector_client=self.host_inspector_client,
            telemetry=self.telemetry,
        )
        base_storage = LocalCheckpointManager(StorageConfig(root_dir=self.root / "storage"))
        self.storage = self.checkpoint_manager_factory(base_storage)
        self.executor = CRExecutor(
            ExecutorConfig(max_workers=max(1, self.max_workers)),
            DefaultCWorker(
                AdapterProcessCWorker(self.runtime),
                AdapterFileSystemCWorker(self.runtime),
                self.storage,
                self.runtime,
                checkpoint_guard=checkpoint_guard_from_inspector(self.inspector),
                telemetry=self.telemetry,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(self.runtime),
                AdapterFileSystemRWorker(self.runtime),
                self.storage,
                telemetry=self.telemetry,
            ),
            self.telemetry,
        )
        self.system = AgentCRSystem(
            scheduler=CRScheduler(
                self.scheduler_config,
                self.inspector,
                self.runtime,
                InMemorySchedulerStateStore(),
                self.telemetry,
                self.scheduler_policy,
            ),
            executor=self.executor,
            storage=self.storage,
            inspector=self.inspector,
            runtime=self.runtime,
            telemetry=self.telemetry,
            request_state_store=self.request_state_store,
            relaunch_handler=self._relaunch_sandbox if self.auto_cr else None,
            extra_checkpoint_metadata_provider=self._llm_service_checkpoint_metadata,
            restore_metadata_handler=self._restore_llm_service_state,
            recovery_delay_seconds=self.transfer_delay_ms / 1000.0 if self.auto_cr else 0.0,
        )
        self.interceptor_hook.add_hook(TelemetryRequestInterceptorHook(self.telemetry))
        self.interceptor = AgentCRRequestInterceptorServer(
            upstream_url=f"http://127.0.0.1:{self.llm_server.server_address[1]}",
            request_state_store=self.request_state_store,
            hook=self.interceptor_hook,
            telemetry=self.telemetry,
            on_state_change=self.system.notify_interceptor_state_change,
            response_gate_registry=self.system.response_gate_registry,
            sandbox_id_resolver=self.resolve_interceptor_sandbox_id,
            host="0.0.0.0",
            port=0,
        )
        self.interceptor.start()
        wait_for_http_json(f"http://127.0.0.1:{self.interceptor.port}/healthz")
        if self.auto_cr:
            self.system.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # input("wait for signal to cleanup")
        for sandbox in self.sandboxes:
            if sandbox.task_run is not None:
                sandbox.task_run.request_stop()
        if self.system is not None and self.auto_cr:
            self.system.stop()
        if self.interceptor is not None:
            self.interceptor.stop()
        if self.runtime is not None:
            for sandbox in self.sandboxes:
                self.runtime.delete_runtime(sandbox.sandbox_id, force=True, ignore_missing=True)
        self._task_executor.shutdown(wait=True, cancel_futures=True)
        if self.executor is not None:
            self.executor.shutdown()
        self.network_manager.cleanup()
        for sandbox in self.sandboxes:
            if self.llm_server is not None:
                self.llm_server.benchmark_llm_router.unregister_sandbox(str(sandbox.sandbox_id))  # type: ignore[attr-defined]
        # for image in self._sandbox_images.values():
            # subprocess.run(
            #     ["docker", "rmi", "-f", image.image_tag],
            #     check=False,
            #     stdout=subprocess.DEVNULL,
            #     stderr=subprocess.DEVNULL,
            # )
        # for image_tag in sorted(self._compose_image_tags):
            # subprocess.run(
            #     ["docker", "rmi", "-f", image_tag],
            #     check=False,
            #     stdout=subprocess.DEVNULL,
            #     stderr=subprocess.DEVNULL,
            # )
        if self.pool_name:
            dataset = f"{self.pool_name}/agent-cr"
            subprocess.run(
                ["zfs", "destroy", "-r", dataset],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not self.reuse_zpool:
                subprocess.run(
                    ["zpool", "destroy", "-f", self.pool_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        if self.llm_server is not None:
            self.llm_server.shutdown()
            self.llm_server.server_close()
        if self.llm_thread is not None:
            self.llm_thread.join(timeout=5.0)
        self._stop_host_inspector_server()
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def _ensure_zpool(self) -> None:
        assert self.root is not None
        dataset = f"{self.pool_name}/agent-cr"
        zpool_image_path = self.configured_zpool_image or (self.root / "zpool.img")
        self._zpool_image_path = zpool_image_path
        if self.reuse_zpool:
            if _zpool_exists(self.pool_name):
                logger.info("Reusing existing benchmark zpool name=%s", self.pool_name)
            else:
                zpool_image_path.parent.mkdir(parents=True, exist_ok=True)
                if not zpool_image_path.exists():
                    logger.info(
                        "Creating reusable benchmark zpool image path=%s size=%s",
                        zpool_image_path,
                        self.zpool_size,
                    )
                    subprocess.run(["truncate", "-s", self.zpool_size, str(zpool_image_path)], check=True)
                logger.info("Creating reusable benchmark zpool name=%s image=%s", self.pool_name, zpool_image_path)
                subprocess.run(["zpool", "create", "-f", self.pool_name, str(zpool_image_path)], check=True)
            if _zfs_dataset_exists(dataset):
                subprocess.run(
                    ["zfs", "destroy", "-r", dataset],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            subprocess.run(["zfs", "create", dataset], check=True)
            return

        subprocess.run(["truncate", "-s", self.zpool_size, str(zpool_image_path)], check=True)
        subprocess.run(["zpool", "create", "-f", self.pool_name, str(zpool_image_path)], check=True)
        subprocess.run(["zfs", "create", dataset], check=True)

    def get_agent_class(self, agent_type: str) -> type[BaseAgent]:
        try:
            return self._agent_registry[agent_type]
        except KeyError as exc:
            raise ValueError(f"unsupported benchmark agent type: {agent_type}") from exc

    def build_task_run(
        self,
        agent_type: str,
        sandbox: SandboxHandle,
        task_description: TaskDescription,
        task_config: TaskConfig,
    ) -> BaseAgent:
        assert self.root is not None
        return self.get_agent_class(agent_type)(
            sandbox,
            task_description,
            task_config,
            runtime_state_root=self.runtime_state_root,
            runtime=self.runtime,
            agent_host_dir=self.root / agent_type / str(sandbox.sandbox_id),
            llm_base_url=sandbox.llm_base_url,
        )

    def _agent_requires_benchmark_network(self, agent_type: str) -> bool:
        return bool(self.get_agent_class(agent_type).requires_network_namespace)

    def resolve_llm_service_type(self, *, agent_type: str, llm_service_type: str | None) -> str:
        resolved = llm_service_type or default_llm_service_type_for_agent(agent_type)
        validate_llm_service_type(provider=self.provider, llm_service_type=resolved)
        return resolved

    def _sandbox_dockerfile_path(self, agent_type: str) -> Path:
        if agent_type == "iflow":
            return IFLOW_DOCKERFILE_PATH
        if agent_type == "simulated":
            return SIMULATED_DOCKERFILE_PATH
        raise ValueError(f"unsupported benchmark agent type for sandbox image: {agent_type}")

    def _sandbox_build_context_path(self, agent_type: str) -> Path:
        if agent_type == "iflow":
            return IFLOW_DOCKERFILE_PATH.parent
        if agent_type == "simulated":
            return SIMULATED_DOCKERFILE_PATH.parent
        raise ValueError(f"unsupported benchmark agent type for sandbox image: {agent_type}")

    def _sandbox_image_tag(self, agent_type: str) -> str:
        return f"agent-cr-{sandbox_image.docker_tag_component(agent_type)}-bench:workspace"

    def ensure_sandbox_image(self, agent_type: str) -> AgentSandboxImage:
        assert self.root is not None
        with self._sandbox_image_lock:
            cached = self._sandbox_images.get(agent_type)
            if cached is not None:
                return cached
            image_tag = self._sandbox_image_tag(agent_type)
            sandbox_image.build_image(
                tag=image_tag,
                build_context=self._sandbox_build_context_path(agent_type),
                dockerfile_path=self._sandbox_dockerfile_path(agent_type),
                telemetry=self.telemetry,
            )
            image_defaults = sandbox_image.inspect_image_runtime_defaults(
                tag=image_tag,
                cache_root=self.image_cache_root,
                telemetry=self.telemetry,
            )
            exported_rootfs = sandbox_image.export_image_rootfs(
                tag=image_tag,
                output_dir=self.root / "image" / agent_type,
                cache_root=self.image_cache_root,
                telemetry=self.telemetry,
            )
            built = AgentSandboxImage(
                agent_type=agent_type,
                image_tag=image_tag,
                exported_rootfs=exported_rootfs,
                image_defaults=image_defaults,
            )
            self._sandbox_images[agent_type] = built
            return built

    def get_sandbox_handle(self, sandbox_id: str | SandboxId) -> SandboxHandle:
        target = SandboxId(str(sandbox_id))
        return self._sandbox_by_id[target]

    def load_dataset(self, path: Path) -> list[benchmark_support.BenchmarkTaskRecord]:
        return benchmark_core.load_task_dataset(path)

    def _resolve_dataset_service_config(
        self,
        dataset_root: Path,
        raw_value: object,
    ) -> dict[str, object] | None:
        if raw_value is None:
            return None
        if not isinstance(raw_value, dict):
            raise ValueError(f"llm_service_config must be an object, got {raw_value!r}")
        config = dict(raw_value)
        trace_path = config.get("trace_path")
        if isinstance(trace_path, str):
            config["trace_path"] = str((dataset_root / trace_path).resolve())
        return config

    def select_task_record(
        self,
        dataset: list[benchmark_support.BenchmarkTaskRecord] | None,
        *,
        sandbox_index: int,
        default_agent_type: str,
        default_llm_service_type: str | None,
        default_task_description: TaskDescription,
        default_task_config: TaskConfig,
    ) -> benchmark_support.BenchmarkTaskRecord:
        return benchmark_core.select_task_record(
            dataset,
            sandbox_index=sandbox_index,
            default_agent_type=default_agent_type,
            default_llm_service_type=default_llm_service_type,
            default_task_description=default_task_description,
            default_task_config=default_task_config,
        )

    def launch_sandbox(
        self,
        sandbox_name: str,
        *,
        agent_type: str = "simulated",
        llm_service_type: str | None = None,
        llm_service_config: dict[str, object] | None = None,
        task_description: TaskDescription | None = None,
        task_config: TaskConfig | None = None,
    ) -> SandboxHandle:
        assert self.root is not None
        assert self.base_inspector is not None
        assert self.system is not None
        assert self.interceptor is not None

        if (task_description is None) != (task_config is None):
            raise ValueError("task_description and task_config must be provided together")

        resolved_task_description = task_description or TaskDescription("")
        resolved_task_config = task_config or TaskConfig()
        resolved_llm_service_type = self.resolve_llm_service_type(agent_type=agent_type, llm_service_type=llm_service_type)
        sandbox_image = self.ensure_sandbox_image(agent_type)
        network_lease = (
            self.network_manager.allocate_lease(SandboxId(sandbox_name))
            if self._agent_requires_benchmark_network(agent_type)
            else None
        )
        handle, work_dir_host_path = self._prepare_sandbox_handle(
            sandbox_name,
            interceptor_host=self.network_manager.bridge_ip if network_lease is not None else "127.0.0.1",
            network_lease=network_lease,
            agent_type=agent_type,
            llm_service_type=resolved_llm_service_type,
            llm_service_config=llm_service_config,
            image_defaults=sandbox_image.image_defaults,
            image_rootfs_dir=sandbox_image.exported_rootfs,
        )
        task_run = self.build_task_run(agent_type, handle, resolved_task_description, resolved_task_config)
        task_run.prepare_sandbox()
        task_run.configure_bundle()
        sandbox_id = handle.sandbox_id
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        if network_lease is not None:
            self.network_manager.register_guest_ip(network_lease.guest_ip, sandbox_id)
        launch_metadata = {
            "sandbox_id": sandbox_name,
            "bundle_path": str(handle.bundle_dir),
            "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
            "rootfs_init_dirs": task_run.rootfs_init_dirs(),
            "rootfs_copy_paths": [{"source": str(sandbox_image.exported_rootfs), "destination": "/"}],
            **task_run.extra_launch_metadata(),
            **handle.launch_metadata.get("runtime", {}),
        }
        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError("runtime is not initialized")
        runtime.launch("runc", launch_metadata)
        task_run.wait_for_task_ready()
        logger.info(
            "Launched benchmark sandbox name=%s sandbox_id=%s status_host=%s status_port=%d auto_cr=%s agent_type=%s",
            sandbox_name,
            sandbox_id,
            handle.status_host,
            handle.status_port,
            self.auto_cr,
            agent_type,
        )
        return handle

    def launch_task(
        self,
        agent_type: str,
        task_description: TaskDescription,
        task_config: TaskConfig,
        sandbox_id: str,
    ) -> BaseAgent:
        handle = self.get_sandbox_handle(sandbox_id)
        if handle.task_future is not None and not handle.task_future.done():
            if handle.task_run is not None:
                handle.task_run.request_stop()
            handle.task_future.cancel()
        handle.agent_type = agent_type
        handle.task_description = task_description
        handle.task_config = task_config
        task_run = self.build_task_run(agent_type, handle, task_description, task_config)
        handle.task_run = task_run
        handle.task_future = self._task_executor.submit(task_run.perform_task)
        return task_run

    def launch_sandbox_and_task(
        self,
        sandbox_name: str,
        *,
        agent_type: str,
        llm_service_type: str | None = None,
        llm_service_config: dict[str, object] | None = None,
        task_description: TaskDescription,
        task_config: TaskConfig,
    ) -> SandboxHandle:
        handle = self.launch_sandbox(
            sandbox_name,
            agent_type=agent_type,
            llm_service_type=llm_service_type,
            llm_service_config=llm_service_config,
            task_description=task_description,
            task_config=task_config,
        )
        self.launch_task(agent_type, task_description, task_config, str(handle.sandbox_id))
        return handle

    def launch_task_record(
        self,
        sandbox_name: str,
        task_record: benchmark_support.BenchmarkTaskRecord,
    ) -> SandboxHandle:
        if task_record.docker_compose_file is not None:
            handle = self.launch_sandbox_from_docker_compose_file(
                task_record.docker_compose_file,
                task_record.env_file,
                sandbox_name=sandbox_name,
                service_name=task_record.service_name,
                agent_type=task_record.agent_type,
                llm_service_type=task_record.llm_service_type,
                llm_service_config=task_record.llm_service_config,
                task_description=task_record.task_description,
                task_config=task_record.task_config,
                task_root=task_record.task_root,
            )
        else:
            handle = self.launch_sandbox_and_task(
                sandbox_name,
                agent_type=task_record.agent_type,
                llm_service_type=task_record.llm_service_type,
                llm_service_config=task_record.llm_service_config,
                task_description=task_record.task_description,
                task_config=task_record.task_config,
            )
        handle.launch_metadata["benchmark"] = {
            "task_id": task_record.task_id or (task_record.task_root.name if task_record.task_root is not None else sandbox_name),
            "trace_response_count": task_record.trace_response_count,
            "trace_malformed_line_count": task_record.trace_malformed_line_count,
            "llm_service_config": None if task_record.llm_service_config is None else dict(task_record.llm_service_config),
        }
        return handle

    def launch_sandbox_from_docker_compose_file(
        self,
        compose_file: Path,
        env_file: Path | None,
        *,
        sandbox_name: str,
        service_name: str | None = None,
        status_host: str | None = None,
        status_port: int | None = None,
        agent_type: str | None = None,
        llm_service_type: str | None = None,
        llm_service_config: dict[str, object] | None = None,
        task_description: TaskDescription | None = None,
        task_config: TaskConfig | None = None,
        task_root: Path | None = None,
    ) -> SandboxHandle:
        compose_env = self._build_termnius_compose_env(
            sandbox_name=sandbox_name,
            task_root=task_root,
        )
        service_name, service = sandbox_compose.load_compose_service(
            compose_file=compose_file,
            env_file=env_file,
            extra_env=compose_env,
            service_name=service_name,
        )
        network_lease = self.network_manager.allocate_lease(SandboxId(sandbox_name))
        handle, work_dir_host_path = self._prepare_sandbox_handle(
            sandbox_name,
            interceptor_host=self.network_manager.bridge_ip,
            network_lease=network_lease,
            agent_type=agent_type or "simulated",
            llm_service_type=self.resolve_llm_service_type(
                agent_type=agent_type or "simulated",
                llm_service_type=llm_service_type,
            ),
            llm_service_config=llm_service_config,
            status_port=status_port,
            status_host=status_host if status_host is not None else network_lease.guest_ip,
        )
        prelaunch_task_run = None
        if agent_type is not None and task_description is not None and task_config is not None:
            handle.task_description = task_description
            handle.task_config = task_config
            prelaunch_task_run = self.build_task_run(agent_type, handle, task_description, task_config)
            prelaunch_task_run.prepare_sandbox()
        translation = sandbox_compose.translate_compose_service(
            compose_file=compose_file,
            service_name=service_name,
            service=service,
            bundle_dir=handle.bundle_dir,
            sandbox_id=str(handle.sandbox_id),
            work_dir_host_path=work_dir_host_path,
            compose_image_root=self.image_cache_root,
            compose_image_tags=self._compose_image_tags,
            telemetry=self.telemetry,
        )
        assert self.base_inspector is not None
        assert self.system is not None
        handle.launch_source = "compose"
        handle.launch_metadata["runtime"] = dict(translation.runtime_launch_metadata)
        handle.launch_metadata["compose"] = dict(translation.compose_launch_metadata)
        if task_root is not None:
            handle.launch_metadata["task_root"] = str(task_root)
            self._extend_termnius_rootfs_materialization(handle.launch_metadata["runtime"], task_root)
        self._ensure_termnius_dns_materialization(handle.launch_metadata["runtime"])
        self._configure_termnius_bundle_privileges(handle.bundle_dir)
        if prelaunch_task_run is not None:
            runtime_metadata = handle.launch_metadata["runtime"]
            runtime_metadata["rootfs_init_dirs"] = sorted(
                set(runtime_metadata.get("rootfs_init_dirs", [])) | set(prelaunch_task_run.rootfs_init_dirs())
            )
            runtime_metadata.update(prelaunch_task_run.extra_launch_metadata())
            prelaunch_task_run.configure_bundle()
        self.network_manager.register_guest_ip(network_lease.guest_ip, handle.sandbox_id)
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=handle.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        runtime = self._active_runtime
        if runtime is None:
            raise RuntimeError("runtime is not initialized")
        runtime.launch("runc", handle.launch_metadata["runtime"])
        handle.last_status = {}
        if agent_type is not None and task_description is not None and task_config is not None:
            self.launch_task(agent_type, task_description, task_config, str(handle.sandbox_id))
            if handle.task_run is not None:
                try:
                    handle.task_run.wait_for_task_ready()
                except RuntimeError:
                    pass
        return handle

    def _build_termnius_compose_env(
        self,
        *,
        sandbox_name: str,
        task_root: Path | None,
    ) -> dict[str, str]:
        assert self.root is not None
        task_id = "task" if task_root is None else task_root.name
        host_logs_root = self.root / "termnius-logs" / sandbox_name
        task_logs_path = host_logs_root / "logs"
        task_agent_logs_path = host_logs_root / "agent-logs"
        task_logs_path.mkdir(parents=True, exist_ok=True)
        task_agent_logs_path.mkdir(parents=True, exist_ok=True)
        image_component = sandbox_image.docker_tag_component(f"{task_id}-{sandbox_name}")
        return {
            "T_BENCH_CONTAINER_LOGS_PATH": "/logs",
            "T_BENCH_CONTAINER_AGENT_LOGS_PATH": "/agent-logs",
            "T_BENCH_TASK_LOGS_PATH": str(task_logs_path),
            "T_BENCH_TASK_AGENT_LOGS_PATH": str(task_agent_logs_path),
            "T_BENCH_TASK_DOCKER_CLIENT_CONTAINER_NAME": f"agent-cr-{image_component}",
            "T_BENCH_TASK_DOCKER_CLIENT_IMAGE_NAME": f"agent-cr-termnius-{image_component}",
            "T_BENCH_TEST_DIR": "/tests",
        }

    def _extend_termnius_rootfs_materialization(
        self,
        runtime_metadata: dict[str, object],
        task_root: Path,
    ) -> None:
        tests_dir = task_root / "tests"
        run_tests = task_root / "run-tests.sh"
        if not run_tests.is_file():
            raise FileNotFoundError(f"missing task run-tests.sh: {run_tests}")
        copy_paths = list(runtime_metadata.get("rootfs_copy_paths", []))
        if tests_dir.is_dir():
            copy_paths.append({"source": str(tests_dir), "destination": "/tests"})
        copy_paths.append({"source": str(run_tests), "destination": "/tests/run-tests.sh"})
        runtime_metadata["rootfs_copy_paths"] = copy_paths
        init_dirs = set(runtime_metadata.get("rootfs_init_dirs", []))
        init_dirs.add("tests")
        runtime_metadata["rootfs_init_dirs"] = sorted(init_dirs)

    def _ensure_termnius_dns_materialization(self, runtime_metadata: dict[str, object]) -> None:
        host_resolv_conf = Path("/run/systemd/resolve/resolv.conf")
        if not host_resolv_conf.is_file():
            host_resolv_conf = Path("/etc/resolv.conf")
        if not host_resolv_conf.is_file():
            return
        copy_paths = list(runtime_metadata.get("rootfs_copy_paths", []))
        resolv_item = {"source": str(host_resolv_conf), "destination": "/etc/resolv.conf"}
        if resolv_item not in copy_paths:
            copy_paths.append(resolv_item)
        runtime_metadata["rootfs_copy_paths"] = copy_paths

    def _configure_termnius_bundle_privileges(self, bundle_dir: Path) -> None:
        config_path = bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        process = cfg.get("process", {})
        if not isinstance(process, dict):
            raise ValueError(f"unsupported bundle process config in {config_path}")
        capabilities = {
            "bounding": list(_TERMNIUS_PROCESS_CAPABILITIES),
            "effective": list(_TERMNIUS_PROCESS_CAPABILITIES),
            "permitted": list(_TERMNIUS_PROCESS_CAPABILITIES),
            "inheritable": list(_TERMNIUS_PROCESS_CAPABILITIES),
            "ambient": list(_TERMNIUS_PROCESS_CAPABILITIES),
        }
        process["capabilities"] = capabilities
        process["noNewPrivileges"] = False
        cfg["process"] = process
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def add_interceptor_hook(self, hook: RequestInterceptorHook) -> None:
        self.interceptor_hook.add_hook(hook)
        logger.debug("Registered interceptor hook %s", type(hook).__name__)

    def resolve_interceptor_sandbox_id(
        self,
        client_host: str | None,
        headers: dict[str, str],
        body: bytes,
    ) -> str | None:
        _ = body
        sandbox_id_from_header = str(headers.get("X-Agent-Sandbox-Id", "")).strip()
        if sandbox_id_from_header:
            return sandbox_id_from_header
        sandbox_id = self.network_manager.resolve_sandbox_id(client_host)
        if sandbox_id is None:
            return None
        return str(sandbox_id)

    def drain_request_state_changes(self) -> int:
        if self.request_state_store is None:
            return 0
        drained = 0
        while self.request_state_store.wait_for_change(timeout=0.0) is not None:
            drained += 1
        if drained:
            logger.info("Drained %d queued request-state changes", drained)
        return drained

    def set_snapshot_metadata(self, sandbox: SandboxHandle, **metadata: object) -> None:
        self.set_snapshot_metadata_by_id(sandbox.sandbox_id, **metadata)

    def set_snapshot_metadata_by_id(self, sandbox_id: SandboxId, **metadata: object) -> None:
        assert self.base_inspector is not None
        snapshot = self.base_inspector.inspect(sandbox_id)
        merged = {**snapshot.metadata, **metadata}
        self.base_inspector.upsert_snapshot(
            replace(
                snapshot,
                observed_at=utc_now(),
                metadata=merged,
            )
        )

    def clear_snapshot_metadata(self, sandbox: SandboxHandle, *keys: str) -> None:
        self.clear_snapshot_metadata_by_id(sandbox.sandbox_id, *keys)

    def clear_snapshot_metadata_by_id(self, sandbox_id: SandboxId, *keys: str) -> None:
        assert self.base_inspector is not None
        snapshot = self.base_inspector.inspect(sandbox_id)
        metadata = dict(snapshot.metadata)
        for key in keys:
            metadata.pop(key, None)
        self.base_inspector.upsert_snapshot(
            replace(
                snapshot,
                observed_at=utc_now(),
                metadata=metadata,
            )
        )

    def checkpoint_manual(self, sandbox: SandboxHandle, leave_running: bool=False):
        assert self.system is not None
        if sandbox.launch_source not in {"runc", "compose"}:
            raise RuntimeError(f"checkpoint_manual unsupported for launch_source={sandbox.launch_source}")
        logger.debug("Benchmark requesting checkpoint_manual for sandbox=%s", sandbox.sandbox_id)
        return self.system.checkpoint_once(sandbox.sandbox_id, leave_running=leave_running)

    def checkpoint_if_due(self, sandbox: SandboxHandle):
        assert self.system is not None
        if sandbox.launch_source not in {"runc", "compose"}:
            raise RuntimeError(f"checkpoint_if_due unsupported for launch_source={sandbox.launch_source}")
        logger.debug("Benchmark requesting checkpoint_if_due for sandbox=%s", sandbox.sandbox_id)
        return self.system.checkpoint_if_due(sandbox.sandbox_id)

    def restore_once(self, sandbox: SandboxHandle, checkpoint_id: CheckpointId):
        assert self.system is not None
        if sandbox.launch_source not in {"runc", "compose"}:
            raise RuntimeError(f"restore_once unsupported for launch_source={sandbox.launch_source}")
        logger.info(
            "Benchmark requesting restore sandbox=%s checkpoint=%s transfer_delay_ms=%.1f",
            sandbox.sandbox_id,
            checkpoint_id,
            self.transfer_delay_ms,
        )
        if self.transfer_delay_ms > 0:
            time.sleep(self.transfer_delay_ms / 1000.0)
        result = self.system.restore_once(sandbox.sandbox_id, checkpoint_id)
        if result.status.value == "succeeded" and sandbox.task_run is not None:
            sandbox.task_run.on_restore_complete()
        return result

    def notify_fault(self, sandbox: SandboxHandle, *, reason: str = "fault") -> None:
        assert self.system is not None
        logger.info("Benchmark notifying fault sandbox=%s reason=%s", sandbox.sandbox_id, reason)
        self.system.notify_fault(sandbox.sandbox_id, reason=reason)

    def notify_preemption(self, sandbox: SandboxHandle, *, grace_remaining_seconds: float) -> None:
        assert self.system is not None
        logger.info(
            "Benchmark notifying preemption sandbox=%s grace_remaining_seconds=%.3f",
            sandbox.sandbox_id,
            grace_remaining_seconds,
        )
        self.system.notify_preemption(sandbox.sandbox_id, grace_remaining_seconds=grace_remaining_seconds)

    def wait_for_recovery(
        self,
        sandbox: SandboxHandle,
        *,
        event_type: str,
        observed_after,
        timeout_s: float = 60.0,
    ):
        assert self.system is not None

        def _matching_record():
            record = self.system.get_last_recovery_record(sandbox.sandbox_id)
            if record is None:
                return None
            if record.event_type != event_type:
                return None
            if record.started_at < observed_after:
                return None
            return record

        benchmark_support.wait_for(lambda: _matching_record() is not None, timeout_s=timeout_s)
        record = _matching_record()
        if record is not None and record.status in {"restored", "relaunched"} and sandbox.task_run is not None:
            sandbox.task_run.on_restore_complete()
            wait_for_task_ready = getattr(sandbox.task_run, "wait_for_task_ready", None)
            if callable(wait_for_task_ready):
                wait_for_task_ready()
        logger.info(
            "Observed recovery record sandbox=%s event_type=%s status=%s checkpoint=%s",
            sandbox.sandbox_id,
            event_type,
            record.status if record is not None else "missing",
            "" if record is None or record.checkpoint_id is None else record.checkpoint_id,
        )
        return record

    def list_checkpoint_manifests(self, sandbox_id: SandboxId) -> list[CheckpointManifest]:
        assert self.storage is not None
        manifests: list[CheckpointManifest] = []
        for checkpoint_id in self.storage.list_checkpoints(sandbox_id):
            try:
                manifests.append(self.storage.get_manifest(sandbox_id, checkpoint_id))
            except FileNotFoundError:
                logger.debug(
                    "Skipped checkpoint manifest that disappeared during enumeration sandbox=%s checkpoint=%s",
                    sandbox_id,
                    checkpoint_id,
                )
        return manifests

    def collect_tree_search_checkpoints(
        self,
        sandbox_id: SandboxId,
        *,
        initial_steps: int | None = None,
        require_complete: bool = False,
    ) -> dict[int, benchmark_support.TreeSearchCheckpointRecord]:
        return benchmark_support.build_tree_search_checkpoint_index(
            self.list_checkpoint_manifests(sandbox_id),
            initial_steps=initial_steps,
            require_complete=require_complete,
        )

    def wait_for_tree_search_checkpoints(
        self,
        sandbox_id: SandboxId,
        *,
        initial_steps: int,
        timeout_s: float = 45.0,
    ) -> dict[int, benchmark_support.TreeSearchCheckpointRecord]:
        collected: dict[int, benchmark_support.TreeSearchCheckpointRecord] = {}
        last_error = f"missing tree-search checkpoints for steps {list(range(1, initial_steps + 1))}"

        def _ready() -> bool:
            nonlocal collected
            nonlocal last_error
            try:
                collected = self.collect_tree_search_checkpoints(
                    sandbox_id,
                    initial_steps=initial_steps,
                    require_complete=True,
                )
            except ValueError as exc:
                message = str(exc)
                if message.startswith("missing tree-search checkpoints"):
                    last_error = message
                    return False
                raise
            return True

        if not benchmark_support.wait_for(_ready, timeout_s=timeout_s, interval_s=0.2, raise_on_timeout=False):
            raise RuntimeError(f"timed out waiting for tree-search checkpoints for sandbox {sandbox_id}: {last_error}")
        logger.info(
            "Collected tree-search checkpoints sandbox=%s steps=%s",
            sandbox_id,
            sorted(collected.keys()),
        )
        return collected

    def wait_for_checkpoint_count(
        self,
        sandbox_id: SandboxId,
        *,
        minimum: int,
        timeout_s: float = 45.0,
    ) -> int:
        assert self.storage is not None
        benchmark_support.wait_for(lambda: len(self.storage.list_checkpoints(sandbox_id)) >= minimum, timeout_s=timeout_s)
        return len(self.storage.list_checkpoints(sandbox_id))

    def wait_for_checkpoint_count_stable(
        self,
        sandbox_id: SandboxId,
        *,
        stable_period_s: float = 1.0,
        timeout_s: float = 15.0,
    ) -> int:
        assert self.storage is not None
        logger.debug(
            "Waiting for checkpoint count to stabilize sandbox=%s stable_period_s=%.1f timeout_s=%.1f",
            sandbox_id,
            stable_period_s,
            timeout_s,
        )
        deadline = time.time() + timeout_s
        stable_since = time.time()
        last_count = len(self.storage.list_checkpoints(sandbox_id))
        while time.time() < deadline:
            current = len(self.storage.list_checkpoints(sandbox_id))
            if current != last_count:
                logger.debug(
                    "Checkpoint count changed sandbox=%s previous=%d current=%d",
                    sandbox_id,
                    last_count,
                    current,
                )
                last_count = current
                stable_since = time.time()
            elif time.time() - stable_since >= stable_period_s:
                logger.info("Checkpoint count stabilized sandbox=%s count=%d", sandbox_id, current)
                return current
            time.sleep(0.2)
        raise RuntimeError(f"checkpoint count did not stabilize for sandbox {sandbox_id}")

    def deactivate_sandbox_runtime(self, sandbox: SandboxHandle) -> None:
        logger.info("Deactivating sandbox runtime sandbox=%s", sandbox.sandbox_id)
        self._delete_runtime(sandbox.sandbox_id)
        self._set_sandbox_running_state(sandbox.sandbox_id, is_running=False)

    def inject_fault(self, sandbox: SandboxHandle) -> None:
        logger.info("Injecting fault into sandbox=%s", sandbox.sandbox_id)
        self._delete_runtime(sandbox.sandbox_id)
        self._set_sandbox_running_state(sandbox.sandbox_id, is_running=False)

    def destroy_sandbox_dataset(self, sandbox: SandboxHandle) -> None:
        assert self.runtime is not None
        self._delete_runtime(sandbox.sandbox_id)
        try:
            self.runtime.describe(sandbox.sandbox_id)
        except KeyError:
            if self.llm_server is not None:
                self.llm_server.benchmark_llm_router.unregister_sandbox(str(sandbox.sandbox_id))  # type: ignore[attr-defined]
            self.network_manager.release_lease(sandbox.sandbox_id)
            self._sandbox_by_id.pop(sandbox.sandbox_id, None)
            return
        self._destroy_filesystem_dataset(sandbox.sandbox_id)
        if self.llm_server is not None:
            self.llm_server.benchmark_llm_router.unregister_sandbox(str(sandbox.sandbox_id))  # type: ignore[attr-defined]
        self.network_manager.release_lease(sandbox.sandbox_id)
        self._sandbox_by_id.pop(sandbox.sandbox_id, None)

    def _relaunch_sandbox(self, sandbox_id: SandboxId, event_type: str) -> None:
        _ = event_type
        handle = self._sandbox_by_id[sandbox_id]
        self.relaunch_sandbox(handle)

    def relaunch_sandbox(self, sandbox: SandboxHandle) -> dict[str, object]:
        assert self.base_inspector is not None
        assert self.runtime is not None

        description = self.runtime.describe(sandbox.sandbox_id)
        metadata = dict(description.metadata)
        logger.info("Relaunching sandbox=%s after recovery fallback", sandbox.sandbox_id)
        self._delete_runtime(sandbox.sandbox_id)
        self._destroy_filesystem_dataset(sandbox.sandbox_id)
        self.runtime.launch("runc", metadata)
        self._reset_llm_service_state(sandbox.sandbox_id)
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
                metadata={},
            )
        )
        payload = sandbox.last_status
        preserve_task_run = (
            sandbox.task_run is not None
            and sandbox.task_future is not None
            and not sandbox.task_future.done()
            and sandbox.task_run.survives_fault_relaunch()
            and not (sandbox.launch_source == "compose" and sandbox.llm_service_type == "iflow_trace_replay")
        )
        if sandbox.task_run is not None and not preserve_task_run:
            sandbox.task_run.request_stop()
        if sandbox.task_description is not None and sandbox.task_config is not None:
            if not preserve_task_run:
                self.launch_task(
                    sandbox.agent_type,
                    sandbox.task_description,
                    sandbox.task_config,
                    str(sandbox.sandbox_id),
                )
            assert sandbox.task_run is not None
            sandbox.task_run.wait_for_task_ready()
            if preserve_task_run:
                sandbox.task_run.on_restore_complete()
            try:
                payload = sandbox.task_run.poll_status()
            except RuntimeError:
                payload = sandbox.last_status
        sandbox.last_status = payload
        logger.info("Relaunched sandbox=%s and recovered status endpoint", sandbox.sandbox_id)
        return payload

    def clone_checkpoint_to_fork(
        self,
        source: SandboxHandle,
        checkpoint_id: CheckpointId,
        fork_name: str,
    ) -> SandboxHandle:
        assert self.root is not None
        assert self.storage is not None
        assert self.runtime is not None
        assert self.base_inspector is not None

        network_lease = (
            self.network_manager.allocate_lease(SandboxId(fork_name))
            if self._agent_requires_benchmark_network(source.agent_type)
            else None
        )
        target, work_dir_host_path = self._prepare_sandbox_handle(
            fork_name,
            interceptor_host=self.network_manager.bridge_ip if network_lease is not None else "127.0.0.1",
            network_lease=network_lease,
            agent_type=source.agent_type,
            llm_service_type=source.llm_service_type,
            llm_service_config=self._sandbox_llm_service_config(source),
        )
        if network_lease is not None:
            self.network_manager.register_guest_ip(network_lease.guest_ip, target.sandbox_id)
        self._clone_host_work_dir(source.sandbox_id, target.sandbox_id)

        rootfs_path = target.bundle_dir / "rootfs"
        rootfs_path.mkdir(parents=True, exist_ok=True)
        manifests = {manifest.checkpoint_id: manifest for manifest in self.list_checkpoint_manifests(source.sandbox_id)}
        checkpoint_order = list(manifests.keys())
        copy_plan = benchmark_support.resolve_checkpoint_copy_plan(checkpoint_order, manifests, checkpoint_id)
        filesystem_checkpoint_id = next(copy_id for copy_id, _, copy_filesystem in reversed(copy_plan) if copy_filesystem)
        source_dataset = self._dataset_name_for(source.sandbox_id)
        target_dataset = self._clone_filesystem_snapshot(
            source.sandbox_id,
            filesystem_checkpoint_id,
            target.sandbox_id,
            target_rootfs_path=rootfs_path,
        )
        logger.info(
            "Cloning checkpoint to fork source=%s target=%s selected_checkpoint=%s filesystem_checkpoint=%s copy_plan=%s",
            source.sandbox_id,
            target.sandbox_id,
            checkpoint_id,
            filesystem_checkpoint_id,
            [(str(copy_id), copy_process, copy_filesystem) for copy_id, copy_process, copy_filesystem in copy_plan],
        )

        for copy_id, copy_process, copy_filesystem in copy_plan:
            source_manifest = manifests[copy_id]
            process_refs = []
            filesystem_refs = []
            if copy_process:
                for reference in source_manifest.process_artifacts:
                    payload = self.storage.get_artifact(source.sandbox_id, copy_id, reference)
                    process_refs.append(
                        self.storage.put_artifact(
                            target.sandbox_id,
                            copy_id,
                            ArtifactPayload(
                                kind=reference.kind,
                                name=reference.name,
                                data=self._rewrite_process_artifact(
                                    payload,
                                    source.sandbox_id,
                                    target.sandbox_id,
                                    copy_id,
                                ),
                                metadata=dict(reference.metadata),
                            ),
                        )
                    )
            if copy_filesystem:
                for reference in source_manifest.filesystem_artifacts:
                    payload = self.storage.get_artifact(source.sandbox_id, copy_id, reference)
                    filesystem_refs.append(
                        self.storage.put_artifact(
                            target.sandbox_id,
                            copy_id,
                            ArtifactPayload(
                                kind=reference.kind,
                                name=reference.name,
                                data=self._rewrite_filesystem_artifact(
                                    payload,
                                    source.sandbox_id,
                                    target.sandbox_id,
                                    copy_id,
                                ),
                                metadata=dict(reference.metadata),
                            ),
                        )
                    )
            manifest = CheckpointManifest(
                schema_version=source_manifest.schema_version,
                checkpoint_id=source_manifest.checkpoint_id,
                sandbox_id=target.sandbox_id,
                created_at=source_manifest.created_at,
                runtime_name=source_manifest.runtime_name,
                runtime_version=source_manifest.runtime_version,
                process_artifacts=process_refs,
                filesystem_artifacts=filesystem_refs,
                metadata=dict(source_manifest.metadata),
            ).with_integrity()
            self.storage.put_manifest(manifest)

        description = SandboxDescription(
            sandbox_id=target.sandbox_id,
            runtime_name="runc",
            status="stopped",
            metadata={
                "sandbox_id": str(target.sandbox_id),
                "bundle_path": str(target.bundle_dir),
                "rootfs_path": str(rootfs_path),
                "zfs_dataset": target_dataset,
                "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
            },
        )
        self.runtime._items[target.sandbox_id] = description
        self.runtime._persist(description)
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=target.sandbox_id,
                runtime_name="runc",
                is_running=False,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        target.agent_type = source.agent_type
        target.llm_service_type = source.llm_service_type
        target.llm_base_url = source.llm_base_url
        target.task_description = source.task_description
        target.task_config = source.task_config
        target.launch_source = source.launch_source
        target.launch_metadata = dict(source.launch_metadata)
        target.task_run = None
        target.task_future = None
        if target.task_description is not None and target.task_config is not None:
            target.task_run = self.build_task_run(
                target.agent_type,
                target,
                target.task_description,
                target.task_config,
            )
            target.task_run.prepare_sandbox()
            extra_launch_metadata = dict(target.task_run.extra_launch_metadata())
            if extra_launch_metadata:
                description = replace(
                    description,
                    metadata={**description.metadata, **extra_launch_metadata},
                )
                self.runtime._items[target.sandbox_id] = description
                self.runtime._persist(description)
        return target

    def _clone_host_work_dir(self, source_sandbox_id: SandboxId, target_sandbox_id: SandboxId) -> None:
        source_work_dir = benchmark_support.resolve_work_dir_host_path(self.work_dir_host_root, str(source_sandbox_id))
        target_work_dir = benchmark_support.resolve_work_dir_host_path(self.work_dir_host_root, str(target_sandbox_id))
        if source_work_dir is None or target_work_dir is None:
            return
        if target_work_dir.exists():
            shutil.rmtree(target_work_dir, ignore_errors=True)
        if not source_work_dir.exists():
            target_work_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(
                "Prepared empty fork work dir because source work dir is missing source=%s target=%s",
                source_work_dir,
                target_work_dir,
            )
            return
        shutil.copytree(source_work_dir, target_work_dir)
        logger.debug(
            "Cloned host work dir for fork source_sandbox=%s target_sandbox=%s source=%s target=%s",
            source_sandbox_id,
            target_sandbox_id,
            source_work_dir,
            target_work_dir,
        )

    def clone_tree_search_checkpoint_to_fork(
        self,
        source: SandboxHandle,
        checkpoint_id: CheckpointId,
        fork_name: str,
    ) -> SandboxHandle:
        target = self.clone_checkpoint_to_fork(source, checkpoint_id, fork_name)
        network_lease = self.network_manager.lease_for(target.sandbox_id)
        work_dir_host_path = benchmark_support.resolve_work_dir_host_path(self.work_dir_host_root, str(target.sandbox_id))
        sandbox_image = self.ensure_sandbox_image(target.agent_type) if target.agent_type is not None else None
        sandbox_bundle.write_bundle_config(
            bundle_dir=target.bundle_dir,
            llm_base_url=target.llm_base_url or "",
            provider=self.provider,
            sandbox_name=str(target.sandbox_id),
            status_port=source.status_port,
            cgroup_path=f"agent-cr-bench/{self.pool_name}/{target.sandbox_id}",
            work_dir_host_path=work_dir_host_path,
            network_namespace_path=None if network_lease is None else network_lease.namespace_path,
            image_defaults=None if sandbox_image is None else sandbox_image.image_defaults,
            image_rootfs_dir=None if sandbox_image is None else sandbox_image.exported_rootfs,
        )
        if target.task_run is not None and target.agent_type != "simulated":
            target.task_run.configure_bundle()
        target.status_host = "127.0.0.1" if network_lease is None else network_lease.guest_ip
        target.status_port = source.status_port
        if network_lease is not None:
            self.network_manager.register_guest_ip(network_lease.guest_ip, target.sandbox_id)
        logger.info(
            "Prepared tree-search fork sandbox=%s status_host=%s status_port=%d source=%s checkpoint=%s",
            target.sandbox_id,
            target.status_host,
            target.status_port,
            source.sandbox_id,
            checkpoint_id,
        )
        return target

    def _sandbox_llm_service_config(self, sandbox: SandboxHandle) -> dict[str, object] | None:
        benchmark_metadata = sandbox.launch_metadata.get("benchmark")
        if not isinstance(benchmark_metadata, dict):
            return None
        raw_config = benchmark_metadata.get("llm_service_config")
        if not isinstance(raw_config, dict):
            return None
        return dict(raw_config)

    def sandbox_bundle_config(self, sandbox: SandboxHandle) -> dict[str, object]:
        config_path = sandbox.bundle_dir / "config.json"
        return json.loads(config_path.read_text(encoding="utf-8"))

    def _bundle_spec_writer(self):
        manager = self._active_runtime
        if manager is None or not hasattr(manager, "write_bundle_spec"):
            raise RuntimeError("runtime with bundle spec support is required")
        return manager.write_bundle_spec

    def _delete_runtime(self, sandbox_id: SandboxId) -> None:
        if self.runtime is None or not hasattr(self.runtime, "delete_runtime"):
            return
        self.runtime.delete_runtime(sandbox_id, force=True, ignore_missing=True)

    def _destroy_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        if self.runtime is None or not hasattr(self.runtime, "destroy_filesystem_dataset"):
            return
        self.runtime.destroy_filesystem_dataset(sandbox_id)

    def _dataset_name_for(self, sandbox_id: SandboxId) -> str:
        if self.runtime is not None and hasattr(self.runtime, "dataset_name_for"):
            return self.runtime.dataset_name_for(sandbox_id)
        return f"{self.pool_name}/agent-cr/{sandbox_id}"

    def _clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path: Path,
    ) -> str:
        if self.runtime is None or not hasattr(self.runtime, "clone_filesystem_snapshot"):
            return f"{self.pool_name}/agent-cr/{target_sandbox_id}"
        return self.runtime.clone_filesystem_snapshot(
            source_sandbox_id,
            checkpoint_id,
            target_sandbox_id,
            target_rootfs_path=target_rootfs_path,
        )

    def sandbox_process_cwd(self, sandbox: SandboxHandle) -> str:
        process = self.sandbox_bundle_config(sandbox).get("process", {})
        if not isinstance(process, dict):
            return "/app"
        cwd = process.get("cwd")
        return str(cwd) if isinstance(cwd, str) and cwd else "/app"

    def _sandbox_process_env(self, sandbox: SandboxHandle) -> list[str]:
        process = self.sandbox_bundle_config(sandbox).get("process", {})
        if not isinstance(process, dict):
            return []
        env = process.get("env", [])
        if not isinstance(env, list):
            return []
        return [str(item) for item in env]

    def _sandbox_process_user(self, sandbox: SandboxHandle) -> str | None:
        process = self.sandbox_bundle_config(sandbox).get("process", {})
        if not isinstance(process, dict):
            return None
        user = process.get("user")
        if not isinstance(user, dict):
            return None
        uid = user.get("uid")
        gid = user.get("gid")
        if uid is None:
            return None
        return f"{uid}:{gid}" if gid is not None else str(uid)

    def exec_in_sandbox(
        self,
        sandbox: SandboxHandle,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        assert self.runtime is not None
        merged_env = sandbox_bundle.merge_environment_defaults(
            self._sandbox_process_env(sandbox),
            [] if env is None else [f"{key}={value}" for key, value in env.items()],
        )
        resolved_cwd = cwd or self.sandbox_process_cwd(sandbox)
        user = self._sandbox_process_user(sandbox)
        env_map: dict[str, object] = {}
        for item in merged_env:
            key, _, value = item.partition("=")
            env_map[key] = value
        return self.runtime.exec(
            sandbox.sandbox_id,
            command,
            cwd=resolved_cwd,
            env=env_map,
            user=user,
            timeout_s=timeout_s,
            capture_output=capture_output,
        )

    def wait_for_task_completion(self, sandbox: SandboxHandle, *, timeout_s: float | None = None) -> None:
        if sandbox.task_future is None:
            return
        sandbox.task_future.result(timeout=timeout_s)

    def verify_task_accuracy(
        self,
        sandbox: SandboxHandle,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        command_started = time.perf_counter()
        result = self.exec_in_sandbox(
            sandbox,
            ["/bin/bash", "-lc", "bash /tests/run-tests.sh"],
            cwd=self.sandbox_process_cwd(sandbox),
            env={"TEST_DIR": "/tests"},
            timeout_s=timeout_s,
        )
        verification_ms = (time.perf_counter() - command_started) * 1000.0
        stdout = result.stdout.rstrip()
        stderr = result.stderr.rstrip()
        logger.info(
            "Completed run-tests.sh sandbox=%s exit_code=%s command=%s",
            sandbox.sandbox_id,
            result.returncode,
            " ".join(shlex.quote(part) for part in result.args),
        )
        if stdout:
            logger.info("run-tests stdout sandbox=%s\n%s", sandbox.sandbox_id, stdout)
        if stderr:
            logger.warning("run-tests stderr sandbox=%s\n%s", sandbox.sandbox_id, stderr)
        return {
            "verification_status": "passed" if result.returncode == 0 else "failed",
            "verification_exit_code": result.returncode,
            "verification_ms": verification_ms,
            "verification_stdout": result.stdout,
            "verification_stderr": result.stderr,
            "verification_command": " ".join(shlex.quote(part) for part in result.args),
        }

    def _prepare_sandbox_handle(
        self,
        sandbox_name: str,
        *,
        interceptor_host: str,
        network_lease: sandbox_network.BenchmarkNetworkLease | None = None,
        agent_type: str = "simulated",
        llm_service_type: str = "simulated",
        llm_service_config: dict[str, object] | None = None,
        status_port: int | None = None,
        status_host: str | None = None,
        image_defaults: sandbox_image.ImageRuntimeDefaults | None = None,
        image_rootfs_dir: Path | None = None,
    ) -> tuple[SandboxHandle, Path | None]:
        assert self.root is not None
        assert self.interceptor is not None
        assert self.llm_server is not None
        work_dir_host_path = benchmark_support.resolve_work_dir_host_path(self.work_dir_host_root, sandbox_name)
        prepared = sandbox_launcher.prepare_bundle_launch(
            bundle_root=self.root / "bundles",
            sandbox_name=sandbox_name,
            provider=self.provider,
            pool_name=self.pool_name,
            interceptor_host=interceptor_host,
            interceptor_port=self.interceptor.port,
            status_host=status_host if status_host is not None else ("127.0.0.1" if network_lease is None else network_lease.guest_ip),
            status_port=status_port,
            work_dir_host_path=work_dir_host_path,
            network_namespace_path=None if network_lease is None else network_lease.namespace_path,
            image_defaults=image_defaults,
            image_rootfs_dir=image_rootfs_dir,
            bundle_spec_writer=self._bundle_spec_writer(),
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId(sandbox_name),
            bundle_dir=prepared.bundle_dir,
            status_port=prepared.status_port,
            last_status={},
            status_host=prepared.status_host,
            agent_type=agent_type,
            llm_service_type=llm_service_type,
            llm_base_url=prepared.llm_base_url,
            llm_control_base_url=f"http://127.0.0.1:{self.llm_server.server_address[1]}",
        )
        self.sandboxes.append(handle)
        self._sandbox_by_id[handle.sandbox_id] = handle
        self.llm_server.benchmark_llm_router.register_sandbox(  # type: ignore[attr-defined]
            sandbox_id=str(handle.sandbox_id),
            llm_service_type=llm_service_type,
            llm_service_config=llm_service_config,
        )
        return handle, prepared.work_dir_host_path

    def _llm_service_checkpoint_metadata(self, sandbox_id: SandboxId) -> dict[str, object]:
        if self.llm_server is None:
            return {}
        try:
            return self.llm_server.benchmark_llm_router.checkpoint_metadata(str(sandbox_id))  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to capture llm service checkpoint metadata for sandbox=%s", sandbox_id)
            return {}

    def _restore_llm_service_state(self, sandbox_id: SandboxId, manifest: CheckpointManifest) -> None:
        if self.llm_server is None:
            return
        self.llm_server.benchmark_llm_router.restore_from_checkpoint_metadata(  # type: ignore[attr-defined]
            str(sandbox_id),
            manifest.metadata,
        )

    def _reset_llm_service_state(self, sandbox_id: SandboxId) -> None:
        if self.llm_server is None:
            return
        try:
            self.llm_server.benchmark_llm_router.reset_sandbox(str(sandbox_id))  # type: ignore[attr-defined]
        except Exception:
            logger.exception("Failed to reset llm service state for sandbox=%s", sandbox_id)

    def _set_sandbox_running_state(self, sandbox_id: SandboxId, *, is_running: bool) -> None:
        assert self.base_inspector is not None
        snapshot = self.base_inspector.inspect(sandbox_id)
        self.base_inspector.upsert_snapshot(
            replace(
                snapshot,
                is_running=is_running,
                observed_at=utc_now(),
            )
        )

    def _rewrite_process_artifact(
        self,
        payload: bytes,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> bytes:
        data = json.loads(payload.decode("utf-8"))
        process_root = self.root / "checkpoints" / str(target_sandbox_id) / str(checkpoint_id)
        shutil.copytree(
            self.root / "checkpoints" / str(source_sandbox_id) / str(checkpoint_id),
            process_root,
            dirs_exist_ok=True,
        )
        data["sandbox_id"] = str(target_sandbox_id)
        data["process_checkpoint_location"] = str(process_root / "process")
        status = data.get("status", {})
        if isinstance(status, dict):
            metadata = status.get("metadata", {})
            if isinstance(metadata, dict):
                metadata["sandbox_id"] = str(target_sandbox_id)
                metadata["checkpoint_id"] = str(checkpoint_id)
                metadata["bundle_path"] = str(self.root / "bundles" / str(target_sandbox_id))
                metadata["image_path"] = str(process_root / "process")
                metadata["work_path"] = str(process_root / "work")
        return json.dumps(data, sort_keys=True, indent=2).encode("utf-8")

    def _rewrite_filesystem_artifact(
        self,
        payload: bytes,
        source_sandbox_id: SandboxId,
        target_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> bytes:
        _ = source_sandbox_id
        data = json.loads(payload.decode("utf-8"))
        target_snapshot = f"{self.pool_name}/agent-cr/{target_sandbox_id}@{checkpoint_id}"
        filesystem = data.get("filesystem", {})
        if isinstance(filesystem, dict):
            filesystem["dataset"] = f"{self.pool_name}/agent-cr/{target_sandbox_id}"
            filesystem["snapshot"] = target_snapshot
            filesystem["mountpoint"] = str(self.root / "bundles" / str(target_sandbox_id) / "rootfs")
        status = data.get("status", {})
        if isinstance(status, dict):
            metadata = status.get("metadata", {})
            if isinstance(metadata, dict):
                metadata["sandbox_id"] = str(target_sandbox_id)
                metadata["checkpoint_id"] = str(checkpoint_id)
                metadata["dataset"] = f"{self.pool_name}/agent-cr/{target_sandbox_id}"
                metadata["snapshot"] = target_snapshot
                metadata["mountpoint"] = str(self.root / "bundles" / str(target_sandbox_id) / "rootfs")
        data["sandbox_id"] = str(target_sandbox_id)
        return json.dumps(data, sort_keys=True, indent=2).encode("utf-8")

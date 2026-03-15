#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import csv
import ipaddress
import json
import logging
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable

import yaml

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
    EBPFEvent,
    EBPFEventKind,
    EBPFSandboxInspector,
    HostInspectorServiceClient,
    RemoteSandboxInspector,
    ExecutorConfig,
    InMemoryEBPFEventCollector,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RequestContext,
    RequestInterceptorHook,
    RequestAwareSandboxInspector,
    RuncRuntimeAdapter,
    RuncRuntimePaths,
    RuncSandboxManager,
    RuncSandboxManagerPaths,
    SandboxDescription,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
    TelemetryRequestInterceptorHook,
)
from agent_cr.models import ArtifactPayload, utc_now
from agent_cr.host_inspector.fs_helper import LibbpfFilesystemMonitor
from agent_cr.host_inspector.runtime_resolver import RuntimeResolver
from agent_cr.host_inspector.server import HostInspectorDaemon, HostInspectorServer
from benchmarks.agents import BaseAgent, TaskConfig, TaskDescription, build_agent_registry
# from simulated_agent.image import build_image, export_image_rootfs
from agents.iflow_integration import build_image, export_image_rootfs
from simulated_agent.service import serve

logger = logging.getLogger(__name__)

_HOST_INSPECTOR_HOST = "127.0.0.1"
_HOST_INSPECTOR_PORT = 9782


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


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def bounded_probability(raw: str) -> float:
    value = float(raw)
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(f"expected probability in [0.0, 1.0], got {raw}")
    return value


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--agent-type", choices=["simulated", "iflow"], default="simulated")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--out", default="")
    parser.add_argument("--transfer-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--work-dir-host-root",
        type=Path,
        default=None,
        help="Host directory root for per-sandbox /work bind mounts",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )


def require_binaries() -> None:
    required = ["docker", "runc", "criu", "zfs", "zpool", "ip"]
    missing = [name for name in required if shutil.which(name) is None]
    if missing:
        raise SystemExit(f"missing required binaries: {', '.join(missing)}")


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def wait_for(
    predicate,
    *,
    timeout_s: float = 30.0,
    interval_s: float = 0.2,
    raise_on_timeout: bool = True,
):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    if raise_on_timeout:
        raise RuntimeError("timed out waiting for predicate")
    return False


def parse_ipv4_route_networks(raw_routes: str) -> list[ipaddress.IPv4Network]:
    networks: list[ipaddress.IPv4Network] = []
    for raw_line in raw_routes.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        destination = line.split()[0]
        if destination == "default":
            continue
        try:
            network = ipaddress.ip_network(destination, strict=False)
        except ValueError:
            continue
        if isinstance(network, ipaddress.IPv4Network) and network.prefixlen < 32:
            networks.append(network)
    return networks


def select_benchmark_network(*, existing_routes: str, candidate_pool: str = "10.250.0.0/16") -> tuple[str, str]:
    pool = ipaddress.ip_network(candidate_pool, strict=False)
    if not isinstance(pool, ipaddress.IPv4Network):
        raise ValueError(f"candidate pool must be IPv4, got {candidate_pool}")
    if pool.prefixlen > 24:
        raise ValueError(f"candidate pool must be at most /24, got {candidate_pool}")
    existing_networks = parse_ipv4_route_networks(existing_routes)
    candidates = [pool] if pool.prefixlen == 24 else list(pool.subnets(new_prefix=24))
    for network in candidates:
        if any(network.overlaps(existing) for existing in existing_networks):
            continue
        return str(next(network.hosts())), str(network)
    raise RuntimeError(f"unable to find an available benchmark /24 inside {candidate_pool}")


def enough_progress(payload: dict[str, object], *, total_actions: int = 6) -> bool:
    return (
        int(payload["total_actions"]) >= total_actions
        and int(payload["filesystem_actions"]) >= 1
        and int(payload["process_actions"]) >= 1
        and int(payload["network_actions"]) >= 1
        and int(payload["stateful_actions"]) >= 1
    )


def export_docker_image_rootfs(*, tag: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    container_id = (
        subprocess.run(
            ["docker", "create", tag],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    tar_path = output_dir / "rootfs.tar"
    rootfs_dir = output_dir / "rootfs"
    try:
        with tar_path.open("wb") as fh:
            subprocess.run(["docker", "export", container_id], check=True, stdout=fh)
        if rootfs_dir.exists():
            shutil.rmtree(rootfs_dir)
        rootfs_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(tar_path), "-C", str(rootfs_dir)], check=True)
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rootfs_dir


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"invalid env file line in {path}: {raw_line!r}")
        env[key.strip()] = value.strip()
    return env


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def interpolate_compose_value(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda match: env.get(match.group(1), match.group(2) or ""), value)
    if isinstance(value, list):
        return [interpolate_compose_value(item, env) for item in value]
    if isinstance(value, dict):
        return {str(key): interpolate_compose_value(item, env) for key, item in value.items()}
    return value


def write_bundle_config(
    *,
    bundle_dir: Path,
    interceptor_port: int,
    interceptor_host: str,
    provider: str,
    sandbox_name: str,
    status_port: int,
    cgroup_path: str,
    work_dir_host_path: Path | None = None,
    network_namespace_path: Path | None = None,
) -> None:
    config_path = bundle_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    linux_cfg = cfg.get("linux", {})
    namespaces = []
    network_namespace_found = False
    for namespace in linux_cfg.get("namespaces", []):
        ns_type = namespace.get("type")
        if ns_type == "cgroup":
            continue
        if ns_type == "network":
            network_namespace_found = True
            if network_namespace_path is None:
                continue
            namespace = {**namespace, "path": str(network_namespace_path)}
        namespaces.append(namespace)
    if network_namespace_path is not None and not network_namespace_found:
        namespaces.append({"type": "network", "path": str(network_namespace_path)})
    linux_cfg["namespaces"] = namespaces
    linux_cfg["cgroupsPath"] = cgroup_path
    linux_cfg.pop("seccomp", None)
    cfg["linux"] = linux_cfg
    mounts = [mount for mount in cfg.get("mounts", []) if mount.get("destination") != "/work"]
    if work_dir_host_path is not None:
        work_dir_host_path.mkdir(parents=True, exist_ok=True)
        mounts.append(
            {
                "destination": "/work",
                "source": str(work_dir_host_path),
                "type": "bind",
                "options": ["rbind", "rw"],
            }
        )
    cfg["mounts"] = mounts
    cfg["process"]["terminal"] = False
    cfg["process"]["cwd"] = "/work"
    cfg["process"]["args"] = [
        "/bin/sh",
        "-lc",
        f"exec /usr/local/bin/agent-cli run --provider {provider} >/dev/null 2>/dev/null",
    ]
    cfg["process"]["env"] = [
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "PYTHONUNBUFFERED=1",
        f"INTERCEPTOR_URL=http://{interceptor_host}:{interceptor_port}",
        f"STATUS_PORT={status_port}",
        "POLL_INTERVAL_S=0.2",
        "AGENT_WORK_DIR=/work",
        f"AGENT_SANDBOX_ID={sandbox_name}",
        f"AGENT_PROVIDER={provider}",
    ]
    cfg["root"]["path"] = "rootfs"
    cfg["root"]["readonly"] = False
    config_path.write_text(json.dumps(cfg, indent=2))


def record_activity_events(
    *,
    collector: InMemoryEBPFEventCollector,
    sandbox_id: SandboxId,
    previous: dict[str, object],
    current: dict[str, object],
) -> None:
    mappings = [
        ("filesystem_actions", EBPFEventKind.FILE_WRITE, {"path": "/work/tool_artifact.txt"}),
        ("process_actions", EBPFEventKind.PROCESS_EXEC, {"command": "/bin/sh"}),
        ("network_actions", EBPFEventKind.NETWORK_EGRESS, {"address": "127.0.0.1"}),
    ]
    for field, kind, metadata in mappings:
        if int(current.get(field, 0)) > int(previous.get(field, 0)):
            collector.record(
                EBPFEvent(
                    sandbox_id=sandbox_id,
                    kind=kind,
                    observed_at=utc_now(),
                    metadata=metadata,
                )
            )


def total_actions(payload: dict[str, object]) -> int:
    return int(payload.get("total_actions", 0))


def resolve_work_dir_host_path(work_dir_host_root: Path | None, sandbox_name: str) -> Path | None:
    if work_dir_host_root is None:
        return None
    return work_dir_host_root.expanduser().resolve() / sandbox_name


def compute_summary(rows: list[dict[str, object]], metric_keys: Iterable[str]) -> dict[str, float]:
    summary: dict[str, float] = {}
    if not rows:
        return summary
    for key in metric_keys:
        summary[key] = sum(float(row[key]) for row in rows) / len(rows)
    return summary


def select_injected_indices(
    population_size: int,
    *,
    iteration: int,
    rate: float,
    first_forced_iteration: int,
    rng: random.Random,
) -> list[int]:
    if population_size <= 0:
        return []
    if first_forced_iteration > 0 and iteration < first_forced_iteration:
        return []
    selected = [index for index in range(population_size) if rng.random() < rate]
    if first_forced_iteration > 0 and iteration == first_forced_iteration and 0 not in selected:
        selected.insert(0, 0)
    return sorted(set(selected))


def resolve_checkpoint_copy_plan(
    checkpoint_order: list[CheckpointId],
    manifests: dict[CheckpointId, CheckpointManifest],
    checkpoint_id: CheckpointId,
) -> list[tuple[CheckpointId, bool, bool]]:
    manifest = manifests[checkpoint_id]
    plan: list[tuple[CheckpointId, bool, bool]] = [
        (checkpoint_id, bool(manifest.process_artifacts), bool(manifest.filesystem_artifacts))
    ]
    need_process = not bool(manifest.process_artifacts)
    need_filesystem = not bool(manifest.filesystem_artifacts)
    if not need_process and not need_filesystem:
        return plan

    try:
        current_index = checkpoint_order.index(checkpoint_id)
        candidates = list(reversed(checkpoint_order[:current_index]))
    except ValueError:
        candidates = list(reversed(checkpoint_order))

    for candidate_id in candidates:
        if not need_process and not need_filesystem:
            break
        candidate = manifests[candidate_id]
        copy_process = need_process and bool(candidate.process_artifacts)
        copy_filesystem = need_filesystem and bool(candidate.filesystem_artifacts)
        if not copy_process and not copy_filesystem:
            continue
        plan.insert(0, (candidate_id, copy_process, copy_filesystem))
        if copy_process:
            need_process = False
        if copy_filesystem:
            need_filesystem = False

    if need_process or need_filesystem:
        raise ValueError(f"unable to resolve restore dependencies for checkpoint {checkpoint_id}")
    return plan


@dataclass(frozen=True)
class TreeSearchCheckpointRecord:
    checkpoint_id: CheckpointId
    replay_actions: int
    checkpoint_ms: float = 0.0


def build_tree_search_checkpoint_index(
    manifests: Iterable[CheckpointManifest],
    *,
    initial_steps: int | None = None,
    require_complete: bool = False,
) -> dict[int, TreeSearchCheckpointRecord]:
    indexed: dict[int, TreeSearchCheckpointRecord] = {}
    for manifest in manifests:
        raw_step = manifest.metadata.get("tree_search_step")
        if raw_step is None:
            continue
        try:
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid tree_search_step={raw_step!r} for checkpoint {manifest.checkpoint_id}"
            ) from exc
        if step <= 0:
            continue
        if initial_steps is not None and step > initial_steps:
            continue
        if step in indexed:
            raise ValueError(f"duplicate tree-search checkpoint for step {step}")
        indexed[step] = TreeSearchCheckpointRecord(
            checkpoint_id=manifest.checkpoint_id,
            replay_actions=step,
        )

    if require_complete and initial_steps is not None:
        missing = [step for step in range(1, initial_steps + 1) if step not in indexed]
        if missing:
            raise ValueError(f"missing tree-search checkpoints for steps {missing}")
    return dict(sorted(indexed.items()))


class NoopRequestInterceptorHook(RequestInterceptorHook):
    def on_request_start(self, context: RequestContext) -> None:
        _ = context

    def on_request_end(self, context: RequestContext) -> None:
        _ = context


@dataclass
class SandboxHandle:
    sandbox_id: SandboxId
    bundle_dir: Path
    status_port: int
    last_status: dict[str, object]
    status_host: str = "127.0.0.1"
    agent_type: str = "simulated"
    task_description: TaskDescription | None = None
    task_config: TaskConfig | None = None
    launch_source: str = "runc"
    launch_metadata: dict[str, object] = field(default_factory=dict)
    task_run: BaseAgent | None = None
    task_future: Future[None] | None = None

    @property
    def status_url(self) -> str:
        return f"http://{self.status_host}:{self.status_port}/status"


@dataclass(frozen=True)
class BenchmarkTaskRecord:
    agent_type: str
    task_description: TaskDescription
    task_config: TaskConfig
    docker_compose_file: Path | None = None
    env_file: Path | None = None
    service_name: str | None = None


@dataclass(frozen=True)
class BenchmarkNetworkLease:
    sandbox_id: SandboxId
    namespace_name: str
    namespace_path: Path
    host_veth_name: str
    guest_veth_name: str
    guest_ip: str


class RealHostScenarioHarness:
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
    ) -> None:
        self.provider = provider
        self.transfer_delay_ms = transfer_delay_ms
        self.scheduler_config = scheduler_config
        self.scheduler_policy = scheduler_policy
        self.checkpoint_manager_factory = checkpoint_manager_factory
        self.max_workers = max_workers
        self.auto_cr = auto_cr
        self.work_dir_host_root = work_dir_host_root
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.root: Path | None = None
        self.pool_name = ""
        self.image_tag = ""
        self.runtime_state_root: Path | None = None
        self._host_inspector_server: HostInspectorServer | None = None
        self.telemetry: InMemoryTelemetrySink | None = None
        self.collector: InMemoryEBPFEventCollector | None = None
        self.request_state_store: InMemoryRequestStateStore | None = None
        self.base_inspector: EBPFSandboxInspector | None = None
        self.inspector: RequestAwareSandboxInspector | None = None
        self.runtime: RuncRuntimeAdapter | None = None
        self.storage: CheckpointManager | None = None
        self.executor: CRExecutor | None = None
        self.sandbox_manager: RuncSandboxManager | None = None
        self.system: AgentCRSystem | None = None
        self.interceptor: AgentCRRequestInterceptorServer | None = None
        self.interceptor_hook = CompositeRequestInterceptorHook()
        self.llm_server = None
        self.llm_thread: threading.Thread | None = None
        self.sandboxes: list[SandboxHandle] = []
        self._sandbox_by_id: dict[SandboxId, SandboxHandle] = {}
        self._benchmark_bridge_name: str | None = None
        self._benchmark_bridge_ip = "10.250.0.1"
        self._benchmark_network_cidr = "10.250.0.0/24"
        self._benchmark_ip_cursor = 2
        self._benchmark_network_leases: dict[SandboxId, BenchmarkNetworkLease] = {}
        self._benchmark_ip_to_sandbox: dict[str, SandboxId] = {}
        self._compose_image_tags: set[str] = set()
        self._agent_registry = build_agent_registry()
        self._task_executor = ThreadPoolExecutor(max_workers=max(1, self.max_workers))

    def _start_host_inspector_server(self) -> str:
        assert self.runtime_state_root is not None
        if self._host_inspector_server is not None:
            return f"http://{_HOST_INSPECTOR_HOST}:{_HOST_INSPECTOR_PORT}"

        self.runtime_state_root.mkdir(parents=True, exist_ok=True)
        self._host_inspector_server = HostInspectorServer(
            host=_HOST_INSPECTOR_HOST,
            port=_HOST_INSPECTOR_PORT,
            daemon=HostInspectorDaemon(
                resolver=RuntimeResolver(runc_state_root=self.runtime_state_root),
                fs_monitor=LibbpfFilesystemMonitor(),
            ),
        )
        logger.info(
            "Starting host inspector server in-process host=%s port=%d runc_state_root=%s",
            _HOST_INSPECTOR_HOST,
            _HOST_INSPECTOR_PORT,
            self.runtime_state_root,
        )
        self._host_inspector_server.start()
        url = f"http://{_HOST_INSPECTOR_HOST}:{_HOST_INSPECTOR_PORT}"
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

    def _configure_benchmark_network(self) -> None:
        configured_cidr = os.environ.get("AGENT_CR_BENCHMARK_NETWORK_CIDR", "").strip()
        if configured_cidr:
            network = ipaddress.ip_network(configured_cidr, strict=False)
            if not isinstance(network, ipaddress.IPv4Network):
                raise ValueError(f"benchmark network must be IPv4, got {configured_cidr}")
            if network.prefixlen != 24:
                raise ValueError(f"benchmark network must be a /24, got {configured_cidr}")
            self._benchmark_network_cidr = str(network)
            self._benchmark_bridge_ip = str(next(network.hosts()))
            return
        route_result = subprocess.run(
            ["ip", "-4", "route", "show", "table", "main"],
            check=True,
            capture_output=True,
            text=True,
        )
        self._benchmark_bridge_ip, self._benchmark_network_cidr = select_benchmark_network(
            existing_routes=route_result.stdout,
        )

    def __enter__(self) -> "RealHostScenarioHarness":
        require_binaries()
        self._configure_benchmark_network()
        logger.info(
            "Selected benchmark network cidr=%s bridge_ip=%s",
            self._benchmark_network_cidr,
            self._benchmark_bridge_ip,
        )
        self._tmpdir = tempfile.TemporaryDirectory(prefix="agent_cr_scenario_bench_")
        self.root = Path(self._tmpdir.name)
        unique_suffix = uuid.uuid4().hex[:10]
        self.pool_name = f"agentcrbench{unique_suffix}"
        self.image_tag = f"agent-cr-scenario-bench:{unique_suffix}"
        self.runtime_state_root = self.root / "runtime-state"
        host_inspector_url = self._start_host_inspector_server()
        self.llm_server = serve(host="127.0.0.1", port=0, response_delay_ms=250)
        self.llm_thread = threading.Thread(target=self.llm_server.serve_forever, daemon=True)
        self.llm_thread.start()

        build_image(tag=self.image_tag)
        exported_rootfs = export_image_rootfs(tag=self.image_tag, output_dir=self.root / "image")
        subprocess.run(["truncate", "-s", "10G", str(self.root / "zpool.img")], check=True)
        subprocess.run(["zpool", "create", "-f", self.pool_name, str(self.root / "zpool.img")], check=True)
        subprocess.run(["zfs", "create", f"{self.pool_name}/agent-cr"], check=True)

        self.telemetry = InMemoryTelemetrySink()
        self.collector = InMemoryEBPFEventCollector()
        self.request_state_store = InMemoryRequestStateStore()

        host_inspector_client = HostInspectorServiceClient(host_inspector_url)
        self.base_inspector = RemoteSandboxInspector(host_inspector_client)
        self.inspector = RequestAwareSandboxInspector(self.base_inspector, self.request_state_store)
        self.runtime = RuncRuntimeAdapter(
            paths=RuncRuntimePaths(
                state_root=self.runtime_state_root,
                bundle_root=self.root / "bundles",
                checkpoint_root=self.root / "checkpoints",
                zfs_dataset_prefix=f"{self.pool_name}/agent-cr",
            )
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
            ),
            DefaultRWorker(
                AdapterProcessRWorker(self.runtime),
                AdapterFileSystemRWorker(self.runtime),
                self.storage,
            ),
            self.telemetry,
        )
        self.sandbox_manager = RuncSandboxManager(
            paths=RuncSandboxManagerPaths(
                state_root=self.runtime_state_root,
                bundle_root=self.root / "bundles",
                metadata_root=self.root / "sandbox-meta",
                zfs_dataset_prefix=f"{self.pool_name}/agent-cr",
            ),
            host_inspector_client=host_inspector_client
        )
        self.system = AgentCRSystem(
            scheduler=CRScheduler(
                self.scheduler_config,
                self.inspector,
                self.sandbox_manager,
                InMemorySchedulerStateStore(),
                self.telemetry,
                self.scheduler_policy,
            ),
            executor=self.executor,
            storage=self.storage,
            inspector=self.inspector,
            sandbox_manager=self.sandbox_manager,
            telemetry=self.telemetry,
            request_state_store=self.request_state_store,
            relaunch_handler=self._relaunch_sandbox if self.auto_cr else None,
            recovery_delay_seconds=self.transfer_delay_ms / 1000.0 if self.auto_cr else 0.0,
        )
        self.interceptor_hook.add_hook(TelemetryRequestInterceptorHook(self.telemetry))
        self.interceptor = AgentCRRequestInterceptorServer(
            upstream_url=f"http://127.0.0.1:{self.llm_server.server_address[1]}",
            request_state_store=self.request_state_store,
            hook=self.interceptor_hook,
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
        self.exported_rootfs = exported_rootfs
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.system is not None and self.auto_cr:
            self.system.stop()
        if self.interceptor is not None:
            self.interceptor.stop()
        self._task_executor.shutdown(wait=False, cancel_futures=True)
        if self.executor is not None:
            self.executor.shutdown()
        if self.runtime_state_root is not None:
            for sandbox in self.sandboxes:
                subprocess.run(
                    ["runc", "--root", str(self.runtime_state_root), "delete", "-f", str(sandbox.sandbox_id)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        for sandbox_id in list(self._benchmark_network_leases):
            self._release_benchmark_network_lease(sandbox_id)
        if self._benchmark_bridge_name is not None:
            subprocess.run(
                ["ip", "link", "delete", self._benchmark_bridge_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._benchmark_bridge_name = None
        if self.image_tag:
            subprocess.run(
                ["docker", "rmi", "-f", self.image_tag],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        for image_tag in sorted(self._compose_image_tags):
            subprocess.run(
                ["docker", "rmi", "-f", image_tag],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        if self.pool_name:
            subprocess.run(
                ["zfs", "destroy", "-r", f"{self.pool_name}/agent-cr"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
        return self.get_agent_class(agent_type)(self, sandbox, task_description, task_config)

    def _agent_requires_benchmark_network(self, agent_type: str) -> bool:
        return bool(self.get_agent_class(agent_type).requires_network_namespace)

    def get_sandbox_handle(self, sandbox_id: str | SandboxId) -> SandboxHandle:
        target = SandboxId(str(sandbox_id))
        return self._sandbox_by_id[target]

    def load_dataset(self, path: Path) -> list[BenchmarkTaskRecord]:
        dataset_root = path.expanduser().resolve().parent
        records: list[BenchmarkTaskRecord] = []
        for line_number, raw_line in enumerate(path.expanduser().resolve().read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"dataset row {line_number} in {path} must be an object")
            compose_file = payload.get("docker_compose_file")
            env_file = payload.get("env_file")
            records.append(
                BenchmarkTaskRecord(
                    agent_type=str(payload.get("agent_type", "simulated")),
                    task_description=TaskDescription.from_json_value(payload.get("task_description", "")),
                    task_config=TaskConfig.from_json_value(payload.get("task_config")),
                    docker_compose_file=None if compose_file is None else (dataset_root / str(compose_file)).resolve(),
                    env_file=None if env_file is None else (dataset_root / str(env_file)).resolve(),
                    service_name=None if payload.get("service_name") is None else str(payload["service_name"]),
                )
            )
        if not records:
            raise ValueError(f"dataset {path} did not contain any task rows")
        return records

    def select_task_record(
        self,
        dataset: list[BenchmarkTaskRecord] | None,
        *,
        sandbox_index: int,
        default_agent_type: str,
        default_task_description: TaskDescription,
        default_task_config: TaskConfig,
    ) -> BenchmarkTaskRecord:
        if dataset:
            return dataset[sandbox_index % len(dataset)]
        return BenchmarkTaskRecord(
            agent_type=default_agent_type,
            task_description=default_task_description,
            task_config=default_task_config,
        )

    def launch_sandbox(self, sandbox_name: str, *, agent_type: str = "simulated") -> SandboxHandle:
        assert self.root is not None
        assert self.base_inspector is not None
        assert self.system is not None
        assert self.interceptor is not None

        default_task_description = TaskDescription("")
        default_task_config = TaskConfig()
        network_lease = (
            self._allocate_benchmark_network_lease(SandboxId(sandbox_name))
            if self._agent_requires_benchmark_network(agent_type)
            else None
        )
        handle, work_dir_host_path = self._prepare_sandbox_handle(
            sandbox_name,
            interceptor_host=self._benchmark_bridge_ip if network_lease is not None else "127.0.0.1",
            network_lease=network_lease,
            agent_type=agent_type,
        )
        task_run = self.build_task_run(agent_type, handle, default_task_description, default_task_config)
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
            self._benchmark_ip_to_sandbox[network_lease.guest_ip] = sandbox_id
        launch_metadata = {
            "sandbox_id": sandbox_name,
            "bundle_path": str(handle.bundle_dir),
            "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
            "rootfs_init_dirs": task_run.rootfs_init_dirs(),
            "rootfs_copy_paths": [{"source": str(self.exported_rootfs), "destination": "/"}],
            **task_run.extra_launch_metadata(),
            **handle.launch_metadata.get("runtime", {}),
        }
        self.system.sandbox_manager.launch("runc", launch_metadata)
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
        task_description: TaskDescription,
        task_config: TaskConfig,
    ) -> SandboxHandle:
        handle = self.launch_sandbox(sandbox_name, agent_type=agent_type)
        self.launch_task(agent_type, task_description, task_config, str(handle.sandbox_id))
        return handle

    def launch_task_record(
        self,
        sandbox_name: str,
        task_record: BenchmarkTaskRecord,
    ) -> SandboxHandle:
        if task_record.docker_compose_file is not None:
            if task_record.env_file is None:
                raise ValueError("compose-backed dataset rows must include env_file")
            return self.launch_sandbox_from_docker_compose_file(
                task_record.docker_compose_file,
                task_record.env_file,
                sandbox_name=sandbox_name,
                service_name=task_record.service_name,
                agent_type=task_record.agent_type,
                task_description=task_record.task_description,
                task_config=task_record.task_config,
            )
        return self.launch_sandbox_and_task(
            sandbox_name,
            agent_type=task_record.agent_type,
            task_description=task_record.task_description,
            task_config=task_record.task_config,
        )

    def launch_sandbox_from_docker_compose_file(
        self,
        compose_file: Path,
        env_file: Path,
        *,
        sandbox_name: str,
        service_name: str | None = None,
        status_host: str | None = None,
        status_port: int | None = None,
        agent_type: str | None = None,
        task_description: TaskDescription | None = None,
        task_config: TaskConfig | None = None,
    ) -> SandboxHandle:
        compose_env = {**os.environ, **parse_env_file(env_file)}
        payload = yaml.safe_load(compose_file.read_text(encoding="utf-8")) or {}
        payload = interpolate_compose_value(payload, compose_env)
        services = payload.get("services")
        if not isinstance(services, dict) or not services:
            raise ValueError(f"compose file {compose_file} does not define any services")
        if service_name is None:
            if len(services) != 1:
                raise ValueError(f"compose file {compose_file} contains multiple services; specify service_name")
            service_name = next(iter(services))
        if service_name not in services:
            raise ValueError(f"compose service {service_name!r} not found in {compose_file}")
        service = services[service_name]
        unsupported = {"depends_on", "profiles", "networks", "configs", "secrets", "healthcheck"}
        found_unsupported = sorted(key for key in unsupported if key in service)
        if found_unsupported:
            raise ValueError(f"unsupported compose features for benchmark translation: {found_unsupported}")
        network_lease = self._allocate_benchmark_network_lease(SandboxId(sandbox_name))
        handle, work_dir_host_path = self._prepare_sandbox_handle(
            sandbox_name,
            interceptor_host=self._benchmark_bridge_ip,
            network_lease=network_lease,
            agent_type=agent_type or "simulated",
            status_port=status_port,
            status_host=status_host if status_host is not None else network_lease.guest_ip,
        )
        translated = self._translate_compose_service(
            compose_file=compose_file,
            service_name=service_name,
            service=service,
            handle=handle,
            work_dir_host_path=work_dir_host_path,
        )
        assert self.base_inspector is not None
        assert self.system is not None
        self._benchmark_ip_to_sandbox[network_lease.guest_ip] = handle.sandbox_id
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
        self.system.sandbox_manager.launch("runc", translated)
        handle.launch_source = "compose"
        handle.launch_metadata["runtime"] = dict(translated)
        handle.last_status = {}
        if agent_type is not None and task_description is not None and task_config is not None:
            self.launch_task(agent_type, task_description, task_config, str(handle.sandbox_id))
            if handle.task_run is not None:
                try:
                    handle.task_run.wait_for_task_ready()
                except RuntimeError:
                    pass
        return handle

    def add_interceptor_hook(self, hook: RequestInterceptorHook) -> None:
        self.interceptor_hook.add_hook(hook)
        logger.debug("Registered interceptor hook %s", type(hook).__name__)

    def resolve_interceptor_sandbox_id(
        self,
        client_host: str | None,
        headers: dict[str, str],
        body: bytes,
    ) -> str | None:
        _ = (headers, body)
        if client_host is None:
            return None
        sandbox_id = self._benchmark_ip_to_sandbox.get(client_host)
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

    def record_activity(self, sandbox: SandboxHandle, current: dict[str, object]) -> None:
        assert self.collector is not None
        record_activity_events(
            collector=self.collector,
            sandbox_id=sandbox.sandbox_id,
            previous=sandbox.last_status,
            current=current,
        )
        sandbox.last_status = current

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
        if sandbox.launch_source != "runc":
            raise RuntimeError(f"checkpoint_manual unsupported for launch_source={sandbox.launch_source}")
        logger.debug("Benchmark requesting checkpoint_manual for sandbox=%s", sandbox.sandbox_id)
        return self.system.checkpoint_once(sandbox.sandbox_id, leave_running=leave_running)

    def checkpoint_if_due(self, sandbox: SandboxHandle):
        assert self.system is not None
        if sandbox.launch_source != "runc":
            raise RuntimeError(f"checkpoint_if_due unsupported for launch_source={sandbox.launch_source}")
        logger.debug("Benchmark requesting checkpoint_if_due for sandbox=%s", sandbox.sandbox_id)
        return self.system.checkpoint_if_due(sandbox.sandbox_id)

    def restore_once(self, sandbox: SandboxHandle, checkpoint_id: CheckpointId):
        assert self.system is not None
        if sandbox.launch_source != "runc":
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

        wait_for(lambda: _matching_record() is not None, timeout_s=timeout_s)
        record = _matching_record()
        if record is not None and record.status == "restored" and sandbox.task_run is not None:
            sandbox.task_run.on_restore_complete()
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
        return [self.storage.get_manifest(sandbox_id, checkpoint_id) for checkpoint_id in self.storage.list_checkpoints(sandbox_id)]

    def collect_tree_search_checkpoints(
        self,
        sandbox_id: SandboxId,
        *,
        initial_steps: int | None = None,
        require_complete: bool = False,
    ) -> dict[int, TreeSearchCheckpointRecord]:
        return build_tree_search_checkpoint_index(
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
    ) -> dict[int, TreeSearchCheckpointRecord]:
        collected: dict[int, TreeSearchCheckpointRecord] = {}
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

        if not wait_for(_ready, timeout_s=timeout_s, interval_s=0.2, raise_on_timeout=False):
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
        wait_for(lambda: len(self.storage.list_checkpoints(sandbox_id)) >= minimum, timeout_s=timeout_s)
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
        assert self.runtime_state_root is not None
        logger.info("Deactivating sandbox runtime sandbox=%s", sandbox.sandbox_id)
        subprocess.run(
            ["runc", "--root", str(self.runtime_state_root), "delete", "-f", str(sandbox.sandbox_id)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._set_sandbox_running_state(sandbox.sandbox_id, is_running=False)

    def inject_fault(self, sandbox: SandboxHandle) -> None:
        assert self.runtime_state_root is not None
        logger.info("Injecting fault into sandbox=%s", sandbox.sandbox_id)
        subprocess.run(
            ["runc", "--root", str(self.runtime_state_root), "delete", "-f", str(sandbox.sandbox_id)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._set_sandbox_running_state(sandbox.sandbox_id, is_running=False)

    def destroy_sandbox_dataset(self, sandbox: SandboxHandle) -> None:
        assert self.sandbox_manager is not None
        assert self.runtime_state_root is not None
        subprocess.run(
            ["runc", "--root", str(self.runtime_state_root), "delete", "-f", str(sandbox.sandbox_id)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            description = self.sandbox_manager.describe(sandbox.sandbox_id)
        except KeyError:
            self._release_benchmark_network_lease(sandbox.sandbox_id)
            self._sandbox_by_id.pop(sandbox.sandbox_id, None)
            return
        dataset = str(description.metadata.get("zfs_dataset", ""))
        if dataset:
            subprocess.run(
                ["zfs", "destroy", "-r", dataset],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self._release_benchmark_network_lease(sandbox.sandbox_id)
        self._sandbox_by_id.pop(sandbox.sandbox_id, None)

    def _relaunch_sandbox(self, sandbox_id: SandboxId, event_type: str) -> None:
        _ = event_type
        handle = self._sandbox_by_id[sandbox_id]
        self.relaunch_sandbox(handle)

    def relaunch_sandbox(self, sandbox: SandboxHandle) -> dict[str, object]:
        assert self.base_inspector is not None
        assert self.sandbox_manager is not None
        assert self.runtime_state_root is not None

        description = self.sandbox_manager.describe(sandbox.sandbox_id)
        metadata = dict(description.metadata)
        logger.info("Relaunching sandbox=%s after recovery fallback", sandbox.sandbox_id)
        subprocess.run(
            ["runc", "--root", str(self.runtime_state_root), "delete", "-f", str(sandbox.sandbox_id)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        dataset = str(metadata.get("zfs_dataset", ""))
        if dataset:
            subprocess.run(
                ["zfs", "destroy", "-r", dataset],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        self.sandbox_manager.launch("runc", metadata)
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
        if sandbox.task_description is not None and sandbox.task_config is not None:
            self.launch_task(
                sandbox.agent_type,
                sandbox.task_description,
                sandbox.task_config,
                str(sandbox.sandbox_id),
            )
            assert sandbox.task_run is not None
            sandbox.task_run.wait_for_task_ready()
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
        assert self.sandbox_manager is not None
        assert self.base_inspector is not None

        network_lease = (
            self._allocate_benchmark_network_lease(SandboxId(fork_name))
            if self._agent_requires_benchmark_network(source.agent_type)
            else None
        )
        target, work_dir_host_path = self._prepare_sandbox_handle(
            fork_name,
            interceptor_host=self._benchmark_bridge_ip if network_lease is not None else "127.0.0.1",
            network_lease=network_lease,
            agent_type=source.agent_type,
        )
        if network_lease is not None:
            self._benchmark_ip_to_sandbox[network_lease.guest_ip] = target.sandbox_id

        source_dataset = f"{self.pool_name}/agent-cr/{source.sandbox_id}"
        target_dataset = f"{self.pool_name}/agent-cr/{target.sandbox_id}"
        rootfs_path = target.bundle_dir / "rootfs"
        rootfs_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(["zfs", "destroy", "-r", target_dataset], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        manifests = {manifest.checkpoint_id: manifest for manifest in self.list_checkpoint_manifests(source.sandbox_id)}
        checkpoint_order = list(manifests.keys())
        copy_plan = resolve_checkpoint_copy_plan(checkpoint_order, manifests, checkpoint_id)
        filesystem_checkpoint_id = next(copy_id for copy_id, _, copy_filesystem in reversed(copy_plan) if copy_filesystem)
        logger.info(
            "Cloning checkpoint to fork source=%s target=%s selected_checkpoint=%s filesystem_checkpoint=%s copy_plan=%s",
            source.sandbox_id,
            target.sandbox_id,
            checkpoint_id,
            filesystem_checkpoint_id,
            [(str(copy_id), copy_process, copy_filesystem) for copy_id, copy_process, copy_filesystem in copy_plan],
        )
        subprocess.run(
            [
                "zfs",
                "clone",
                "-o",
                f"mountpoint={rootfs_path}",
                f"{source_dataset}@{filesystem_checkpoint_id}",
                target_dataset,
            ],
            check=True,
        )
        subprocess.run(["zfs", "snapshot", f"{target_dataset}@{filesystem_checkpoint_id}"], check=True)

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
        self.sandbox_manager._items[target.sandbox_id] = description
        self.sandbox_manager._persist(description)
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
        return target

    def clone_tree_search_checkpoint_to_fork(
        self,
        source: SandboxHandle,
        checkpoint_id: CheckpointId,
        fork_name: str,
    ) -> SandboxHandle:
        target = self.clone_checkpoint_to_fork(source, checkpoint_id, fork_name)
        network_lease = self._benchmark_network_leases.get(target.sandbox_id)
        work_dir_host_path = resolve_work_dir_host_path(self.work_dir_host_root, str(target.sandbox_id))
        write_bundle_config(
            bundle_dir=target.bundle_dir,
            interceptor_port=self.interceptor.port if self.interceptor is not None else 0,
            interceptor_host=self._benchmark_bridge_ip if network_lease is not None else "127.0.0.1",
            provider=self.provider,
            sandbox_name=str(target.sandbox_id),
            status_port=source.status_port,
            cgroup_path=f"agent-cr-bench/{self.pool_name}/{target.sandbox_id}",
            work_dir_host_path=work_dir_host_path,
            network_namespace_path=None if network_lease is None else network_lease.namespace_path,
        )
        if target.task_run is not None and target.agent_type != "simulated":
            target.task_run.configure_bundle()
        target.status_host = "127.0.0.1" if network_lease is None else network_lease.guest_ip
        target.status_port = source.status_port
        if network_lease is not None:
            self._benchmark_ip_to_sandbox[network_lease.guest_ip] = target.sandbox_id
        logger.info(
            "Prepared tree-search fork sandbox=%s status_host=%s status_port=%d source=%s checkpoint=%s",
            target.sandbox_id,
            target.status_host,
            target.status_port,
            source.sandbox_id,
            checkpoint_id,
        )
        return target

    def _translate_compose_service(
        self,
        *,
        compose_file: Path,
        service_name: str,
        service: dict[str, object],
        handle: SandboxHandle,
        work_dir_host_path: Path | None,
    ) -> dict[str, object]:
        bundle_config = handle.bundle_dir / "config.json"
        config = json.loads(bundle_config.read_text(encoding="utf-8"))
        image_ref = self._resolve_compose_image_ref(
            compose_file=compose_file,
            service_name=service_name,
            service=service,
        )
        rootfs_dir = export_docker_image_rootfs(
            tag=image_ref,
            output_dir=self.root / "compose-images" / str(handle.sandbox_id),
        )
        config["process"]["cwd"] = str(service.get("working_dir", "/work"))
        config["process"]["terminal"] = bool(service.get("tty", False))
        config["process"]["env"] = self._compose_environment(service)
        config["process"]["args"] = self._compose_process_args(service)
        mounts = [mount for mount in config.get("mounts", []) if mount.get("destination") != "/work"]
        if work_dir_host_path is not None:
            work_dir_host_path.mkdir(parents=True, exist_ok=True)
            mounts.append(
                {
                    "destination": "/work",
                    "source": str(work_dir_host_path),
                    "type": "bind",
                    "options": ["rbind", "rw"],
                }
            )
        mounts.extend(self._compose_mounts(service))
        config["mounts"] = mounts
        bundle_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
        handle.launch_metadata["compose"] = {
            "service_name": service_name,
            "image_ref": image_ref,
            "ports": list(service.get("ports", [])),
        }
        return {
            "sandbox_id": str(handle.sandbox_id),
            "bundle_path": str(handle.bundle_dir),
            "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
            "rootfs_init_dirs": [
                "work",
                "tmp",
                "proc",
                "dev",
                "dev/pts",
                "dev/shm",
                "dev/mqueue",
                "sys",
                "run",
                "var",
            ],
            "rootfs_copy_paths": [{"source": str(rootfs_dir), "destination": "/"}],
            "compose_service_name": service_name,
            "compose_ports": list(service.get("ports", [])),
        }

    def _resolve_compose_image_ref(
        self,
        *,
        compose_file: Path,
        service_name: str,
        service: dict[str, object],
    ) -> str:
        image_ref = service.get("image")
        build_spec = service.get("build")
        if image_ref and build_spec:
            raise ValueError(f"compose service {service_name} cannot specify both image and build")
        if isinstance(image_ref, str) and image_ref:
            return image_ref
        if build_spec is None:
            raise ValueError(f"compose service {service_name} requires image or build")
        tag = f"agent-cr-compose-{service_name}-{uuid.uuid4().hex[:10]}"
        self._compose_image_tags.add(tag)
        build_context = compose_file.parent
        dockerfile = None
        build_args: list[str] = []
        if isinstance(build_spec, str):
            build_context = (compose_file.parent / build_spec).resolve()
        elif isinstance(build_spec, dict):
            context_value = build_spec.get("context", ".")
            build_context = (compose_file.parent / str(context_value)).resolve()
            dockerfile_value = build_spec.get("dockerfile")
            if dockerfile_value is not None:
                dockerfile = (build_context / str(dockerfile_value)).resolve()
            args_value = build_spec.get("args", {})
            if isinstance(args_value, dict):
                for key, value in sorted(args_value.items()):
                    build_args.extend(["--build-arg", f"{key}={value}"])
        else:
            raise ValueError(f"unsupported compose build definition for service {service_name}: {build_spec!r}")
        command = ["docker", "build", "-t", tag]
        if dockerfile is not None:
            command.extend(["-f", str(dockerfile)])
        command.extend(build_args)
        command.append(str(build_context))
        subprocess.run(command, check=True)
        return tag

    def _compose_environment(self, service: dict[str, object]) -> list[str]:
        environment = service.get("environment", {})
        if isinstance(environment, dict):
            items = environment.items()
        elif isinstance(environment, list):
            items = []
            for item in environment:
                key, sep, value = str(item).partition("=")
                items.append((key, value if sep else os.environ.get(key, "")))
        else:
            raise ValueError(f"unsupported compose environment: {environment!r}")
        return [f"{key}={value}" for key, value in items]

    def _compose_process_args(self, service: dict[str, object]) -> list[str]:
        entrypoint = service.get("entrypoint")
        command = service.get("command")
        segments: list[str] = []
        for value in (entrypoint, command):
            if value is None:
                continue
            if isinstance(value, list):
                segments.extend(str(item) for item in value)
            elif isinstance(value, str):
                segments.extend(["/bin/sh", "-lc", value])
            else:
                raise ValueError(f"unsupported compose command/entrypoint value: {value!r}")
                # pragma: no cover
        if segments:
            return segments
        return ["/bin/sh"]

    def _compose_mounts(self, service: dict[str, object]) -> list[dict[str, object]]:
        mounts: list[dict[str, object]] = []
        for item in service.get("volumes", []):
            if isinstance(item, str):
                parts = item.split(":")
                if len(parts) < 2:
                    raise ValueError(f"unsupported compose volume syntax: {item!r}")
                source = Path(parts[0]).expanduser().resolve()
                destination = parts[1]
                options = ["rbind", "rw"]
                if len(parts) > 2 and parts[2] == "ro":
                    options = ["rbind", "ro"]
                mounts.append(
                    {
                        "destination": destination,
                        "source": str(source),
                        "type": "bind",
                        "options": options,
                    }
                )
                continue
            if isinstance(item, dict) and item.get("type", "bind") == "bind":
                source = Path(str(item["source"])).expanduser().resolve()
                read_only = bool(item.get("read_only", False))
                mounts.append(
                    {
                        "destination": str(item["target"]),
                        "source": str(source),
                        "type": "bind",
                        "options": ["rbind", "ro" if read_only else "rw"],
                    }
                )
                continue
            raise ValueError(f"unsupported compose volume definition: {item!r}")
        return mounts

    def _prepare_sandbox_handle(
        self,
        sandbox_name: str,
        *,
        interceptor_host: str,
        network_lease: BenchmarkNetworkLease | None = None,
        agent_type: str = "simulated",
        status_port: int | None = None,
        status_host: str | None = None,
    ) -> tuple[SandboxHandle, Path | None]:
        assert self.root is not None
        assert self.interceptor is not None
        resolved_status_port = find_free_port() if status_port is None else status_port
        bundle_dir = self.root / "bundles" / sandbox_name
        work_dir_host_path = resolve_work_dir_host_path(self.work_dir_host_root, sandbox_name)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True)
        write_bundle_config(
            bundle_dir=bundle_dir,
            interceptor_port=self.interceptor.port,
            interceptor_host=interceptor_host,
            provider=self.provider,
            sandbox_name=sandbox_name,
            status_port=resolved_status_port,
            cgroup_path=f"agent-cr-bench/{self.pool_name}/{sandbox_name}",
            work_dir_host_path=work_dir_host_path,
            network_namespace_path=None if network_lease is None else network_lease.namespace_path,
        )
        handle = SandboxHandle(
            sandbox_id=SandboxId(sandbox_name),
            bundle_dir=bundle_dir,
            status_port=resolved_status_port,
            last_status={},
            status_host=(
                status_host
                if status_host is not None
                else ("127.0.0.1" if network_lease is None else network_lease.guest_ip)
            ),
            agent_type=agent_type,
        )
        self.sandboxes.append(handle)
        self._sandbox_by_id[handle.sandbox_id] = handle
        return handle, work_dir_host_path

    def _ensure_benchmark_bridge(self) -> None:
        if self._benchmark_bridge_name is not None:
            return
        assert self.root is not None
        bridge_name = f"acb{uuid.uuid4().hex[:8]}"
        subprocess.run(["ip", "link", "add", bridge_name, "type", "bridge"], check=True)
        subprocess.run(
            ["ip", "addr", "add", f"{self._benchmark_bridge_ip}/24", "dev", bridge_name],
            check=True,
        )
        subprocess.run(["ip", "link", "set", bridge_name, "up"], check=True)
        self._benchmark_bridge_name = bridge_name
        logger.info(
            "Created benchmark bridge name=%s bridge_ip=%s",
            bridge_name,
            self._benchmark_bridge_ip,
        )

    def _allocate_benchmark_network_lease(self, sandbox_id: SandboxId) -> BenchmarkNetworkLease:
        self._ensure_benchmark_bridge()
        assert self._benchmark_bridge_name is not None
        network = ipaddress.ip_network(self._benchmark_network_cidr)
        if self._benchmark_ip_cursor >= network.num_addresses - 1:
            raise RuntimeError("benchmark network exhausted guest IP capacity")
        guest_ip = str(network[self._benchmark_ip_cursor])
        self._benchmark_ip_cursor += 1
        suffix = uuid.uuid4().hex[:8]
        namespace_name = f"ts-{suffix}"
        host_veth_name = f"vh{suffix[:6]}"
        guest_veth_name = f"vg{suffix[:6]}"
        subprocess.run(["ip", "netns", "add", namespace_name], check=True)
        subprocess.run(
            ["ip", "link", "add", host_veth_name, "type", "veth", "peer", "name", guest_veth_name],
            check=True,
        )
        subprocess.run(["ip", "link", "set", host_veth_name, "master", self._benchmark_bridge_name], check=True)
        subprocess.run(["ip", "link", "set", host_veth_name, "up"], check=True)
        subprocess.run(["ip", "link", "set", guest_veth_name, "netns", namespace_name], check=True)
        subprocess.run(["ip", "netns", "exec", namespace_name, "ip", "link", "set", "lo", "up"], check=True)
        subprocess.run(
            ["ip", "netns", "exec", namespace_name, "ip", "link", "set", guest_veth_name, "name", "eth0"],
            check=True,
        )
        subprocess.run(
            ["ip", "netns", "exec", namespace_name, "ip", "addr", "add", f"{guest_ip}/24", "dev", "eth0"],
            check=True,
        )
        subprocess.run(["ip", "netns", "exec", namespace_name, "ip", "link", "set", "eth0", "up"], check=True)
        subprocess.run(
            [
                "ip",
                "netns",
                "exec",
                namespace_name,
                "ip",
                "route",
                "replace",
                "default",
                "via",
                self._benchmark_bridge_ip,
            ],
            check=True,
        )
        lease = BenchmarkNetworkLease(
            sandbox_id=sandbox_id,
            namespace_name=namespace_name,
            namespace_path=Path("/var/run/netns") / namespace_name,
            host_veth_name=host_veth_name,
            guest_veth_name=guest_veth_name,
            guest_ip=guest_ip,
        )
        self._benchmark_network_leases[sandbox_id] = lease
        logger.info(
            "Allocated benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
            sandbox_id,
            guest_ip,
            namespace_name,
        )
        return lease

    def _release_benchmark_network_lease(self, sandbox_id: SandboxId) -> None:
        lease = self._benchmark_network_leases.pop(sandbox_id, None)
        if lease is None:
            return
        self._benchmark_ip_to_sandbox.pop(lease.guest_ip, None)
        subprocess.run(
            ["ip", "netns", "del", lease.namespace_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["ip", "link", "delete", lease.host_veth_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info(
            "Released benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
            sandbox_id,
            lease.guest_ip,
            lease.namespace_name,
        )

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


def write_rows(path: str, rows: list[dict[str, object]]) -> None:
    if not path or not rows:
        return
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

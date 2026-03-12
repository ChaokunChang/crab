#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

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
from simulated_agent.image import build_image, export_image_rootfs
from simulated_agent.service import serve

logger = logging.getLogger(__name__)


def configure_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--out", default="")
    parser.add_argument("--transfer-delay-ms", type=float, default=0.0)
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
    )


def require_binaries() -> None:
    required = ["docker", "runc", "criu", "zfs", "zpool"]
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


def enough_progress(payload: dict[str, object], *, total_actions: int = 6) -> bool:
    return (
        int(payload["total_actions"]) >= total_actions
        and int(payload["filesystem_actions"]) >= 1
        and int(payload["process_actions"]) >= 1
        and int(payload["network_actions"]) >= 1
        and int(payload["stateful_actions"]) >= 1
    )


def write_bundle_config(
    *,
    bundle_dir: Path,
    interceptor_port: int,
    provider: str,
    sandbox_name: str,
    status_port: int,
    cgroup_path: str,
) -> None:
    config_path = bundle_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    linux_cfg = cfg.get("linux", {})
    linux_cfg["namespaces"] = [
        ns
        for ns in linux_cfg.get("namespaces", [])
        if ns.get("type") not in {"network", "cgroup"}
    ]
    linux_cfg["cgroupsPath"] = cgroup_path
    linux_cfg.pop("seccomp", None)
    cfg["linux"] = linux_cfg
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
        f"INTERCEPTOR_URL=http://127.0.0.1:{interceptor_port}",
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

    @property
    def status_url(self) -> str:
        return f"http://127.0.0.1:{self.status_port}/status"


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
    ) -> None:
        self.provider = provider
        self.transfer_delay_ms = transfer_delay_ms
        self.scheduler_config = scheduler_config
        self.scheduler_policy = scheduler_policy
        self.checkpoint_manager_factory = checkpoint_manager_factory
        self.max_workers = max_workers
        self.auto_cr = auto_cr
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None
        self.root: Path | None = None
        self.pool_name = ""
        self.image_tag = ""
        self.runtime_state_root: Path | None = None
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

    def __enter__(self) -> "RealHostScenarioHarness":
        require_binaries()
        self._tmpdir = tempfile.TemporaryDirectory(prefix="agent_cr_scenario_bench_")
        self.root = Path(self._tmpdir.name)
        unique_suffix = uuid.uuid4().hex[:10]
        self.pool_name = f"agentcrbench{unique_suffix}"
        self.image_tag = f"agent-cr-scenario-bench:{unique_suffix}"
        self.runtime_state_root = self.root / "runtime-state"
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
        self.base_inspector = EBPFSandboxInspector(self.collector)
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
            )
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
            host="127.0.0.1",
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
        if self.image_tag:
            subprocess.run(
                ["docker", "rmi", "-f", self.image_tag],
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
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def launch_sandbox(self, sandbox_name: str) -> SandboxHandle:
        assert self.root is not None
        assert self.inspector is not None
        assert self.base_inspector is not None
        assert self.system is not None
        assert self.interceptor is not None

        status_port = find_free_port()
        bundle_dir = self.root / "bundles" / sandbox_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True)
        write_bundle_config(
            bundle_dir=bundle_dir,
            interceptor_port=self.interceptor.port,
            provider=self.provider,
            sandbox_name=sandbox_name,
            status_port=status_port,
            cgroup_path=f"agent-cr-bench/{self.pool_name}/{sandbox_name}",
        )
        sandbox_id = SandboxId(sandbox_name)
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
        self.system.sandbox_manager.launch(
            "runc",
            {
                "sandbox_id": sandbox_name,
                "bundle_path": str(bundle_dir),
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
                "rootfs_copy_paths": [{"source": str(self.exported_rootfs), "destination": "/"}],
            },
        )
        payload = wait_for_http_json(f"http://127.0.0.1:{status_port}/status")
        handle = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=bundle_dir,
            status_port=status_port,
            last_status=payload,
        )
        self.sandboxes.append(handle)
        self._sandbox_by_id[sandbox_id] = handle
        logger.info(
            "Launched benchmark sandbox name=%s sandbox_id=%s status_port=%d auto_cr=%s",
            sandbox_name,
            sandbox_id,
            status_port,
            self.auto_cr,
        )
        return handle

    def add_interceptor_hook(self, hook: RequestInterceptorHook) -> None:
        self.interceptor_hook.add_hook(hook)
        logger.debug("Registered interceptor hook %s", type(hook).__name__)

    def poll_status(self, sandbox: SandboxHandle) -> dict[str, object]:
        return wait_for_http_json(sandbox.status_url)

    def wait_for_progress(self, sandbox: SandboxHandle, *, minimum_actions: int) -> dict[str, object]:
        wait_for(lambda: enough_progress(self.poll_status(sandbox), total_actions=minimum_actions), timeout_s=45.0)
        payload = self.poll_status(sandbox)
        self.record_activity(sandbox, payload)
        return payload

    def wait_for_action_delta(self, sandbox: SandboxHandle, *, delta: int) -> dict[str, object]:
        baseline = total_actions(sandbox.last_status)
        wait_for(lambda: total_actions(self.poll_status(sandbox)) >= baseline + delta, timeout_s=45.0)
        payload = self.poll_status(sandbox)
        self.record_activity(sandbox, payload)
        return payload

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

    def checkpoint_if_due(self, sandbox: SandboxHandle):
        assert self.system is not None
        logger.debug("Benchmark requesting checkpoint_if_due for sandbox=%s", sandbox.sandbox_id)
        return self.system.checkpoint_if_due(sandbox.sandbox_id)

    def restore_once(self, sandbox: SandboxHandle, checkpoint_id: CheckpointId):
        assert self.system is not None
        logger.info(
            "Benchmark requesting restore sandbox=%s checkpoint=%s transfer_delay_ms=%.1f",
            sandbox.sandbox_id,
            checkpoint_id,
            self.transfer_delay_ms,
        )
        if self.transfer_delay_ms > 0:
            time.sleep(self.transfer_delay_ms / 1000.0)
        return self.system.restore_once(sandbox.sandbox_id, checkpoint_id)

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

    def inject_fault(self, sandbox: SandboxHandle) -> None:
        assert self.runtime_state_root is not None
        logger.info("Injecting fault into sandbox=%s", sandbox.sandbox_id)
        subprocess.run(
            ["runc", "--root", str(self.runtime_state_root), "delete", "-f", str(sandbox.sandbox_id)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert self.base_inspector is not None
        snapshot = self.base_inspector.inspect(sandbox.sandbox_id)
        self.base_inspector.upsert_snapshot(
            replace(
                snapshot,
                is_running=False,
                observed_at=utc_now(),
            )
        )

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
            return
        dataset = str(description.metadata.get("zfs_dataset", ""))
        if dataset:
            subprocess.run(
                ["zfs", "destroy", "-r", dataset],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
        payload = wait_for_http_json(sandbox.status_url)
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

        target = self.launch_sandbox(fork_name)
        subprocess.run(
            ["runc", "--root", str(self.runtime_state_root), "delete", "-f", fork_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        source_dataset = f"{self.pool_name}/agent-cr/{source.sandbox_id}"
        target_dataset = f"{self.pool_name}/agent-cr/{target.sandbox_id}"
        rootfs_path = target.bundle_dir / "rootfs"
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
            },
        )
        self.sandbox_manager._items[target.sandbox_id] = description
        self.sandbox_manager._persist(description)
        self.base_inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=target.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        return target

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

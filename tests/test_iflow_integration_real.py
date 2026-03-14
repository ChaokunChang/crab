from __future__ import annotations

from dataclasses import dataclass
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path
from typing import Any

from agent_cr import (
    AgentCRRequestInterceptorServer,
    AgentCRSystem,
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CheckpointId,
    CRExecutor,
    CRScheduler,
    CompositeRequestInterceptorHook,
    DefaultCWorker,
    DefaultRWorker,
    ExecutorConfig,
    HostInspectorServiceClient,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RemoteSandboxInspector,
    RequestAwareSandboxInspector,
    RuncRuntimeAdapter,
    RuncRuntimePaths,
    RuncSandboxManager,
    RuncSandboxManagerPaths,
    SandboxId,
    SchedulerConfig,
    StorageConfig,
    TelemetryRequestInterceptorHook,
)
from agent_cr.host_inspector.fs_helper import LibbpfFilesystemMonitor
from agent_cr.host_inspector.runtime_resolver import RuntimeResolver
from agent_cr.host_inspector.server import HostInspectorDaemon, HostInspectorServer
from agent_cr.models import JobStatus, utc_now
from agents.iflow_integration import (
    BridgeNetworkNamespace,
    build_image,
    export_image_rootfs,
    ensure_cache_files,
    launch_manual_iflow,
    prepare_iflow_runtime,
    prepare_iflow_state,
    stop_manual_iflow,
)
from agents.iflow_integration.harness import rootfs_copy_paths, write_bundle_config
from agents.iflow_integration.service import default_script_steps, serve, serve_manual


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_http_json(url: str, *, timeout_s: float = 20.0) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - real host only
            last_exc = exc
            time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {url}: {last_exc}")


def _wait_for(predicate, *, timeout_s: float = 20.0, interval_s: float = 0.2) -> Any:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval_s)
    raise RuntimeError("timed out waiting for predicate")


def _helper_binary() -> Path:
    return Path(__file__).resolve().parents[1] / "agent_cr" / "host_inspector" / "bpf" / "fs_monitor"


def _ensure_helper_built() -> Path:
    helper = _helper_binary()
    subprocess.run(["make"], cwd=helper.parent, check=True)
    return helper


def _status_for(client: HostInspectorServiceClient, sandbox_id: SandboxId) -> dict[str, object]:
    status = dict(client.get_proc_and_fs_status(sandbox_id)["status"])
    metadata = dict(status.get("metadata", {}))
    metadata["current_process_identities"] = [
        _proc_identity(int(pid))
        for pid in metadata.get("current_pids", [])
    ]
    status["metadata"] = metadata
    return status


def _wait_for_status(
    client: HostInspectorServiceClient,
    sandbox_id: SandboxId,
    *,
    predicate,
    timeout_s: float = 20.0,
    interval_s: float = 0.2,
) -> dict[str, object]:
    deadline = time.time() + timeout_s
    last_status: dict[str, object] | None = None
    while time.time() < deadline:
        last_status = _status_for(client, sandbox_id)
        if predicate(last_status):
            return last_status
        time.sleep(interval_s)
    raise RuntimeError(f"timed out waiting for sandbox status; last_status={last_status}")


def _report_tree(root: Path) -> list[str]:
    if not root.exists():
        return []
    entries: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.is_dir():
            entries.append(f"{rel}/")
        else:
            entries.append(str(rel))
    return entries


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _proc_identity(pid: int) -> dict[str, object]:
    exe_path = Path(f"/proc/{pid}/exe")
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    status_path = Path(f"/proc/{pid}/status")
    try:
        exe = str(exe_path.resolve(strict=True))
    except OSError as exc:
        exe = f"<unavailable:{exc}>"
    try:
        cmdline = [
            part.decode("utf-8", errors="replace")
            for part in cmdline_path.read_bytes().split(b"\0")
            if part
        ]
    except OSError as exc:
        cmdline = [f"<unavailable:{exc}>"]
    status_name = None
    try:
        for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("Name:"):
                status_name = line.split(":", 1)[1].strip()
                break
    except OSError:
        status_name = None
    return {"pid": pid, "exe": exe, "cmdline": cmdline, "name": status_name}


def _checkpoint_log_excerpt(checkpoint_root: Path, sandbox_id: SandboxId, checkpoint_id: str) -> str | None:
    sandbox_root = checkpoint_root / str(sandbox_id)
    root = sandbox_root / checkpoint_id
    if not root.exists():
        candidates = [path for path in sandbox_root.iterdir() if path.is_dir()] if sandbox_root.exists() else []
        if not candidates:
            return None
        root = max(candidates, key=lambda path: path.stat().st_mtime)
    parts: list[str] = []
    for path in sorted(root.rglob("*.log")):
        content = _read_text_if_exists(path)
        if content:
            parts.append(f"{path.relative_to(root)}:\n{content}")
    if not parts:
        return None
    return "\n\n".join(parts)


def _wait_for_phase_request(scripted_state, phase: str, *, timeout_s: float = 120.0) -> dict[str, Any]:
    def _match() -> dict[str, Any] | None:
        snapshot = scripted_state.snapshot()
        for event in snapshot["events"]:
            if event["event"] == "request" and event["phase"] == phase:
                return event
        return None

    return _wait_for(_match, timeout_s=timeout_s, interval_s=0.2)


def _runc_status(runtime_state_root: Path, sandbox_id: SandboxId) -> str:
    result = subprocess.run(
        ["runc", "--root", str(runtime_state_root), "state", str(sandbox_id)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "missing"
    return str(json.loads(result.stdout).get("status", "unknown"))


def _runc_checkpoint_leaves_running() -> bool:
    adapter = RuncRuntimeAdapter()
    return "--leave-running=true" in adapter._checkpoint_cmd(SandboxId("sbx-check"), CheckpointId("ckpt-check"))


def _unique_test_suffix() -> str:
    return str(time.time_ns())


def _host_pids_with_cmdline(*needles: str) -> list[int]:
    matches: list[int] = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit():
            continue
        try:
            cmdline = proc_dir.joinpath("cmdline").read_text(encoding="utf-8", errors="replace").replace("\0", " ")
        except OSError:
            continue
        if all(needle in cmdline for needle in needles):
            matches.append(int(proc_dir.name))
    return matches


def _wait_for_phase_request_or_fail(
    scripted_state,
    phase: str,
    *,
    runtime_state_root: Path,
    sandbox_id: SandboxId,
    work_dir: Path,
    logs_dir: Path | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        snapshot = scripted_state.snapshot()
        for event in snapshot["events"]:
            if event["event"] == "request" and event["phase"] == phase:
                return event
        status = _runc_status(runtime_state_root, sandbox_id)
        if status == "stopped":
            stderr = None if logs_dir is None else _read_text_if_exists(logs_dir / "iflow.stderr")
            stdout = None if logs_dir is None else _read_text_if_exists(logs_dir / "iflow.stdout")
            raise AssertionError(
                "sandbox stopped before phase "
                f"{phase}\nstdout={stdout!r}\nstderr={stderr!r}\n"
            )
        time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for phase request: {phase}")


def _wait_for_manual_turn(manual_state, sandbox_id: str, *, minimum_turns: int, timeout_s: float = 120.0) -> dict[str, Any]:
    def _match() -> dict[str, Any] | None:
        snapshot = manual_state.snapshot()
        if int(snapshot["turns"].get(sandbox_id, 0)) >= minimum_turns:
            return snapshot
        return None

    return _wait_for(_match, timeout_s=timeout_s, interval_s=0.2)


def _wait_for_manual_turn_or_fail(
    manual_state,
    sandbox_id: str,
    *,
    minimum_turns: int,
    runtime_state_root: Path,
    runtime_sandbox_id: SandboxId,
    host_client: HostInspectorServiceClient | None = None,
    logs_dir: Path | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        snapshot = manual_state.snapshot()
        if int(snapshot["turns"].get(sandbox_id, 0)) >= minimum_turns:
            return snapshot
        status = _runc_status(runtime_state_root, runtime_sandbox_id)
        if status in {"stopped", "missing"}:
            if host_client is not None:
                host_status = _status_for(host_client, runtime_sandbox_id)
                metadata = dict(host_status.get("metadata", {}))
                if bool(host_status.get("is_running")) or bool(metadata.get("ignored_pids")):
                    time.sleep(0.2)
                    continue
            stderr = None if logs_dir is None else _read_text_if_exists(logs_dir / "iflow.stderr")
            stdout = None if logs_dir is None else _read_text_if_exists(logs_dir / "iflow.stdout")
            raise AssertionError(
                "sandbox stopped before reaching manual turn "
                f"{minimum_turns}\nstdout={stdout!r}\nstderr={stderr!r}\n"
            )
        time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for manual turn >= {minimum_turns}")


def _wait_for_file_text(path: Path, *, predicate, timeout_s: float = 60.0) -> str:
    def _match() -> str | None:
        content = _read_text_if_exists(path)
        if content is None:
            return None
        return content if predicate(content) else None

    return _wait_for(_match, timeout_s=timeout_s, interval_s=0.2)


def _wait_for_http_text(url: str, *, predicate, timeout_s: float = 30.0) -> str:
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            if predicate(body):
                return body
        except Exception as exc:  # pragma: no cover - real host only
            last_exc = exc
        time.sleep(0.2)
    raise RuntimeError(f"timed out waiting for {url}: {last_exc}")


@dataclass
class _RealIFlowManualFixture:
    root: Path
    keep_root: bool
    report_path: Path
    sandbox_id: SandboxId
    sandbox_ip: str
    image_tag: str
    pool_name: str
    bundle_dir: Path
    runtime_state_root: Path
    checkpoint_root: Path
    work_dir: Path
    root_dir: Path
    logs_dir: Path
    host_client: HostInspectorServiceClient
    system: AgentCRSystem
    executor: CRExecutor
    request_state_store: InMemoryRequestStateStore
    telemetry: InMemoryTelemetrySink
    manual_server: Any
    manual_thread: threading.Thread
    inspector_server: HostInspectorServer
    interceptor: AgentCRRequestInterceptorServer
    network: BridgeNetworkNamespace
    observation: dict[str, Any]


class IFlowRealIntegrationTests(unittest.TestCase):
    def _require_real_iflow_prereqs(self) -> tuple[dict[str, Path], Path]:
        if (
            os.geteuid() != 0
            or shutil.which("docker") is None
            or shutil.which("runc") is None
            or shutil.which("criu") is None
            or shutil.which("zfs") is None
            or shutil.which("ip") is None
        ):
            self.skipTest("root plus docker/runc/criu/zfs/ip required")
        return ensure_cache_files(), _ensure_helper_built()

    def _create_real_iflow_manual_fixture(
        self,
        *,
        name: str,
        task_description: str,
    ) -> _RealIFlowManualFixture:
        cache_files, helper_path = self._require_real_iflow_prereqs()
        unique_suffix = _unique_test_suffix()
        keep_root = os.environ.get("AGENT_CR_KEEP_IFLOW_TMP", "0") == "1"
        root = Path(tempfile.mkdtemp(prefix=f"agent_cr_iflow_{name}_"))
        sandbox_id = SandboxId(f"sbx-iflow-{name}")
        report_path = root / f"{name}_observation.json"
        image_tag = f"agent-cr-iflow-agent:{name}-{unique_suffix}"
        pool_name = f"agentcriflow{name}{unique_suffix}"
        bundle_dir = root / "bundles" / str(sandbox_id)
        runtime_state_root = root / "runtime-state"
        checkpoint_root = root / "checkpoints"
        storage_root = root / "storage"
        sandbox_metadata_root = root / "sandbox-meta"
        image_root = root / "image"
        pool_file = root / "zpool.img"
        sandbox_ip = os.environ.get("AGENT_CR_IFLOW_SANDBOX_IP", "172.17.0.240")
        manual_port = _find_free_port()
        observation: dict[str, Any] = {
            "task_description": task_description,
            "cache_files": {item_name: str(path) for item_name, path in cache_files.items()},
            "phases": {},
        }

        manual_server = serve_manual(host="127.0.0.1", port=manual_port, default_sandbox_id=str(sandbox_id))
        manual_thread = threading.Thread(target=manual_server.serve_forever, daemon=True)
        manual_thread.start()
        _wait_for_http_json(f"http://127.0.0.1:{manual_port}/healthz")

        inspector_server = HostInspectorServer(
            host="127.0.0.1",
            port=_find_free_port(),
            daemon=HostInspectorDaemon(
                resolver=RuntimeResolver(runc_state_root=runtime_state_root),
                process_poll_interval_s=0.2,
                fs_monitor=LibbpfFilesystemMonitor(helper_path=str(helper_path)),
            ),
        )
        inspector_server.start()
        host_client = HostInspectorServiceClient(f"http://127.0.0.1:{inspector_server.port}")

        network = BridgeNetworkNamespace(
            name=f"agentcriflow-{name}-{unique_suffix}",
            ip_address=sandbox_ip,
        )

        executor: CRExecutor | None = None
        interceptor: AgentCRRequestInterceptorServer | None = None
        fixture: _RealIFlowManualFixture | None = None
        try:
            build_image(tag=image_tag)
            exported_rootfs = export_image_rootfs(tag=image_tag, output_dir=image_root)
            alternate_node_runtime_dir_raw = os.environ.get("AGENT_CR_IFLOW_NODE_RUNTIME_DIR")
            prepared_runtime = prepare_iflow_runtime(
                work_root=root,
                cache_dir=cache_files["node-v22.18.0-linux-x64.tar.xz"].parent,
                alternate_node_runtime_dir=None
                if alternate_node_runtime_dir_raw is None
                else Path(alternate_node_runtime_dir_raw),
            )

            subprocess.run(["truncate", "-s", "1024M", str(pool_file)], check=True)
            subprocess.run(["zpool", "create", "-f", pool_name, str(pool_file)], check=True)
            subprocess.run(["zfs", "create", f"{pool_name}/agent-cr"], check=True)

            bundle_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True)
            network.create()

            telemetry = InMemoryTelemetrySink()
            request_state_store = InMemoryRequestStateStore()
            base_inspector = RemoteSandboxInspector(host_client)
            inspector = RequestAwareSandboxInspector(base_inspector, request_state_store)
            runtime = RuncRuntimeAdapter(
                paths=RuncRuntimePaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    checkpoint_root=checkpoint_root,
                    zfs_dataset_prefix=f"{pool_name}/agent-cr",
                )
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=storage_root))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                host_inspector_client=host_client,
                paths=RuncSandboxManagerPaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    metadata_root=sandbox_metadata_root,
                    zfs_dataset_prefix=f"{pool_name}/agent-cr",
                ),
            )
            scheduler = CRScheduler(
                SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                    prefer_checkpoint_during_llm_request=True,
                    require_llm_request_for_checkpoint=True,
                ),
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                sandbox_manager=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_state_store,
            )
            interceptor = AgentCRRequestInterceptorServer(
                upstream_url=f"http://127.0.0.1:{manual_port}",
                request_state_store=request_state_store,
                hook=CompositeRequestInterceptorHook([TelemetryRequestInterceptorHook(telemetry)]),
                on_state_change=system.notify_interceptor_state_change,
                default_sandbox_id=sandbox_id,
                host="0.0.0.0",
                port=0,
            )
            interceptor.start()
            _wait_for_http_json(f"http://127.0.0.1:{interceptor.port}/healthz")

            prepared_state = prepare_iflow_state(
                work_root=root,
                base_url=os.environ.get("AGENT_CR_IFLOW_BASE_URL", f"http://172.17.0.1:{interceptor.port}/v1"),
                model_name=os.environ.get("AGENT_CR_IFLOW_MODEL_NAME", "agent-cr-iflow-manual"),
            )
            observation["runtime_strategy"] = prepared_runtime.runtime_strategy
            observation["runtime_root"] = str(prepared_runtime.root)
            observation["node_source"] = str(prepared_runtime.node_source)
            observation["mounted_state_paths"] = {
                "iflow_home": str(prepared_state.iflow_home),
                "npm_home": str(prepared_state.npm_home),
                "logs_dir": str(prepared_state.logs_dir),
            }
            observation["ignore_process_rules"] = prepared_runtime.ignore_process_rules

            write_bundle_config(
                bundle_dir=bundle_dir,
                interceptor_port=interceptor.port,
                cgroup_path=f"agent-cr-tests/{pool_name}/{sandbox_id}",
                sandbox_id=sandbox_id,
                task_description=task_description,
                prepared_runtime=prepared_runtime,
                prepared_state=prepared_state,
                network_namespace_path=network.namespace_path,
            )

            launched_id = sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": str(sandbox_id),
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
                        "root",
                        "root/.iflow",
                        "root/.npm",
                        "opt/iflow-runtime",
                        "opt/iflow-logs",
                    ],
                    "rootfs_copy_paths": rootfs_copy_paths(exported_rootfs=exported_rootfs),
                    "host_inspector_ignore_process_rules": prepared_runtime.ignore_process_rules,
                },
            )
            self.assertEqual(launched_id, sandbox_id)

            fixture = _RealIFlowManualFixture(
                root=root,
                keep_root=keep_root,
                report_path=report_path,
                sandbox_id=sandbox_id,
                sandbox_ip=sandbox_ip,
                image_tag=image_tag,
                pool_name=pool_name,
                bundle_dir=bundle_dir,
                runtime_state_root=runtime_state_root,
                checkpoint_root=checkpoint_root,
                work_dir=bundle_dir / "rootfs" / "work",
                root_dir=bundle_dir / "rootfs" / "root",
                logs_dir=prepared_state.logs_dir,
                host_client=host_client,
                system=system,
                executor=executor,
                request_state_store=request_state_store,
                telemetry=telemetry,
                manual_server=manual_server,
                manual_thread=manual_thread,
                inspector_server=inspector_server,
                interceptor=interceptor,
                network=network,
                observation=observation,
            )
            return fixture
        except Exception:
            if interceptor is not None:
                interceptor.stop()
            if executor is not None:
                executor.shutdown()
            inspector_server.stop()
            manual_server.shutdown()
            manual_server.server_close()
            manual_thread.join(5.0)
            subprocess.run(
                ["runc", "--root", str(runtime_state_root), "delete", "-f", str(sandbox_id)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zfs", "destroy", "-r", f"{pool_name}/agent-cr/{sandbox_id}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zfs", "destroy", "-r", f"{pool_name}/agent-cr"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zpool", "destroy", "-f", pool_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            network.cleanup()
            subprocess.run(
                ["docker", "rmi", "-f", image_tag],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if not keep_root:
                shutil.rmtree(root, ignore_errors=True)
            raise

    def _cleanup_real_iflow_manual_fixture(self, fixture: _RealIFlowManualFixture) -> None:
        fixture.observation["manual_llm"] = fixture.manual_server.manual_state.snapshot()  # type: ignore[attr-defined]
        fixture.observation["work_tree"] = _report_tree(fixture.work_dir)
        fixture.observation["root_tree"] = _report_tree(fixture.root_dir)
        fixture.observation["iflow_stdout"] = _read_text_if_exists(fixture.logs_dir / "iflow.stdout")
        fixture.observation["iflow_stderr"] = _read_text_if_exists(fixture.logs_dir / "iflow.stderr")
        fixture.report_path.write_text(json.dumps(fixture.observation, indent=2, sort_keys=True), encoding="utf-8")

        try:
            fixture.system.sandbox_manager.delete(fixture.sandbox_id)
        except Exception:
            pass
        fixture.interceptor.stop()
        fixture.executor.shutdown()
        fixture.inspector_server.stop()
        fixture.manual_server.shutdown()
        fixture.manual_server.server_close()
        fixture.manual_thread.join(5.0)
        subprocess.run(
            ["runc", "--root", str(fixture.runtime_state_root), "delete", "-f", str(fixture.sandbox_id)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["zfs", "destroy", "-r", f"{fixture.pool_name}/agent-cr/{fixture.sandbox_id}"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["zfs", "destroy", "-r", f"{fixture.pool_name}/agent-cr"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["zpool", "destroy", "-f", fixture.pool_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        fixture.network.cleanup()
        subprocess.run(
            ["docker", "rmi", "-f", fixture.image_tag],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not fixture.keep_root:
            shutil.rmtree(fixture.root, ignore_errors=True)

    def _wait_for_request_in_flight(self, fixture: _RealIFlowManualFixture, *, timeout_s: float = 60.0) -> None:
        _wait_for(
            lambda: fixture.request_state_store.get(fixture.sandbox_id).llm_request_in_flight,
            timeout_s=timeout_s,
            interval_s=0.1,
        )

    def _delete_runtime_state(self, fixture: _RealIFlowManualFixture) -> str:
        subprocess.run(
            ["runc", "--root", str(fixture.runtime_state_root), "delete", "-f", str(fixture.sandbox_id)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _wait_for(
            lambda: (
                status
                if (status := _runc_status(fixture.runtime_state_root, fixture.sandbox_id)) == "missing"
                else None
            ),
            timeout_s=30.0,
            interval_s=0.2,
        )

    def _checkpoint_and_restore_fixture(
        self,
        fixture: _RealIFlowManualFixture,
        *,
        use_scheduler: bool,
        observation_key: str,
    ) -> tuple[Any, Any]:
        checkpoint_result = (
            fixture.system.checkpoint_if_due(fixture.sandbox_id)
            if use_scheduler
            else fixture.system.checkpoint_once(fixture.sandbox_id)
        )
        self.assertIsNotNone(checkpoint_result)
        assert checkpoint_result is not None
        fixture.observation[observation_key] = {
            "checkpoint": {
                "checkpoint_id": str(checkpoint_result.checkpoint_id),
                "status": checkpoint_result.status.value,
                "message": checkpoint_result.message,
                "operation_statuses": [
                    {
                        "phase": status.metadata.get("phase"),
                        "executed": bool(status.executed),
                        "reason": str(status.reason),
                    }
                    for status in checkpoint_result.operation_statuses
                ],
            }
        }
        if checkpoint_result.status != JobStatus.SUCCEEDED:
            fixture.observation[observation_key]["checkpoint"]["log_excerpt"] = _checkpoint_log_excerpt(
                checkpoint_root=fixture.checkpoint_root,
                sandbox_id=fixture.sandbox_id,
                checkpoint_id=str(checkpoint_result.checkpoint_id),
            )
            self.fail(
                f"{observation_key} checkpoint failed; report={fixture.report_path}"
            )

        fixture.observation[observation_key]["post_checkpoint_runtime_state"] = _runc_status(
            fixture.runtime_state_root,
            fixture.sandbox_id,
        )
        fixture.observation[observation_key]["runtime_state_after_delete"] = self._delete_runtime_state(fixture)

        restore_result = fixture.system.restore_once(fixture.sandbox_id, checkpoint_result.checkpoint_id)
        fixture.observation[observation_key]["restore"] = {
            "status": restore_result.status.value,
            "checkpoint_id": str(restore_result.checkpoint_id),
            "message": restore_result.message,
        }
        self.assertEqual(
            restore_result.status,
            JobStatus.SUCCEEDED,
            f"{observation_key} restore failed; report={fixture.report_path}",
        )
        fixture.observation[observation_key]["post_restore_runtime_state"] = _runc_status(
            fixture.runtime_state_root,
            fixture.sandbox_id,
        )
        return checkpoint_result, restore_result

    def _assert_iflow_alive_after_restore(
        self,
        fixture: _RealIFlowManualFixture,
        *,
        observation_key: str,
        expect_no_tracked_pids: bool,
    ) -> dict[str, object]:
        reset_status = _status_for(fixture.host_client, fixture.sandbox_id)
        if not bool(reset_status.get("is_running")) or not bool(reset_status.get("metadata", {}).get("ignored_pids")):
            reset_status = fixture.host_client.reset_sandbox(fixture.sandbox_id, utc_now())["status"]
            reset_status = _status_for(fixture.host_client, fixture.sandbox_id)
        fixture.observation[observation_key]["post_restore_status"] = reset_status
        self.assertTrue(bool(reset_status["is_running"]), reset_status)
        self.assertTrue(bool(reset_status["metadata"].get("ignored_pids")), reset_status)
        if expect_no_tracked_pids:
            self.assertFalse(bool(reset_status["metadata"].get("tracked_pids")), reset_status)
        return reset_status

    def test_manual_iflow_cli_read_only_and_write_matrix(self) -> None:
        if (
            os.geteuid() != 0
            or shutil.which("docker") is None
            or shutil.which("runc") is None
            or shutil.which("zfs") is None
            or shutil.which("ip") is None
        ):
            self.skipTest("root plus docker/runc/zfs/ip required")

        helper_path = _ensure_helper_built()
        tmpdir = tempfile.TemporaryDirectory(prefix="agent_cr_iflow_manual_it_")
        self.addCleanup(tmpdir.cleanup)
        root = Path(tmpdir.name)
        sandbox_id = SandboxId("sbx-iflow-manual-real")
        manual_port = _find_free_port()
        inspector_server = HostInspectorServer(
            host="127.0.0.1",
            port=_find_free_port(),
            daemon=HostInspectorDaemon(
                resolver=RuntimeResolver(runc_state_root=root / "runtime-state"),
                process_poll_interval_s=0.2,
                fs_monitor=LibbpfFilesystemMonitor(helper_path=str(helper_path)),
            ),
        )
        inspector_server.start()
        self.addCleanup(inspector_server.stop)
        host_client = HostInspectorServiceClient(f"http://127.0.0.1:{inspector_server.port}")

        manual_server = serve_manual(host="0.0.0.0", port=manual_port, default_sandbox_id=str(sandbox_id))
        manual_thread = threading.Thread(target=manual_server.serve_forever, daemon=True)
        manual_thread.start()
        self.addCleanup(manual_server.shutdown)
        self.addCleanup(manual_server.server_close)
        self.addCleanup(manual_thread.join, 5.0)

        session = None
        try:
            session = launch_manual_iflow(
                work_root=root,
                llm_base_url=f"http://172.17.0.1:{manual_port}/v1",
                sandbox_id=str(sandbox_id),
                task_description="Wait for the next tool instruction, execute it exactly once, then ask for the next instruction.",
                host_inspector_url=f"http://127.0.0.1:{inspector_server.port}",
            )
            _wait_for_manual_turn(manual_server.manual_state, str(sandbox_id), minimum_turns=1, timeout_s=240.0)  # type: ignore[attr-defined]
            _wait_for_status(
                host_client,
                sandbox_id,
                predicate=lambda status: bool(status["is_running"]),
                timeout_s=60.0,
                interval_s=0.2,
            )

            cases = [
                {
                    "name": "read_only_ls",
                    "command": 'sh -lc "ls -la /work >/dev/null"',
                    "expected_filesystem_changed": False,
                },
                {
                    "name": "read_only_cat",
                    "command": 'sh -lc "cat /etc/hostname >/dev/null"',
                    "expected_filesystem_changed": False,
                },
                {
                    "name": "touch_file",
                    "command": 'sh -lc "touch 1.txt"',
                    "expected_filesystem_changed": True,
                },
                {
                    "name": "shell_redirection_write",
                    "command": 'sh -lc "echo 123 > foobat.txt"',
                    "expected_filesystem_changed": True,
                },
            ]

            for index, case in enumerate(cases, start=1):
                with self.subTest(case=case["name"]):
                    host_client.reset_sandbox(sandbox_id, utc_now())
                    _wait_for_status(
                        host_client,
                        sandbox_id,
                        predicate=lambda status: not bool(status["process_changed"]) and not bool(status["filesystem_changed"]),
                        timeout_s=30.0,
                        interval_s=0.2,
                    )
                    before_turns = int(manual_server.manual_state.snapshot()["turns"].get(str(sandbox_id), 0))  # type: ignore[attr-defined]
                    manual_server.manual_state.enqueue_run_shell_command(command=str(case["command"]), sandbox_id=str(sandbox_id))  # type: ignore[attr-defined]
                    _wait_for_manual_turn(
                        manual_server.manual_state,  # type: ignore[attr-defined]
                        str(sandbox_id),
                        minimum_turns=before_turns + 1,
                        timeout_s=120.0,
                    )
                    status = _status_for(host_client, sandbox_id)
                    self.assertFalse(
                        bool(status["process_changed"]),
                        f"{case['name']} unexpectedly latched process_changed; status={status}",
                    )
                    self.assertEqual(
                        bool(status["filesystem_changed"]),
                        bool(case["expected_filesystem_changed"]),
                        f"{case['name']} filesystem_changed mismatch; status={status}",
                    )
                    if bool(case["expected_filesystem_changed"]):
                        self.assertTrue(status["metadata"]["live_dirty_entries"], status)
                    else:
                        self.assertFalse(status["metadata"]["live_dirty_entries"], status)

            manual_server.manual_state.enqueue_final_response(  # type: ignore[attr-defined]
                sandbox_id=str(sandbox_id),
                content="The verification run is complete. Stop now.",
            )
        finally:
            if session is not None:
                try:
                    stop_manual_iflow(work_root=root, remove_image=False)
                except Exception:
                    pass

    def test_iflow_cli_tool_execution_survives_checkpoint_restore(self) -> None:
        fixture: _RealIFlowManualFixture | None = None
        fixture = self._create_real_iflow_manual_fixture(
            name="tool-restore",
            task_description="Wait for the next tool instruction, execute it exactly once, then ask for the next instruction.",
        )
        try:
            _wait_for_manual_turn_or_fail(
                fixture.manual_server.manual_state,  # type: ignore[attr-defined]
                str(fixture.sandbox_id),
                minimum_turns=1,
                runtime_state_root=fixture.runtime_state_root,
                runtime_sandbox_id=fixture.sandbox_id,
                host_client=fixture.host_client,
                logs_dir=fixture.logs_dir,
                timeout_s=240.0,
            )
            _wait_for_status(
                fixture.host_client,
                fixture.sandbox_id,
                predicate=lambda status: bool(status["is_running"]),
                timeout_s=60.0,
                interval_s=0.2,
            )

            cases = [
                {
                    "name": "counter_resume",
                    "command": (
                        "bash -lc 'mkdir -p /work/iflow-cr-counter; "
                        'count=0; '
                        'while [ "$count" -lt 6 ]; do '
                        'count=$((count + 1)); '
                        'printf "%s\\n" "$count" >/work/iflow-cr-counter/count.txt; '
                        'printf "%s\\n" "$count" >/work/iflow-cr-counter/progress.txt; '
                        'sleep 1; '
                        "done; printf done >/work/iflow-cr-counter/done.txt'"
                    ),
                    "progress_path": Path("/work/iflow-cr-counter/progress.txt"),
                    "verify": lambda root: (
                        (root / "iflow-cr-counter" / "count.txt").read_text(encoding="utf-8").strip(),
                        (root / "iflow-cr-counter" / "done.txt").read_text(encoding="utf-8").strip(),
                    ),
                },
                {
                    "name": "http_server_resume",
                    "command": (
                        "bash -lc 'mkdir -p /work/iflow-cr-http/site; "
                        'printf restored-http >/work/iflow-cr-http/site/index.txt; '
                        'python3 -m http.server 8124 -d /work/iflow-cr-http/site >/work/iflow-cr-http/http.log 2>&1 & '
                        'echo $! >/work/iflow-cr-http/server.pid; '
                        'printf ready >/work/iflow-cr-http/ready.txt; '
                        'count=0; '
                        'while [ "$count" -lt 8 ]; do '
                        'count=$((count + 1)); '
                        'printf "%s\\n" "$count" >/work/iflow-cr-http/progress.txt; '
                        'sleep 1; '
                        "done; kill $(cat /work/iflow-cr-http/server.pid) >/dev/null 2>&1 || true'"
                    ),
                    "progress_path": Path("/work/iflow-cr-http/progress.txt"),
                },
            ]

            for case in cases:
                with self.subTest(case=case["name"]):
                    self._wait_for_request_in_flight(fixture)
                    fixture.host_client.reset_sandbox(fixture.sandbox_id, utc_now())
                    fixture.observation["phases"][case["name"]] = {}

                    before_turns = int(
                        fixture.manual_server.manual_state.snapshot()["turns"].get(str(fixture.sandbox_id), 0)  # type: ignore[attr-defined]
                    )
                    fixture.manual_server.manual_state.enqueue_run_shell_command(  # type: ignore[attr-defined]
                        command=str(case["command"]),
                        sandbox_id=str(fixture.sandbox_id),
                    )

                    progress_path = fixture.work_dir / Path(case["progress_path"]).relative_to("/work")
                    if case["name"] == "counter_resume":
                        done_path = fixture.work_dir / "iflow-cr-counter" / "done.txt"
                        _wait_for_file_text(
                            progress_path,
                            predicate=lambda text: int(text.strip()) >= 3 and not done_path.exists(),
                            timeout_s=60.0,
                        )
                    else:
                        ready_path = fixture.work_dir / "iflow-cr-http" / "ready.txt"
                        _wait_for_file_text(
                            ready_path,
                            predicate=lambda text: text.strip() == "ready",
                            timeout_s=30.0,
                        )
                        _wait_for_file_text(
                            progress_path,
                            predicate=lambda text: int(text.strip()) >= 3,
                            timeout_s=60.0,
                        )
                    fixture.observation["phases"][case["name"]]["pre_checkpoint_progress"] = _read_text_if_exists(progress_path)

                    self._checkpoint_and_restore_fixture(
                        fixture,
                        use_scheduler=False,
                        observation_key=case["name"],
                    )
                    self._assert_iflow_alive_after_restore(
                        fixture,
                        observation_key=case["name"],
                        expect_no_tracked_pids=False,
                    )

                    if case["name"] == "counter_resume":
                        done_path = fixture.work_dir / "iflow-cr-counter" / "done.txt"
                        _wait_for_file_text(
                            done_path,
                            predicate=lambda text: text.strip() == "done",
                            timeout_s=120.0,
                        )
                        _wait_for_manual_turn_or_fail(
                            fixture.manual_server.manual_state,  # type: ignore[attr-defined]
                            str(fixture.sandbox_id),
                            minimum_turns=before_turns + 1,
                            runtime_state_root=fixture.runtime_state_root,
                            runtime_sandbox_id=fixture.sandbox_id,
                            host_client=fixture.host_client,
                            logs_dir=fixture.logs_dir,
                            timeout_s=120.0,
                        )
                        verification = case["verify"](fixture.work_dir)
                        fixture.observation["phases"][case["name"]]["verification"] = verification
                        count_value, done_value = verification
                        self.assertEqual(count_value, "6")
                        self.assertEqual(done_value, "done")
                    else:
                        verification = _wait_for_http_text(
                            f"http://{fixture.sandbox_ip}:8124/index.txt",
                            predicate=lambda body: body.strip() == "restored-http",
                            timeout_s=30.0,
                        )
                        fixture.observation["phases"][case["name"]]["verification"] = verification
                        _wait_for_manual_turn_or_fail(
                            fixture.manual_server.manual_state,  # type: ignore[attr-defined]
                            str(fixture.sandbox_id),
                            minimum_turns=before_turns + 1,
                            runtime_state_root=fixture.runtime_state_root,
                            runtime_sandbox_id=fixture.sandbox_id,
                            host_client=fixture.host_client,
                            logs_dir=fixture.logs_dir,
                            timeout_s=120.0,
                        )
                        self.assertEqual(str(verification).strip(), "restored-http")
        finally:
            if fixture is not None:
                self._cleanup_real_iflow_manual_fixture(fixture)

    def test_iflow_cli_llm_wait_survives_checkpoint_restore(self) -> None:
        if _runc_checkpoint_leaves_running():
            self.skipTest(
                "llm-wait restore requires a non-live process checkpoint; "
                "leave-running checkpoints do not preserve the in-flight LLM request across restore"
            )
        fixture: _RealIFlowManualFixture | None = None
        fixture = self._create_real_iflow_manual_fixture(
            name="llm-wait-restore",
            task_description="Wait for the next tool instruction, execute it exactly once, then ask for the next instruction.",
        )
        try:
            _wait_for_manual_turn_or_fail(
                fixture.manual_server.manual_state,  # type: ignore[attr-defined]
                str(fixture.sandbox_id),
                minimum_turns=1,
                runtime_state_root=fixture.runtime_state_root,
                runtime_sandbox_id=fixture.sandbox_id,
                host_client=fixture.host_client,
                logs_dir=fixture.logs_dir,
                timeout_s=240.0,
            )
            self._wait_for_request_in_flight(fixture)
            initial_reset = fixture.host_client.reset_sandbox(fixture.sandbox_id, utc_now())["status"]
            fixture.observation["phases"]["initial_reset"] = initial_reset

            before_first_tool_turns = int(
                fixture.manual_server.manual_state.snapshot()["turns"].get(str(fixture.sandbox_id), 0)  # type: ignore[attr-defined]
            )
            fixture.manual_server.manual_state.enqueue_run_shell_command(  # type: ignore[attr-defined]
                command='sh -lc "mkdir -p /work/iflow-llmwait && printf pre-restore >/work/iflow-llmwait/pre_restore.txt"',
                sandbox_id=str(fixture.sandbox_id),
            )
            _wait_for_manual_turn_or_fail(
                fixture.manual_server.manual_state,  # type: ignore[attr-defined]
                str(fixture.sandbox_id),
                minimum_turns=before_first_tool_turns + 1,
                runtime_state_root=fixture.runtime_state_root,
                runtime_sandbox_id=fixture.sandbox_id,
                host_client=fixture.host_client,
                logs_dir=fixture.logs_dir,
                timeout_s=120.0,
            )
            pre_restore_path = fixture.work_dir / "iflow-llmwait" / "pre_restore.txt"
            self.assertEqual(
                _wait_for_file_text(
                    pre_restore_path,
                    predicate=lambda text: text.strip() == "pre-restore",
                    timeout_s=30.0,
                ).strip(),
                "pre-restore",
            )
            self._wait_for_request_in_flight(fixture)
            llm_wait_status = _wait_for_status(
                fixture.host_client,
                fixture.sandbox_id,
                predicate=lambda status: bool(status["filesystem_changed"]) and bool(status["is_running"]),
                timeout_s=60.0,
                interval_s=0.2,
            )
            fixture.observation["phases"]["llm_wait_before_checkpoint"] = llm_wait_status

            self._checkpoint_and_restore_fixture(
                fixture,
                use_scheduler=True,
                observation_key="llm_wait_restore",
            )
            self.assertTrue(
                fixture.request_state_store.get(fixture.sandbox_id).llm_request_in_flight,
                fixture.request_state_store.get(fixture.sandbox_id),
            )
            self._assert_iflow_alive_after_restore(
                fixture,
                observation_key="llm_wait_restore",
                expect_no_tracked_pids=True,
            )

            before_post_restore_turns = int(
                fixture.manual_server.manual_state.snapshot()["turns"].get(str(fixture.sandbox_id), 0)  # type: ignore[attr-defined]
            )
            fixture.manual_server.manual_state.enqueue_run_shell_command(  # type: ignore[attr-defined]
                command='sh -lc "mkdir -p /work/iflow-llmwait && printf post-restore >/work/iflow-llmwait/post_restore.txt"',
                sandbox_id=str(fixture.sandbox_id),
            )
            _wait_for_manual_turn_or_fail(
                fixture.manual_server.manual_state,  # type: ignore[attr-defined]
                str(fixture.sandbox_id),
                minimum_turns=before_post_restore_turns + 1,
                runtime_state_root=fixture.runtime_state_root,
                runtime_sandbox_id=fixture.sandbox_id,
                host_client=fixture.host_client,
                logs_dir=fixture.logs_dir,
                timeout_s=120.0,
            )
            post_restore_path = fixture.work_dir / "iflow-llmwait" / "post_restore.txt"
            self.assertEqual(
                _wait_for_file_text(
                    post_restore_path,
                    predicate=lambda text: text.strip() == "post-restore",
                    timeout_s=30.0,
                ).strip(),
                "post-restore",
            )
            self.assertGreaterEqual(
                fixture.request_state_store.get(fixture.sandbox_id).total_llm_requests,
                3,
            )
        finally:
            if fixture is not None:
                self._cleanup_real_iflow_manual_fixture(fixture)

    def test_real_iflow_cli_host_inspector_matrix(self) -> None:
        if (
            os.geteuid() != 0
            or shutil.which("docker") is None
            or shutil.which("runc") is None
            or shutil.which("criu") is None
            or shutil.which("zfs") is None
            or shutil.which("ip") is None
        ):
            self.skipTest("root plus docker/runc/criu/zfs/ip required")

        cache_files = ensure_cache_files()
        helper_path = _ensure_helper_built()
        idle_delay_ms = int(os.environ.get("AGENT_CR_IFLOW_IDLE_DELAY_MS", "200"))

        if os.environ.get("AGENT_CR_KEEP_IFLOW_TMP", "0") == "1":
            root = Path(tempfile.mkdtemp(prefix="agent_cr_iflow_real_it_"))
        else:
            tmpdir = tempfile.TemporaryDirectory(prefix="agent_cr_iflow_real_it_")
            self.addCleanup(tmpdir.cleanup)
            root = Path(tmpdir.name)
        report_path = root / "iflow_observation.json"

        sandbox_id = SandboxId("sbx-iflow-real")
        unique_suffix = _unique_test_suffix()
        pool_name = f"agentcriflow{unique_suffix}"
        image_tag = f"agent-cr-iflow-agent:{unique_suffix}"
        task_description = "Run the requested verification commands exactly once and stop when done."
        bundle_dir = root / "bundles" / str(sandbox_id)
        runtime_state_root = root / "runtime-state"
        checkpoint_root = root / "checkpoints"
        storage_root = root / "storage"
        sandbox_metadata_root = root / "sandbox-meta"
        image_root = root / "image"
        pool_file = root / "zpool.img"
        work_dir = bundle_dir / "rootfs" / "work"
        observation: dict[str, Any] = {
            "task_description": task_description,
            "cache_files": {name: str(path) for name, path in cache_files.items()},
            "phases": {},
        }
        executor: CRExecutor | None = None
        system: AgentCRSystem | None = None
        host_client: HostInspectorServiceClient | None = None

        llm_server = serve(host="127.0.0.1", port=0, steps=default_script_steps(idle_delay_ms=idle_delay_ms))
        llm_thread = threading.Thread(target=llm_server.serve_forever, daemon=True)
        llm_thread.start()
        self.addCleanup(llm_server.shutdown)
        self.addCleanup(llm_server.server_close)
        self.addCleanup(llm_thread.join, 5.0)

        network = BridgeNetworkNamespace(
            name=f"agentcriflow-{unique_suffix}",
            ip_address=os.environ.get("AGENT_CR_IFLOW_SANDBOX_IP", "172.17.0.240"),
        )

        try:
            inspector_server = HostInspectorServer(
                host="127.0.0.1",
                port=_find_free_port(),
                daemon=HostInspectorDaemon(
                    resolver=RuntimeResolver(runc_state_root=runtime_state_root),
                    process_poll_interval_s=0.2,
                    fs_monitor=LibbpfFilesystemMonitor(helper_path=str(helper_path)),
                ),
            )
            inspector_server.start()
            self.addCleanup(inspector_server.stop)
            host_client = HostInspectorServiceClient(f"http://127.0.0.1:{inspector_server.port}")

            build_image(tag=image_tag)
            self.addCleanup(
                lambda: subprocess.run(
                    ["docker", "rmi", "-f", image_tag],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            exported_rootfs = export_image_rootfs(tag=image_tag, output_dir=image_root)
            alternate_node_runtime_dir_raw = os.environ.get("AGENT_CR_IFLOW_NODE_RUNTIME_DIR")
            prepared_runtime = prepare_iflow_runtime(
                work_root=root,
                cache_dir=cache_files["node-v22.18.0-linux-x64.tar.xz"].parent,
                alternate_node_runtime_dir=None
                if alternate_node_runtime_dir_raw is None
                else Path(alternate_node_runtime_dir_raw),
            )

            subprocess.run(["truncate", "-s", "1024M", str(pool_file)], check=True)
            subprocess.run(["zpool", "create", "-f", pool_name, str(pool_file)], check=True)
            self.addCleanup(
                lambda: subprocess.run(
                    ["zpool", "destroy", "-f", pool_name],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
            subprocess.run(["zfs", "create", f"{pool_name}/agent-cr"], check=True)

            bundle_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True)
            network.create()
            self.addCleanup(network.cleanup)

            telemetry = InMemoryTelemetrySink()
            request_state_store = InMemoryRequestStateStore()
            base_inspector = RemoteSandboxInspector(host_client)
            inspector = RequestAwareSandboxInspector(base_inspector, request_state_store)
            runtime = RuncRuntimeAdapter(
                paths=RuncRuntimePaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    checkpoint_root=checkpoint_root,
                    zfs_dataset_prefix=f"{pool_name}/agent-cr",
                )
            )
            storage = LocalCheckpointManager(StorageConfig(root_dir=storage_root))
            executor = CRExecutor(
                ExecutorConfig(max_workers=1),
                DefaultCWorker(
                    AdapterProcessCWorker(runtime),
                    AdapterFileSystemCWorker(runtime),
                    storage,
                    runtime,
                ),
                DefaultRWorker(
                    AdapterProcessRWorker(runtime),
                    AdapterFileSystemRWorker(runtime),
                    storage,
                ),
                telemetry,
            )
            sandbox_manager = RuncSandboxManager(
                host_inspector_client=host_client,
                paths=RuncSandboxManagerPaths(
                    state_root=runtime_state_root,
                    bundle_root=root / "bundles",
                    metadata_root=sandbox_metadata_root,
                    zfs_dataset_prefix=f"{pool_name}/agent-cr",
                ),
            )
            scheduler = CRScheduler(
                SchedulerConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                    prefer_checkpoint_during_llm_request=True,
                    require_llm_request_for_checkpoint=True,
                ),
                inspector,
                sandbox_manager,
                InMemorySchedulerStateStore(),
                telemetry,
            )
            system = AgentCRSystem(
                scheduler=scheduler,
                executor=executor,
                storage=storage,
                inspector=inspector,
                sandbox_manager=sandbox_manager,
                telemetry=telemetry,
                request_state_store=request_state_store,
            )
            interceptor = AgentCRRequestInterceptorServer(
                upstream_url=f"http://127.0.0.1:{llm_server.server_address[1]}",
                request_state_store=request_state_store,
                hook=CompositeRequestInterceptorHook([TelemetryRequestInterceptorHook(telemetry)]),
                on_state_change=system.notify_interceptor_state_change,
                default_sandbox_id=sandbox_id,
                host="0.0.0.0",
                port=0,
            )
            interceptor.start()
            self.addCleanup(interceptor.stop)
            _wait_for_http_json(f"http://127.0.0.1:{interceptor.port}/healthz")
            prepared_state = prepare_iflow_state(
                work_root=root,
                base_url=os.environ.get("AGENT_CR_IFLOW_BASE_URL", f"http://172.17.0.1:{interceptor.port}/v1"),
                model_name=os.environ.get("AGENT_CR_IFLOW_MODEL_NAME", "agent-cr-iflow-scripted"),
            )
            observation["runtime_strategy"] = prepared_runtime.runtime_strategy
            observation["runtime_root"] = str(prepared_runtime.root)
            observation["node_source"] = str(prepared_runtime.node_source)
            observation["mounted_state_paths"] = {
                "iflow_home": str(prepared_state.iflow_home),
                "npm_home": str(prepared_state.npm_home),
                "logs_dir": str(prepared_state.logs_dir),
            }
            observation["ignore_process_rules"] = prepared_runtime.ignore_process_rules

            write_bundle_config(
                bundle_dir=bundle_dir,
                interceptor_port=interceptor.port,
                cgroup_path=f"agent-cr-tests/{pool_name}/{sandbox_id}",
                sandbox_id=sandbox_id,
                task_description=task_description,
                prepared_runtime=prepared_runtime,
                prepared_state=prepared_state,
                network_namespace_path=network.namespace_path,
            )

            launched_id = sandbox_manager.launch(
                "runc",
                {
                    "sandbox_id": str(sandbox_id),
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
                        "root",
                        "root/.iflow",
                        "root/.npm",
                        "opt/iflow-runtime",
                        "opt/iflow-logs",
                    ],
                    "rootfs_copy_paths": rootfs_copy_paths(exported_rootfs=exported_rootfs),
                    "host_inspector_ignore_process_rules": prepared_runtime.ignore_process_rules,
                },
            )
            self.assertEqual(launched_id, sandbox_id)
            self.addCleanup(
                lambda: subprocess.run(
                    ["runc", "--root", str(runtime_state_root), "delete", "-f", str(sandbox_id)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

            _wait_for_phase_request_or_fail(
                llm_server.scripted_state,  # type: ignore[attr-defined]
                "transient_process",
                runtime_state_root=runtime_state_root,
                sandbox_id=sandbox_id,
                work_dir=work_dir,
                logs_dir=prepared_state.logs_dir,
                timeout_s=180.0,
            )
            _wait_for(lambda: request_state_store.get(sandbox_id).llm_request_in_flight, timeout_s=60.0, interval_s=0.1)
            initial_reset_status = host_client.reset_sandbox(sandbox_id, utc_now())["status"]
            initial_reset_status.setdefault("metadata", {})["current_process_identities"] = [
                _proc_identity(int(pid))
                for pid in initial_reset_status.get("metadata", {}).get("current_pids", [])
            ]
            observation["phases"]["post_initial_reset"] = dict(initial_reset_status)
            self.assertFalse(bool(initial_reset_status["process_changed"]))
            self.assertFalse(bool(initial_reset_status["filesystem_changed"]))
            self.assertTrue(bool(initial_reset_status["metadata"].get("ignored_pids")))
            self.assertFalse(bool(initial_reset_status["metadata"].get("tracked_pids")))

            idle_wait_status = _status_for(host_client, sandbox_id)
            observation["phases"]["idle_wait"] = idle_wait_status

            _wait_for_phase_request_or_fail(
                llm_server.scripted_state,  # type: ignore[attr-defined]
                "filesystem_write",
                runtime_state_root=runtime_state_root,
                sandbox_id=sandbox_id,
                work_dir=work_dir,
                logs_dir=prepared_state.logs_dir,
                timeout_s=180.0,
            )
            _wait_for(lambda: request_state_store.get(sandbox_id).llm_request_in_flight, timeout_s=60.0, interval_s=0.1)
            transient_status = _status_for(host_client, sandbox_id)
            observation["phases"]["transient_process"] = transient_status
            self.assertFalse(
                bool(transient_status["process_changed"]),
                f"transient process latched process_changed unexpectedly; report={report_path}",
            )
            self.assertFalse(
                bool(transient_status["filesystem_changed"]),
                f"transient process latched filesystem_changed unexpectedly; report={report_path}",
            )

            _wait_for_phase_request_or_fail(
                llm_server.scripted_state,  # type: ignore[attr-defined]
                "detached_daemon",
                runtime_state_root=runtime_state_root,
                sandbox_id=sandbox_id,
                work_dir=work_dir,
                logs_dir=prepared_state.logs_dir,
                timeout_s=180.0,
            )
            _wait_for(lambda: request_state_store.get(sandbox_id).llm_request_in_flight, timeout_s=60.0, interval_s=0.1)
            filesystem_status = _status_for(host_client, sandbox_id)
            observation["phases"]["filesystem_write"] = filesystem_status
            self.assertFalse(
                bool(filesystem_status["process_changed"]),
                f"filesystem phase latched process_changed unexpectedly; report={report_path}",
            )
            self.assertTrue(
                bool(filesystem_status["filesystem_changed"]),
                f"filesystem phase did not latch filesystem_changed; report={report_path}",
            )

            _wait_for_phase_request_or_fail(
                llm_server.scripted_state,  # type: ignore[attr-defined]
                "final_response",
                runtime_state_root=runtime_state_root,
                sandbox_id=sandbox_id,
                work_dir=work_dir,
                logs_dir=prepared_state.logs_dir,
                timeout_s=180.0,
            )
            _wait_for(lambda: request_state_store.get(sandbox_id).llm_request_in_flight, timeout_s=60.0, interval_s=0.1)
            daemon_status = _wait_for_status(
                host_client,
                sandbox_id,
                predicate=lambda status: bool(status["process_changed"]),
                timeout_s=90.0,
                interval_s=0.2,
            )
            observation["phases"]["detached_daemon"] = daemon_status
            self.assertTrue(
                bool(daemon_status["process_changed"]),
                f"detached daemon phase did not latch process_changed; report={report_path}",
            )
            self.assertTrue(
                bool(daemon_status["filesystem_changed"]),
                f"detached daemon phase did not latch filesystem_changed; report={report_path}",
            )

            pre_checkpoint_status = _status_for(host_client, sandbox_id)
            observation["phases"]["pre_checkpoint"] = pre_checkpoint_status

            checkpoint_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(checkpoint_result)
            assert checkpoint_result is not None
            observation["checkpoint"] = {
                "checkpoint_id": str(checkpoint_result.checkpoint_id),
                "status": checkpoint_result.status.value,
                "message": checkpoint_result.message,
                "operation_statuses": [
                    {
                        "phase": status.metadata.get("phase"),
                        "executed": bool(status.executed),
                        "reason": str(status.reason),
                    }
                    for status in checkpoint_result.operation_statuses
                ],
            }
            if alternate_node_runtime_dir_raw is not None:
                self.assertEqual(
                    checkpoint_result.status,
                    JobStatus.SUCCEEDED,
                    f"alternate node runtime did not make checkpoint succeed; report={report_path}",
                )
            if checkpoint_result.status == JobStatus.SUCCEEDED:
                observation["post_checkpoint_runtime_state"] = _runc_status(runtime_state_root, sandbox_id)
                subprocess.run(
                    ["runc", "--root", str(runtime_state_root), "delete", "-f", str(sandbox_id)],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                observation["runtime_state_after_delete"] = _wait_for(
                    lambda: (
                        status
                        if (status := _runc_status(runtime_state_root, sandbox_id)) == "missing"
                        else None
                    ),
                    timeout_s=30.0,
                    interval_s=0.2,
                )
                restore_result = system.restore_once(sandbox_id, checkpoint_result.checkpoint_id)
                self.assertEqual(restore_result.status, JobStatus.SUCCEEDED)
                observation["restore"] = {
                    "status": restore_result.status.value,
                    "checkpoint_id": str(restore_result.checkpoint_id),
                }
                post_restore_runtime_state = _runc_status(runtime_state_root, sandbox_id)
                observation["post_restore_runtime_state"] = post_restore_runtime_state
                reset_after_restore = host_client.reset_sandbox(sandbox_id, utc_now())["status"]
                observation["phases"]["post_restore_reset"] = reset_after_restore
                try:
                    post_checkpoint_status = _wait_for_status(
                        host_client,
                        sandbox_id,
                        predicate=lambda status: (
                            bool(status["is_running"])
                            and not bool(status["process_changed"])
                            and not bool(status["filesystem_changed"])
                        ),
                        timeout_s=20.0,
                        interval_s=0.2,
                    )
                    observation["phases"]["post_checkpoint_reset"] = post_checkpoint_status
                    observation["checkpoint_reset_supported"] = True
                except RuntimeError as exc:
                    restored_iflow_pids = _host_pids_with_cmdline(
                        "/opt/iflow-runtime/node/bin/node",
                        "@iflow-ai/iflow-cli/bundle/",
                    )
                    observation["post_restore_runtime_state_timeout"] = str(exc)
                    observation["restored_iflow_pids"] = restored_iflow_pids
                    observation["checkpoint_reset_supported"] = False
                    if not restored_iflow_pids or post_restore_runtime_state != "stopped":
                        raise
            else:
                checkpoint_logs = _checkpoint_log_excerpt(
                    checkpoint_root=checkpoint_root,
                    sandbox_id=sandbox_id,
                    checkpoint_id=str(checkpoint_result.checkpoint_id),
                )
                observation["checkpoint"]["log_excerpt"] = checkpoint_logs
                if checkpoint_logs is not None:
                    self.assertIn("anon_inode:[io_uring]", checkpoint_logs)
                else:
                    self.fail(f"checkpoint failed without dump logs; report={report_path}")

            _wait_for(
                lambda: len(
                    [
                        event
                        for event in llm_server.scripted_state.snapshot()["events"]  # type: ignore[attr-defined]
                        if event["event"] == "response"
                    ]
                )
                >= 4,
                timeout_s=90.0,
                interval_s=0.2,
            )

            expected_matrix = {
                "transient_process": {"process_changed": False, "filesystem_changed": False},
                "filesystem_write": {"process_changed": False, "filesystem_changed": True},
                "detached_daemon": {"process_changed": True, "filesystem_changed": True},
            }
            observation["checkpoint_reset_supported"] = bool(observation.get("checkpoint_reset_supported", False))
            observation["expected_matrix"] = expected_matrix
            observation["observed_matrix"] = {
                "transient_process": {
                    "process_changed": bool(transient_status["process_changed"]),
                    "filesystem_changed": bool(transient_status["filesystem_changed"]),
                },
                "filesystem_write": {
                    "process_changed": bool(filesystem_status["process_changed"]),
                    "filesystem_changed": bool(filesystem_status["filesystem_changed"]),
                },
                "detached_daemon": {
                    "process_changed": bool(daemon_status["process_changed"]),
                    "filesystem_changed": bool(daemon_status["filesystem_changed"]),
                },
            }
            for phase_name, expected_flags in expected_matrix.items():
                observed_flags = observation["observed_matrix"][phase_name]
                self.assertEqual(
                    observed_flags["process_changed"],
                    expected_flags["process_changed"],
                    f"{phase_name} process_changed mismatch; report={report_path}",
                )
                if expected_flags["filesystem_changed"] is not None:
                    self.assertEqual(
                        observed_flags["filesystem_changed"],
                        expected_flags["filesystem_changed"],
                        f"{phase_name} filesystem_changed mismatch; report={report_path}",
                    )

            event_names = [name for name, _ in telemetry.events]
            self.assertIn("request.start", event_names)
            self.assertIn("request.end", event_names)
            self.assertIn("scheduler.evaluate", event_names)
            self.assertGreaterEqual(request_state_store.get(sandbox_id).total_llm_requests, 4)

        finally:
            root_dir = bundle_dir / "rootfs" / "root"
            observation["scripted_llm"] = llm_server.scripted_state.snapshot()  # type: ignore[attr-defined]
            observation["work_tree"] = _report_tree(work_dir)
            observation["root_tree"] = _report_tree(root_dir)
            observation["iflow_stdout"] = _read_text_if_exists(root / "iflow-state" / "logs" / "iflow.stdout")
            observation["iflow_stderr"] = _read_text_if_exists(root / "iflow-state" / "logs" / "iflow.stderr")
            report_path.write_text(json.dumps(observation, indent=2, sort_keys=True), encoding="utf-8")
            if system is not None:
                try:
                    system.sandbox_manager.delete(sandbox_id)
                except Exception:
                    pass
            if executor is not None:
                executor.shutdown()
            subprocess.run(
                ["runc", "--root", str(runtime_state_root), "delete", "-f", str(sandbox_id)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zfs", "destroy", "-r", f"{pool_name}/agent-cr/{sandbox_id}"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zfs", "destroy", "-r", f"{pool_name}/agent-cr"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                ["zpool", "destroy", "-f", pool_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )


if __name__ == "__main__":
    unittest.main()

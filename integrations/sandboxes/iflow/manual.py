from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from agent_cr import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    AgentCRSystem,
    CRExecutor,
    CRScheduler,
    CheckpointId,
    DefaultCWorker,
    DefaultRWorker,
    ExecutorConfig,
    HostInspectorServiceClient,
    InMemoryRequestStateStore,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    RemoteSandboxInspector,
    RuncRuntimeAdapter,
    RuncRuntimePaths,
    RuncSandboxManager,
    RuncSandboxManagerPaths,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
    StorageConfig,
)
from agent_cr.models import utc_now

from .harness import (
    BridgeNetworkNamespace,
    prepare_iflow_runtime,
    prepare_iflow_state,
    rootfs_copy_paths,
    write_bundle_config,
)
from . import DOCKERFILE_PATH
from ..image import build_image, export_image_rootfs


@dataclass(frozen=True)
class ManualIFlowSession:
    work_root: str
    sandbox_id: str
    image_tag: str
    pool_name: str
    pool_file: str
    bundle_dir: str
    runtime_state_root: str
    metadata_root: str
    checkpoint_root: str
    image_root: str
    llm_base_url: str
    host_inspector_url: str | None
    network_name: str
    sandbox_ip: str
    runtime_root: str
    runtime_strategy: str
    node_source: str
    iflow_home: str
    npm_home: str
    logs_dir: str
    ignore_process_rules: list[dict[str, object]]
    task_description: str
    created_at: float


class _NoopManualInspector:
    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        return SandboxSnapshot(
            sandbox_id=sandbox_id,
            runtime_name="manual-noop-inspector",
            is_running=True,
            process_changed=True,
            filesystem_changed=True,
            observed_at=utc_now(),
        )

    def mark_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        *,
        process: bool,
        filesystem: bool,
        at,
    ) -> None:
        _ = (sandbox_id, process, filesystem, at)
        return


def session_file(work_root: Path) -> Path:
    return work_root / "manual_session.json"


def load_session(work_root: Path) -> ManualIFlowSession:
    payload = json.loads(session_file(work_root).read_text())
    return ManualIFlowSession(**payload)


def launch_manual_iflow(
    *,
    work_root: Path,
    llm_base_url: str,
    sandbox_id: str,
    task_description: str,
    host_inspector_url: str | None = None,
    image_tag: str | None = None,
    model_name: str | None = None,
    sandbox_ip: str | None = None,
    alternate_node_runtime_dir: Path | None = None,
) -> ManualIFlowSession:
    _require_root_and_tools()
    if session_file(work_root).exists():
        raise FileExistsError(f"manual session already exists: {session_file(work_root)}")

    work_root.mkdir(parents=True, exist_ok=True)
    sandbox = SandboxId(sandbox_id)
    image_tag = image_tag or f"agent-cr-iflow-manual:{int(time.time())}"
    pool_name = f"agentcriflowmanual{int(time.time())}"
    bundle_dir = work_root / "bundles" / str(sandbox)
    runtime_state_root = work_root / "runtime-state"
    metadata_root = work_root / "sandbox-meta"
    checkpoint_root = work_root / "checkpoints"
    image_root = work_root / "image"
    pool_file = work_root / "zpool.img"
    sandbox_ip = sandbox_ip or os.environ.get("AGENT_CR_IFLOW_SANDBOX_IP", "172.17.0.240")
    network = BridgeNetworkNamespace(
        name=f"agentcriflow-manual-{int(time.time())}",
        ip_address=sandbox_ip,
    )

    image_built = False
    pool_created = False
    network_created = False
    launched = False
    try:
        build_image(
            tag=image_tag,
            build_context=Path(__file__).resolve().parents[3],
            dockerfile_path=DOCKERFILE_PATH,
        )
        image_built = True
        exported_rootfs = export_image_rootfs(tag=image_tag, output_dir=image_root)
        prepared_runtime = prepare_iflow_runtime(
            work_root=work_root,
            alternate_node_runtime_dir=alternate_node_runtime_dir,
        )
        prepared_state = prepare_iflow_state(
            work_root=work_root,
            base_url=llm_base_url,
            model_name=model_name or os.environ.get("AGENT_CR_IFLOW_MODEL_NAME", "agent-cr-iflow-manual"),
        )

        subprocess.run(["truncate", "-s", "1024M", str(pool_file)], check=True)
        subprocess.run(["zpool", "create", "-f", pool_name, str(pool_file)], check=True)
        pool_created = True
        subprocess.run(["zfs", "create", f"{pool_name}/agent-cr"], check=True)

        bundle_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["runc", "spec"], cwd=bundle_dir, check=True)
        network.create()
        network_created = True

        write_bundle_config(
            bundle_dir=bundle_dir,
            interceptor_port=0,
            cgroup_path=f"agent-cr-manual/{pool_name}/{sandbox}",
            sandbox_id=sandbox,
            task_description=task_description,
            prepared_runtime=prepared_runtime,
            prepared_state=prepared_state,
            network_namespace_path=network.namespace_path,
            base_url=llm_base_url,
        )

        host_client = None if host_inspector_url is None else HostInspectorServiceClient(host_inspector_url)
        sandbox_manager = RuncSandboxManager(
            host_inspector_client=host_client,
            paths=RuncSandboxManagerPaths(
                state_root=runtime_state_root,
                bundle_root=work_root / "bundles",
                metadata_root=metadata_root,
                zfs_dataset_prefix=f"{pool_name}/agent-cr",
            ),
        )
        sandbox_manager.launch(
            "runc",
            {
                "sandbox_id": str(sandbox),
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
        launched = True

        session = ManualIFlowSession(
            work_root=str(work_root),
            sandbox_id=str(sandbox),
            image_tag=image_tag,
            pool_name=pool_name,
            pool_file=str(pool_file),
            bundle_dir=str(bundle_dir),
            runtime_state_root=str(runtime_state_root),
            metadata_root=str(metadata_root),
            checkpoint_root=str(checkpoint_root),
            image_root=str(image_root),
            llm_base_url=llm_base_url,
            host_inspector_url=host_inspector_url,
            network_name=network.name,
            sandbox_ip=sandbox_ip,
            runtime_root=str(prepared_runtime.root),
            runtime_strategy=prepared_runtime.runtime_strategy,
            node_source=str(prepared_runtime.node_source),
            iflow_home=str(prepared_state.iflow_home),
            npm_home=str(prepared_state.npm_home),
            logs_dir=str(prepared_state.logs_dir),
            ignore_process_rules=prepared_runtime.ignore_process_rules,
            task_description=task_description,
            created_at=time.time(),
        )
        session_file(work_root).write_text(json.dumps(asdict(session), indent=2, sort_keys=True))
        return session
    except Exception:
        if launched:
            _best_effort_delete_sandbox(work_root=work_root, sandbox_id=sandbox_id, host_inspector_url=host_inspector_url)
        if network_created:
            network.cleanup()
        if pool_created:
            subprocess.run(["zpool", "destroy", "-f", pool_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if image_built:
            subprocess.run(["docker", "rmi", "-f", image_tag], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise


def stop_manual_iflow(*, work_root: Path, remove_image: bool = False) -> ManualIFlowSession:
    session = load_session(work_root)
    _best_effort_delete_sandbox(
        work_root=work_root,
        sandbox_id=session.sandbox_id,
        host_inspector_url=session.host_inspector_url,
    )
    BridgeNetworkNamespace(name=session.network_name, ip_address=session.sandbox_ip).cleanup()
    subprocess.run(["zpool", "destroy", "-f", session.pool_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if remove_image:
        subprocess.run(["docker", "rmi", "-f", session.image_tag], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    session_file(work_root).unlink(missing_ok=True)
    return session


def session_summary(session: ManualIFlowSession) -> dict[str, Any]:
    return {
        "sandbox_id": session.sandbox_id,
        "work_root": session.work_root,
        "llm_base_url": session.llm_base_url,
        "host_inspector_url": session.host_inspector_url,
        "logs_dir": session.logs_dir,
        "runtime_state_root": session.runtime_state_root,
        "network_name": session.network_name,
        "sandbox_ip": session.sandbox_ip,
        "ignore_process_rules": session.ignore_process_rules,
        "task_description": session.task_description,
    }


def list_manual_checkpoints(*, work_root: Path) -> list[str]:
    session = load_session(work_root)
    _, _, executor, storage = _build_manual_system(work_root=work_root, session=session)
    try:
        return [str(item) for item in storage.list_checkpoints(SandboxId(session.sandbox_id))]
    finally:
        executor.shutdown()


def checkpoint_manual_iflow(*, work_root: Path) -> dict[str, Any]:
    session = load_session(work_root)
    system, _, executor, storage = _build_manual_system(work_root=work_root, session=session)
    try:
        result = system.checkpoint_once(SandboxId(session.sandbox_id))
        return {
            "sandbox_id": session.sandbox_id,
            "checkpoint_id": str(result.checkpoint_id),
            "status": result.status.value,
            "message": result.message,
            "available_checkpoints": [str(item) for item in storage.list_checkpoints(SandboxId(session.sandbox_id))],
        }
    finally:
        executor.shutdown()


def restore_manual_iflow(*, work_root: Path, checkpoint_id: str | None = None) -> dict[str, Any]:
    session = load_session(work_root)
    system, _, executor, storage = _build_manual_system(work_root=work_root, session=session)
    try:
        target = CheckpointId(checkpoint_id) if checkpoint_id is not None else _latest_checkpoint_id(
            storage=storage,
            sandbox_id=SandboxId(session.sandbox_id),
        )
        result = system.restore_once(SandboxId(session.sandbox_id), target)
        return {
            "sandbox_id": session.sandbox_id,
            "checkpoint_id": str(target),
            "status": result.status.value,
            "message": result.message,
        }
    finally:
        executor.shutdown()


def manual_shell(*, work_root: Path, command: list[str] | None = None) -> int:
    session = load_session(work_root)
    argv = command or ["sh"]
    exec_command = ["runc", "--root", session.runtime_state_root, "exec"]
    if os.isatty(0) and os.isatty(1):
        exec_command.append("-t")
    exec_command.extend([session.sandbox_id, *argv])
    return subprocess.run(exec_command, check=False).returncode


def _latest_checkpoint_id(*, storage: LocalCheckpointManager, sandbox_id: SandboxId) -> CheckpointId:
    checkpoints = storage.list_checkpoints(sandbox_id)
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints found for sandbox {sandbox_id}")
    return checkpoints[-1]


def _build_manual_system(
    *,
    work_root: Path,
    session: ManualIFlowSession,
) -> tuple[AgentCRSystem, RuncSandboxManager, CRExecutor, LocalCheckpointManager]:
    host_client = None if session.host_inspector_url is None else HostInspectorServiceClient(session.host_inspector_url)
    inspector = RemoteSandboxInspector(host_client) if host_client is not None else _NoopManualInspector()
    telemetry = InMemoryTelemetrySink()
    request_state_store = InMemoryRequestStateStore()
    runtime = RuncRuntimeAdapter(
        paths=RuncRuntimePaths(
            state_root=Path(session.runtime_state_root),
            bundle_root=work_root / "bundles",
            checkpoint_root=work_root / "checkpoints",
            zfs_dataset_prefix=f"{session.pool_name}/agent-cr",
        )
    )
    storage = LocalCheckpointManager(StorageConfig(root_dir=work_root / "storage"))
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
            state_root=Path(session.runtime_state_root),
            bundle_root=work_root / "bundles",
            metadata_root=work_root / "sandbox-meta",
            zfs_dataset_prefix=f"{session.pool_name}/agent-cr",
        ),
    )
    scheduler = CRScheduler(
        SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=False,
            prefer_checkpoint_during_llm_request=False,
            require_llm_request_for_checkpoint=False,
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
    return system, sandbox_manager, executor, storage


def _best_effort_delete_sandbox(*, work_root: Path, sandbox_id: str, host_inspector_url: str | None) -> None:
    host_client = None if host_inspector_url is None else HostInspectorServiceClient(host_inspector_url)
    manager = RuncSandboxManager(
        host_inspector_client=host_client,
        paths=RuncSandboxManagerPaths(
            state_root=work_root / "runtime-state",
            bundle_root=work_root / "bundles",
            metadata_root=work_root / "sandbox-meta",
            zfs_dataset_prefix="unused/manual-stop",
        ),
    )
    try:
        manager.delete(SandboxId(sandbox_id))
    except Exception:
        subprocess.run(
            ["runc", "--root", str(work_root / "runtime-state"), "delete", "-f", sandbox_id],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if host_client is not None:
            try:
                host_client.unregister_sandbox(SandboxId(sandbox_id))
            except Exception:
                pass


def _require_root_and_tools() -> None:
    if os.geteuid() != 0:
        raise PermissionError("manual iflow sandbox launch requires root")
    for tool in ("docker", "runc", "zfs", "zpool", "ip", "truncate"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"required tool not found: {tool}")

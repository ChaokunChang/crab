from __future__ import annotations

import json
import logging
import shutil
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Any

from .contracts import SandboxManager, TelemetrySink
from .ids import CheckpointId, SandboxId
from .models import RuntimeOperationStatus, SandboxDescription, SandboxExecResult, SandboxRuntimeState
from .remote_inspector import HostInspectorServiceClient
from .runtime.base import CommandRunner, CommandResult, SubprocessCommandRunner
from .telemetry import NoopTelemetrySink

if TYPE_CHECKING:
    from .runtime.runc import RuncCheckpointOptions, RuncRestoreOptions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuncSandboxManagerPaths:
    state_root: Path = Path("/run/agent-cr/runc")
    bundle_root: Path = Path("/var/lib/agent-cr/bundles")
    checkpoint_root: Path = Path("/var/lib/agent-cr/checkpoints")
    metadata_root: Path = Path("/var/lib/agent-cr/sandbox-metadata")
    zfs_dataset_prefix: str = "agentcr/sandboxes"


class InMemorySandboxManager(SandboxManager):
    def __init__(self, host_inspector_client: HostInspectorServiceClient | None = None) -> None:
        self._lock = Lock()
        self._items: dict[SandboxId, SandboxDescription] = {}
        self._host_inspector_client = host_inspector_client

    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        with self._lock:
            sandbox_id = SandboxId.new()
            self._items[sandbox_id] = SandboxDescription(
                sandbox_id=sandbox_id,
                runtime_name=runtime_name,
                status="running",
                metadata=dict(metadata or {}),
            )
        logger.info("Launched in-memory sandbox %s with runtime=%s", sandbox_id, runtime_name)
        return sandbox_id

    def stop(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="stopped")
        logger.info("Stopped in-memory sandbox %s", sandbox_id)

    def pause(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="paused")
        logger.info("Paused in-memory sandbox %s", sandbox_id)

    def resume(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="running")
        logger.info("Resumed in-memory sandbox %s", sandbox_id)

    def sync_runtime_state(self, sandbox_id: SandboxId, *, is_running: bool) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="running" if is_running else "stopped")
        logger.info("Synced in-memory sandbox %s runtime state running=%s", sandbox_id, is_running)

    def prepare_for_restore(self, sandbox_id: SandboxId) -> None:
        logger.info("Prepared in-memory sandbox %s for restore", sandbox_id)

    def mark_restored(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="running")
        logger.info("Marked in-memory sandbox %s restored", sandbox_id)

    def delete(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            self._items.pop(sandbox_id)
        logger.info("Deleted in-memory sandbox %s", sandbox_id)

    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        with self._lock:
            return self._items[sandbox_id]

    def write_bundle_spec(self, bundle_dir: Path) -> None:
        raise NotImplementedError("bundle spec generation is only supported for runc sandboxes")

    def inspect_runtime(self, sandbox_id: SandboxId) -> SandboxRuntimeState:
        raise NotImplementedError("runtime inspection is only supported for runc sandboxes")

    def exec(
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        _ = (sandbox_id, argv, cwd, env, user, timeout_s, capture_output)
        raise NotImplementedError("sandbox exec is only supported for runc sandboxes")

    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ) -> RuntimeOperationStatus:
        _ = (sandbox_id, checkpoint_id, leave_running)
        raise NotImplementedError("process checkpoint is only supported for runc sandboxes")

    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        _ = (sandbox_id, checkpoint_id)
        raise NotImplementedError("process restore is only supported for runc sandboxes")

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        _ = (sandbox_id, checkpoint_id)
        raise NotImplementedError("filesystem checkpoint is only supported for runc sandboxes")

    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        _ = (sandbox_id, checkpoint_id)
        raise NotImplementedError("filesystem restore is only supported for runc sandboxes")

    def delete_runtime(
        self,
        sandbox_id: SandboxId,
        *,
        force: bool = True,
        ignore_missing: bool = True,
    ) -> None:
        _ = (sandbox_id, force, ignore_missing)
        raise NotImplementedError("runtime deletion is only supported for runc sandboxes")

    def destroy_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        _ = sandbox_id
        raise NotImplementedError("filesystem dataset management is only supported for runc sandboxes")

    def clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path: Path,
    ) -> str:
        _ = (source_sandbox_id, checkpoint_id, target_sandbox_id, target_rootfs_path)
        raise NotImplementedError("filesystem cloning is only supported for runc sandboxes")


class RuncSandboxManager(SandboxManager):
    def __init__(
        self,
        *,
        paths: RuncSandboxManagerPaths | None = None,
        command_runner: CommandRunner | None = None,
        runtime_bin: str = "runc",
        zfs_bin: str = "zfs",
        host_inspector_client: HostInspectorServiceClient | None = None,
        telemetry: TelemetrySink | None = None,
        checkpoint_options: RuncCheckpointOptions | None = None,
        restore_options: RuncRestoreOptions | None = None,
    ) -> None:
        if checkpoint_options is None or restore_options is None:
            from .runtime.runc import RuncCheckpointOptions, RuncRestoreOptions

            checkpoint_options = checkpoint_options or RuncCheckpointOptions()
            restore_options = restore_options or RuncRestoreOptions()
        self._paths = paths or RuncSandboxManagerPaths()
        self._runner = command_runner or SubprocessCommandRunner()
        self._runtime_bin = runtime_bin
        self._zfs_bin = zfs_bin
        self._host_inspector_client = host_inspector_client
        self._telemetry = telemetry or NoopTelemetrySink()
        self._checkpoint_options = checkpoint_options
        self._restore_options = restore_options
        self._lock = Lock()
        self._items: dict[SandboxId, SandboxDescription] = {}
        self._paths.metadata_root.mkdir(parents=True, exist_ok=True)
        self._paths.checkpoint_root.mkdir(parents=True, exist_ok=True)

    @property
    def paths(self) -> RuncSandboxManagerPaths:
        return self._paths

    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        if runtime_name != "runc":
            raise ValueError(f"unsupported runtime for real sandbox manager: {runtime_name}")

        sandbox_id = SandboxId(str((metadata or {}).get("sandbox_id", SandboxId.new())))
        md = dict(metadata or {})
        bundle_path = Path(str(md["bundle_path"])) if "bundle_path" in md else self._paths.bundle_root / str(sandbox_id)
        rootfs_path = bundle_path / "rootfs"
        dataset = str(md.get("zfs_dataset", f"{self._paths.zfs_dataset_prefix}/{sandbox_id}"))
        logger.info("Launching runc sandbox %s with bundle=%s dataset=%s", sandbox_id, bundle_path, dataset)

        bundle_path.mkdir(parents=True, exist_ok=True)
        rootfs_path.mkdir(parents=True, exist_ok=True)
        self._run_command(
            [self._zfs_bin, "create", "-o", f"mountpoint={rootfs_path}", dataset],
            operation="sandbox.zfs_create",
            sandbox_id=sandbox_id,
            metadata={"dataset": dataset, "mountpoint": str(rootfs_path)},
        )
        for rel in md.get("rootfs_init_dirs", []):
            (rootfs_path / str(rel)).mkdir(parents=True, exist_ok=True)
        for item in md.get("rootfs_copy_paths", []):
            source = Path(str(item["source"]))
            destination = rootfs_path / str(item["destination"]).lstrip("/")
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination, follow_symlinks=True)
        self._run_command(
            [self._runtime_bin, "--root", str(self._paths.state_root), "create", "--bundle", str(bundle_path), str(sandbox_id)],
            operation="sandbox.runtime_create",
            sandbox_id=sandbox_id,
            metadata={"bundle_path": str(bundle_path)},
        )
        try:
            self._run_command(
                [self._runtime_bin, "--root", str(self._paths.state_root), "start", str(sandbox_id)],
                operation="sandbox.runtime_start",
                sandbox_id=sandbox_id,
            )
        except Exception:
            try:
                self.delete_runtime(sandbox_id, force=True, ignore_missing=True)
            except Exception:
                logger.exception("Failed to clean up sandbox %s after start failure", sandbox_id)
            raise

        description = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name=runtime_name,
            status="running",
            metadata={
                **md,
                "bundle_path": str(bundle_path),
                "rootfs_path": str(rootfs_path),
                "zfs_dataset": dataset,
            },
        )
        with self._lock:
            self._items[sandbox_id] = description
        self._persist(description)
        self._register_with_host_inspector(description)
        logger.info("Sandbox %s is running with rootfs=%s", sandbox_id, rootfs_path)
        return sandbox_id

    def stop(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        logger.info("Stopping sandbox %s", sandbox_id)
        self._run_command(
            [self._runtime_bin, "--root", str(self._paths.state_root), "kill", str(sandbox_id), "TERM"],
            operation="sandbox.runtime_kill",
            sandbox_id=sandbox_id,
        )
        updated = replace(description, status="stopped")
        self._update_description(updated)
        logger.info("Sandbox %s stopped", sandbox_id)

    def pause(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        logger.info("Pausing sandbox %s", sandbox_id)
        try:
            self._run_command(
                [self._runtime_bin, "--root", str(self._paths.state_root), "pause", str(sandbox_id)],
                operation="sandbox.runtime_pause",
                sandbox_id=sandbox_id,
                expected_error_substrings=("container not running", "container does not exist"),
            )
        except RuntimeError as exc:
            if "container not running" in str(exc) or "container does not exist" in str(exc):
                self.sync_runtime_state(sandbox_id, is_running=False)
            raise
        self._update_description(replace(description, status="paused"))
        logger.info("Sandbox %s paused", sandbox_id)

    def resume(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        logger.info("Resuming sandbox %s", sandbox_id)
        self._run_command(
            [self._runtime_bin, "--root", str(self._paths.state_root), "resume", str(sandbox_id)],
            operation="sandbox.runtime_resume",
            sandbox_id=sandbox_id,
        )
        self._update_description(replace(description, status="running"))
        logger.info("Sandbox %s resumed", sandbox_id)

    def sync_runtime_state(self, sandbox_id: SandboxId, *, is_running: bool) -> None:
        description = self.describe(sandbox_id)
        self._update_description(replace(description, status="running" if is_running else "stopped"))
        logger.info("Synced sandbox %s runtime state running=%s", sandbox_id, is_running)

    def prepare_for_restore(self, sandbox_id: SandboxId) -> None:
        logger.info("Preparing sandbox %s for restore", sandbox_id)
        self.delete_runtime(sandbox_id, force=True, ignore_missing=True)
        logger.info("Sandbox %s runtime state cleared for restore", sandbox_id)

    def mark_restored(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        updated = replace(description, status="running")
        self._update_description(updated)
        self._register_with_host_inspector(updated)
        logger.info("Sandbox %s marked running after restore", sandbox_id)

    def delete(self, sandbox_id: SandboxId) -> None:
        logger.info("Deleting sandbox %s", sandbox_id)
        self.delete_runtime(sandbox_id, force=True, ignore_missing=True)
        self.destroy_filesystem_dataset(sandbox_id)
        metadata_path = self._metadata_path(sandbox_id)
        if metadata_path.exists():
            metadata_path.unlink()
        with self._lock:
            self._items.pop(sandbox_id, None)
        if self._host_inspector_client is not None:
            try:
                self._host_inspector_client.unregister_sandbox(sandbox_id)
            except Exception:
                logger.exception("Failed to unregister sandbox %s from host inspector", sandbox_id)
        logger.info("Sandbox %s deleted", sandbox_id)

    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        with self._lock:
            current = self._items.get(sandbox_id)
        if current is not None:
            return current
        path = self._metadata_path(sandbox_id)
        if not path.exists():
            raise KeyError(sandbox_id)
        raw = json.loads(path.read_text(encoding="utf-8"))
        description = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name=str(raw["runtime_name"]),
            status=str(raw["status"]),
            metadata=dict(raw.get("metadata", {})),
        )
        with self._lock:
            self._items[sandbox_id] = description
        logger.debug("Loaded sandbox %s description from %s", sandbox_id, path)
        return description

    def write_bundle_spec(self, bundle_dir: Path) -> None:
        self._run_command(
            [self._runtime_bin, "spec"],
            operation="sandbox.bundle_spec",
            cwd=bundle_dir,
            metadata={"bundle_dir": str(bundle_dir)},
        )

    def inspect_runtime(self, sandbox_id: SandboxId) -> SandboxRuntimeState:
        description = self._try_describe(sandbox_id)
        result = self._run_command(
            [self._runtime_bin, "--root", str(self._paths.state_root), "state", str(sandbox_id)],
            operation="sandbox.runtime_state",
            sandbox_id=sandbox_id,
            check=False,
        )
        if result.returncode != 0:
            status = "missing"
            return SandboxRuntimeState(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                status=status,
                pid=None,
                bundle_path=None if description is None else str(description.metadata.get("bundle_path", "")) or None,
                metadata={
                    "stdout": result.stdout.strip(),
                    "stderr": result.stderr.strip(),
                    "returncode": result.returncode,
                },
            )
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            payload = {}
        pid = int(payload.get("pid", 0) or 0)
        return SandboxRuntimeState(
            sandbox_id=sandbox_id,
            runtime_name="runc",
            status=str(payload.get("status", "unknown")),
            pid=pid if pid > 0 else None,
            bundle_path=None if description is None else str(description.metadata.get("bundle_path", "")) or None,
            metadata={
                "payload": payload,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            },
        )

    def exec(
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        command = [self._runtime_bin, "--root", str(self._paths.state_root), "exec"]
        if cwd:
            command.extend(["--cwd", cwd])
        if user:
            command.extend(["--user", user])
        for key, value in sorted((env or {}).items()):
            command.extend(["--env", f"{key}={value}"])
        command.append(str(sandbox_id))
        command.extend(argv)
        started = time.perf_counter()
        stdout_target: int | Any = subprocess.PIPE if capture_output else subprocess.DEVNULL
        stderr_target: int | Any = subprocess.PIPE if capture_output else subprocess.DEVNULL
        completed = subprocess.run(
            command,
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=stdout_target,
            stderr=stderr_target,
            text=True,
            timeout=timeout_s,
        )
        duration_ms = (time.perf_counter() - started) * 1000.0
        stdout = "" if completed.stdout is None else completed.stdout
        stderr = "" if completed.stderr is None else completed.stderr
        self._emit_command_telemetry(
            "sandbox.runtime_exec",
            command=command,
            duration_ms=duration_ms,
            sandbox_id=sandbox_id,
            success=completed.returncode == 0,
            metadata={"cwd": cwd, "user": user, "capture_output": capture_output},
            stdout=stdout,
            stderr=stderr,
        )
        return SandboxExecResult(
            args=tuple(command),
            returncode=int(completed.returncode),
            stdout=stdout,
            stderr=stderr,
        )

    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ) -> RuntimeOperationStatus:
        image_path = self.process_checkpoint_location(sandbox_id, checkpoint_id)
        work_path = self.process_work_path(sandbox_id, checkpoint_id)
        image_path.mkdir(parents=True, exist_ok=True)
        work_path.mkdir(parents=True, exist_ok=True)
        command = [
            self._runtime_bin,
            "--root",
            str(self._paths.state_root),
            "checkpoint",
            "--image-path",
            str(image_path),
            "--work-path",
            str(work_path),
            f"--leave-running={'true' if leave_running else 'false'}",
        ]
        command.extend(self._checkpoint_optional_args())
        command.append(str(sandbox_id))
        return self._run_status(
            command,
            operation="sandbox.checkpoint_process",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={
                "phase": "process_checkpoint",
                "runtime": "runc",
                "image_path": str(image_path),
                "work_path": str(work_path),
                "bundle_path": str(self.bundle_path_for(sandbox_id)),
                "state_root": str(self._paths.state_root),
                "leave_running": leave_running,
            },
        )

    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        image_path = self.process_checkpoint_location(sandbox_id, checkpoint_id)
        work_path = self.process_work_path(sandbox_id, checkpoint_id)
        image_path.mkdir(parents=True, exist_ok=True)
        work_path.mkdir(parents=True, exist_ok=True)
        command = [self._runtime_bin, "--root", str(self._paths.state_root), "restore"]
        if self._restore_options.detach:
            command.append("-d")
        command.extend(
            [
                "--bundle",
                str(self.bundle_path_for(sandbox_id)),
                "--image-path",
                str(image_path),
                "--work-path",
                str(work_path),
            ]
        )
        command.extend(self._restore_optional_args())
        command.append(str(sandbox_id))
        return self._run_status(
            command,
            operation="sandbox.restore_process",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={
                "phase": "process_restore",
                "runtime": "runc",
                "image_path": str(image_path),
                "work_path": str(work_path),
                "bundle_path": str(self.bundle_path_for(sandbox_id)),
                "state_root": str(self._paths.state_root),
            },
        )

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        dataset = self.dataset_name_for(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        return self._run_status(
            [self._zfs_bin, "snapshot", snapshot],
            operation="sandbox.checkpoint_filesystem",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={
                "phase": "filesystem_checkpoint",
                "runtime": "runc",
                "dataset": dataset,
                "snapshot": snapshot,
                "mountpoint": str(self.rootfs_path_for(sandbox_id)),
            },
        )

    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        dataset = self.dataset_name_for(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        return self._run_status(
            [self._zfs_bin, "rollback", "-r", snapshot],
            operation="sandbox.restore_filesystem",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={
                "phase": "filesystem_restore",
                "runtime": "runc",
                "dataset": dataset,
                "snapshot": snapshot,
                "mountpoint": str(self.rootfs_path_for(sandbox_id)),
            },
        )

    def delete_runtime(
        self,
        sandbox_id: SandboxId,
        *,
        force: bool = True,
        ignore_missing: bool = True,
    ) -> None:
        command = [self._runtime_bin, "--root", str(self._paths.state_root), "delete"]
        if force:
            command.append("-f")
        command.append(str(sandbox_id))
        result = self._run_command(
            command,
            operation="sandbox.runtime_delete",
            sandbox_id=sandbox_id,
            check=not ignore_missing,
        )
        if result.returncode != 0 and ignore_missing:
            stderr = result.stderr.strip()
            if "does not exist" not in stderr and "container not found" not in stderr:
                raise RuntimeError(
                    f"command failed ({result.returncode}): {' '.join(command)}"
                    f"\nstdout: {result.stdout.strip()}"
                    f"\nstderr: {stderr}"
                )
        description = self._try_describe(sandbox_id)
        if description is not None:
            self._update_description(replace(description, status="stopped"))

    def destroy_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        description = self._try_describe(sandbox_id)
        dataset = None if description is None else str(description.metadata.get("zfs_dataset", "")) or None
        if not dataset:
            dataset = f"{self._paths.zfs_dataset_prefix}/{sandbox_id}"
        result = self._run_command(
            [self._zfs_bin, "destroy", "-r", dataset],
            operation="sandbox.zfs_destroy",
            sandbox_id=sandbox_id,
            check=False,
            metadata={"dataset": dataset},
        )
        if result.returncode != 0 and "does not exist" not in result.stderr:
            raise RuntimeError(
                f"command failed ({result.returncode}): {self._zfs_bin} destroy -r {dataset}"
                f"\nstdout: {result.stdout.strip()}"
                f"\nstderr: {result.stderr.strip()}"
            )

    def clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path: Path,
    ) -> str:
        source_dataset = self.dataset_name_for(source_sandbox_id)
        target_dataset = f"{self._paths.zfs_dataset_prefix}/{target_sandbox_id}"
        self._run_command(
            [self._zfs_bin, "destroy", "-r", target_dataset],
            operation="sandbox.zfs_destroy_clone_target",
            sandbox_id=target_sandbox_id,
            check=False,
            metadata={"dataset": target_dataset},
        )
        snapshot = f"{source_dataset}@{checkpoint_id}"
        self._run_command(
            [
                self._zfs_bin,
                "clone",
                "-o",
                f"mountpoint={target_rootfs_path}",
                snapshot,
                target_dataset,
            ],
            operation="sandbox.zfs_clone",
            sandbox_id=target_sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={
                "source_dataset": source_dataset,
                "target_dataset": target_dataset,
                "mountpoint": str(target_rootfs_path),
            },
        )
        self._run_command(
            [self._zfs_bin, "snapshot", f"{target_dataset}@{checkpoint_id}"],
            operation="sandbox.zfs_clone_snapshot",
            sandbox_id=target_sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={"target_dataset": target_dataset},
        )
        return target_dataset

    def bundle_path_for(self, sandbox_id: SandboxId) -> Path:
        description = self._try_describe(sandbox_id)
        if description is None:
            return self._paths.bundle_root / str(sandbox_id)
        return Path(str(description.metadata.get("bundle_path", self._paths.bundle_root / str(sandbox_id))))

    def rootfs_path_for(self, sandbox_id: SandboxId) -> Path:
        description = self._try_describe(sandbox_id)
        if description is None:
            return self.bundle_path_for(sandbox_id) / "rootfs"
        return Path(str(description.metadata.get("rootfs_path", self.bundle_path_for(sandbox_id) / "rootfs")))

    def dataset_name_for(self, sandbox_id: SandboxId) -> str:
        description = self._try_describe(sandbox_id)
        if description is None:
            return f"{self._paths.zfs_dataset_prefix}/{sandbox_id}"
        return str(description.metadata.get("zfs_dataset", f"{self._paths.zfs_dataset_prefix}/{sandbox_id}"))

    def process_checkpoint_location(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> Path:
        return self._paths.checkpoint_root / str(sandbox_id) / str(checkpoint_id) / "process"

    def process_work_path(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> Path:
        return self._paths.checkpoint_root / str(sandbox_id) / str(checkpoint_id) / "work"

    def _persist(self, description: SandboxDescription) -> None:
        path = self._metadata_path(description.sandbox_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "sandbox_id": str(description.sandbox_id),
                    "runtime_name": description.runtime_name,
                    "status": description.status,
                    "metadata": description.metadata,
                },
                sort_keys=True,
                indent=2,
            ),
            encoding="utf-8",
        )
        tmp.replace(path)
        logger.debug("Persisted sandbox %s metadata to %s", description.sandbox_id, path)

    def _register_with_host_inspector(self, description: SandboxDescription) -> None:
        if self._host_inspector_client is None:
            return
        try:
            ignore_process_rules = description.metadata.get("host_inspector_ignore_process_rules")
            self._host_inspector_client.register_sandbox(
                description.sandbox_id,
                "runc",
                str(description.sandbox_id),
                ignore_process_rules=None if ignore_process_rules is None else list(ignore_process_rules),
            )
        except Exception:
            logger.exception("Failed to register sandbox %s with host inspector", description.sandbox_id)

    def _metadata_path(self, sandbox_id: SandboxId) -> Path:
        return self._paths.metadata_root / f"{sandbox_id}.json"

    def _try_describe(self, sandbox_id: SandboxId) -> SandboxDescription | None:
        try:
            return self.describe(sandbox_id)
        except KeyError:
            return None

    def _update_description(self, description: SandboxDescription) -> None:
        with self._lock:
            self._items[description.sandbox_id] = description
        self._persist(description)

    def _run_status(
        self,
        command: list[str],
        *,
        operation: str,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> RuntimeOperationStatus:
        result = self._run_command(
            command,
            operation=operation,
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata,
        )
        merged = dict(metadata or {})
        merged["stdout"] = result.stdout.strip()
        merged["stderr"] = result.stderr.strip()
        return RuntimeOperationStatus(
            executed=True,
            reason="command_executed",
            command=result.command,
            metadata=merged,
        )

    def _run_command(
        self,
        command: list[str],
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
        checkpoint_id: CheckpointId | None = None,
        cwd: Path | None = None,
        check: bool = True,
        expected_error_substrings: tuple[str, ...] = (),
        metadata: dict[str, object] | None = None,
    ) -> CommandResult:
        logger.debug("Running sandbox manager command: %s", " ".join(command))
        started = time.perf_counter()
        result = self._runner.run(command, cwd=cwd)
        duration_ms = (time.perf_counter() - started) * 1000.0
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        success = result.returncode == 0
        self._emit_command_telemetry(
            operation,
            command=command,
            duration_ms=duration_ms,
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            success=success,
            metadata={**(metadata or {}), "cwd": None if cwd is None else str(cwd)},
            stdout=stdout,
            stderr=stderr,
        )
        if not success and check:
            log_fn = logger.error
            if any(fragment in stderr for fragment in expected_error_substrings):
                log_fn = logger.info
            log_fn(
                "Sandbox manager command failed rc=%d command=%s stdout=%s stderr=%s",
                result.returncode,
                " ".join(command),
                stdout,
                stderr,
            )
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(command)}"
                f"\nstdout: {stdout}"
                f"\nstderr: {stderr}"
            )
        logger.debug("Sandbox manager command completed: %s", " ".join(command))
        return result

    def _emit_command_telemetry(
        self,
        operation: str,
        *,
        command: list[str],
        duration_ms: float,
        sandbox_id: SandboxId | None,
        checkpoint_id: CheckpointId | None = None,
        success: bool,
        metadata: dict[str, object] | None = None,
        stdout: str,
        stderr: str,
    ) -> None:
        attributes = {
            "operation": operation,
            "command": list(command),
            "sandbox_id": "" if sandbox_id is None else str(sandbox_id),
            "checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id),
            "success": success,
            "stdout": stdout,
            "stderr": stderr,
            **(metadata or {}),
        }
        self._telemetry.emit_metric("sandbox.command_duration_ms", duration_ms, attributes)
        self._telemetry.emit_event("sandbox.command", attributes)

    def _checkpoint_optional_args(self) -> list[str]:
        args: list[str] = []
        if self._checkpoint_options.tcp_established:
            args.append("--tcp-established")
        if self._checkpoint_options.shell_job:
            args.append("--shell-job")
        if self._checkpoint_options.tcp_skip_in_flight:
            args.append("--tcp-skip-in-flight")
        args.extend(self._checkpoint_options.extra_args)
        return args

    def _restore_optional_args(self) -> list[str]:
        args: list[str] = []
        if self._restore_options.tcp_established:
            args.append("--tcp-established")
        if self._restore_options.shell_job:
            args.append("--shell-job")
        args.extend(self._restore_options.extra_args)
        return args

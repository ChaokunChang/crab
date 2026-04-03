from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock

from ..contracts import Runtime, TelemetrySink
from ..ids import CheckpointId, SandboxId
from ..models import RuntimeCapabilities, RuntimeOperationStatus, SandboxDescription, SandboxExecResult, SandboxRuntimeState
from ..remote_inspector import HostInspectorServiceClient
from ..telemetry import NoopTelemetrySink, start_operation, telemetry_capture_command_output, telemetry_is_detailed
from .base import CommandResult, CommandRunner, SubprocessCommandRunner

logger = logging.getLogger(__name__)
_HOST_INSPECTOR_REGISTER_ATTEMPTS = 3
_HOST_INSPECTOR_REGISTER_RETRY_DELAY_S = 0.2
_RESILIENT_EXEC_RECOVERY_TIMEOUT_S = 300.0
_DEFAULT_RUNTIME_COMMAND_TIMEOUT_SECONDS = 60.0
_DEFAULT_ZFS_PREPARE_TIMEOUT_SECONDS = 300.0
_DATASET_DESTROY_BUSY_RETRIES = 10
_DATASET_DESTROY_BUSY_RETRY_DELAY_S = 0.5
_LAUNCH_PREPARED_METADATA_KEY = "_agent_cr_runtime_prepared"
_LAUNCH_REUSE_EXISTING_ROOTFS_METADATA_KEY = "_agent_cr_runtime_reuse_existing_rootfs"
_SHARED_ROOTFS_KEY_METADATA_KEY = "shared_rootfs_key"
_SHARED_ROOTFS_PERSIST_METADATA_KEY = "shared_rootfs_persist"
_SHARED_ROOTFS_SNAPSHOT_NAME = "base"
_RESILIENT_EXEC_RETRYABLE_ERROR_FRAGMENTS = (
    "container does not exist",
    "container not running",
    "container not found",
    "unable to start container process",
    "failed to exec in container",
    "cannot allocate tty",
)
_POSTFIX_QUEUE_DIRS = (
    "active",
    "bounce",
    "corrupt",
    "defer",
    "deferred",
    "flush",
    "incoming",
    "private",
    "saved",
)
_POSTFIX_POSTDROP_DIRS = {
    "maildrop": 0o1730,
    "public": 0o2710,
}


def _lookup_unix_id(path: Path, name: str, *, field_index: int) -> int | None:
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    for line in content.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(":")
        if len(parts) <= field_index or parts[0] != name:
            continue
        try:
            return int(parts[field_index])
        except ValueError:
            return None
    return None


def _repair_postfix_rootfs_permissions(rootfs_path: Path) -> None:
    spool_root = rootfs_path / "var" / "spool" / "postfix"
    if not spool_root.is_dir():
        return
    postfix_uid = _lookup_unix_id(rootfs_path / "etc" / "passwd", "postfix", field_index=2)
    postfix_gid = _lookup_unix_id(rootfs_path / "etc" / "group", "postfix", field_index=2)
    postdrop_gid = _lookup_unix_id(rootfs_path / "etc" / "group", "postdrop", field_index=2)
    if postfix_uid is None or postfix_gid is None or postdrop_gid is None:
        return

    os.chown(spool_root, 0, 0)
    os.chmod(spool_root, 0o755)

    for name in _POSTFIX_QUEUE_DIRS:
        path = spool_root / name
        if not path.exists():
            continue
        os.chown(path, postfix_uid, postfix_gid)
        os.chmod(path, 0o700)

    for name, mode in _POSTFIX_POSTDROP_DIRS.items():
        path = spool_root / name
        if not path.exists():
            continue
        os.chown(path, postfix_uid, postdrop_gid)
        os.chmod(path, mode)

    pid_dir = spool_root / "pid"
    if pid_dir.exists():
        os.chown(pid_dir, 0, 0)
        os.chmod(pid_dir, 0o755)

    restart_marker = spool_root / "restart"
    if restart_marker.exists():
        os.chown(restart_marker, 0, 0)
        os.chmod(restart_marker, 0o644)

    var_lib_postfix = rootfs_path / "var" / "lib" / "postfix"
    if var_lib_postfix.exists():
        os.chown(var_lib_postfix, postfix_uid, postdrop_gid)
        os.chmod(var_lib_postfix, 0o755)


@dataclass(frozen=True)
class RuncRuntimePaths:
    state_root: Path = Path("/run/agent-cr/runc")
    bundle_root: Path = Path("/var/lib/agent-cr/bundles")
    checkpoint_root: Path = Path("/var/lib/agent-cr/checkpoints")
    metadata_root: Path = Path("/var/lib/agent-cr/sandbox-metadata")
    zfs_dataset_prefix: str = "agentcr/sandboxes"


@dataclass(frozen=True)
class RuncCheckpointOptions:
    tcp_established: bool = True
    shell_job: bool = True
    tcp_skip_in_flight: bool = True
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuncRestoreOptions:
    detach: bool = True
    tcp_established: bool = True
    shell_job: bool = True
    extra_args: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuncRuntimeOptions:
    checkpoint: RuncCheckpointOptions = field(default_factory=RuncCheckpointOptions)
    restore: RuncRestoreOptions = field(default_factory=RuncRestoreOptions)
    command_timeout_seconds: float = _DEFAULT_RUNTIME_COMMAND_TIMEOUT_SECONDS
    zfs_prepare_timeout_seconds: float = _DEFAULT_ZFS_PREPARE_TIMEOUT_SECONDS


class RuncRuntime(Runtime):
    def __init__(
        self,
        *,
        version: str | None = None,
        paths: RuncRuntimePaths | None = None,
        command_runner: CommandRunner | None = None,
        runtime_bin: str = "runc",
        zfs_bin: str = "zfs",
        host_inspector_client: HostInspectorServiceClient | None = None,
        telemetry: TelemetrySink | None = None,
        checkpoint_options: RuncCheckpointOptions | None = None,
        restore_options: RuncRestoreOptions | None = None,
        options: RuncRuntimeOptions | None = None,
    ) -> None:
        self._version = version
        resolved_options = options or RuncRuntimeOptions()
        self._paths = paths or RuncRuntimePaths()
        self._runner = command_runner or SubprocessCommandRunner(timeout_seconds=resolved_options.command_timeout_seconds)
        self._runtime_bin = runtime_bin
        self._zfs_bin = zfs_bin
        self._host_inspector_client = host_inspector_client
        self._telemetry = telemetry or NoopTelemetrySink()
        self._checkpoint_options = checkpoint_options or resolved_options.checkpoint
        self._restore_options = restore_options or resolved_options.restore
        self._zfs_prepare_timeout_seconds = float(resolved_options.zfs_prepare_timeout_seconds)
        self._lock = Lock()
        self._items: dict[SandboxId, SandboxDescription] = {}
        self._paths.metadata_root.mkdir(parents=True, exist_ok=True)
        self._paths.checkpoint_root.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "runc"

    @property
    def version(self) -> str | None:
        return self._version

    @property
    def paths(self) -> RuncRuntimePaths:
        return self._paths

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_process_checkpoint=True,
            supports_filesystem_checkpoint=True,
            supports_incremental_filesystem=True,
            supports_custom_checkpoint_dir=True,
        )

    def _resolve_launch_request(
        self,
        runtime_name: str,
        metadata: dict[str, object] | None = None,
    ) -> tuple[SandboxId, dict[str, object], Path, Path, str]:
        if runtime_name != self.name:
            raise ValueError(f"unsupported runtime for real runtime: {runtime_name}")
        sandbox_id = SandboxId(str((metadata or {}).get("sandbox_id", SandboxId.new())))
        md = dict(metadata or {})
        bundle_path = Path(str(md["bundle_path"])) if "bundle_path" in md else self._paths.bundle_root / str(sandbox_id)
        rootfs_path = bundle_path / "rootfs"
        dataset = str(md.get("zfs_dataset", f"{self._paths.zfs_dataset_prefix}/{sandbox_id}"))
        return sandbox_id, md, bundle_path, rootfs_path, dataset

    def _shared_rootfs_details(self, key: str, *, persist_across_runs: bool) -> tuple[str, Path]:
        if persist_across_runs:
            dataset = f"{self._paths.zfs_dataset_prefix}-cache-{key}"
        else:
            dataset = f"{self._paths.zfs_dataset_prefix}/_shared_rootfs_{key}"
        safe_prefix = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in self._paths.zfs_dataset_prefix
        )
        scope = "persistent" if persist_across_runs else "run"
        mountpoint = Path("/tmp/agent-cr-rootfs-cache") / safe_prefix / scope / key
        return dataset, mountpoint

    def _shared_rootfs_lock_path(self, key: str, *, persist_across_runs: bool) -> Path:
        safe_prefix = "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in self._paths.zfs_dataset_prefix
        )
        scope = "persistent" if persist_across_runs else "run"
        return Path("/tmp/agent-cr-rootfs-cache-locks") / safe_prefix / scope / f"{key}.lock"

    def _zfs_object_exists(self, name: str) -> bool:
        result = self._run_command(
            [self._zfs_bin, "list", "-H", "-o", "name", name],
            operation="sandbox.zfs_list",
            check=False,
            metadata={"name": name},
        )
        return result.returncode == 0

    def _destroy_dataset_by_name(
        self,
        dataset: str,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
    ) -> None:
        self._run_command(
            [self._zfs_bin, "destroy", "-r", dataset],
            operation=operation,
            sandbox_id=sandbox_id,
            check=False,
            metadata={"dataset": dataset},
        )

    def _prepare_launch_attributes(
        self,
        *,
        sandbox_id: SandboxId | None = None,
        metadata: dict[str, object] | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        attributes: dict[str, object] = {"launch_phase": "prepare"}
        if sandbox_id is not None:
            attributes["sandbox_id"] = str(sandbox_id)
        if metadata is not None:
            key = str(metadata.get(_SHARED_ROOTFS_KEY_METADATA_KEY, "")).strip()
            if key:
                attributes["shared_rootfs_key"] = key
                attributes["shared_rootfs_persist"] = bool(metadata.get(_SHARED_ROOTFS_PERSIST_METADATA_KEY, False))
            attributes["rootfs_init_dir_count"] = len(metadata.get("rootfs_init_dirs", []))
            attributes["rootfs_copy_path_count"] = len(metadata.get("rootfs_copy_paths", []))
        if extra:
            attributes.update(extra)
        return attributes

    def _materialize_rootfs(
        self,
        rootfs_path: Path,
        metadata: dict[str, object],
        *,
        sandbox_id: SandboxId | None = None,
    ) -> None:
        operation = start_operation(
            self._telemetry,
            "sandbox.rootfs_materialize",
            self._prepare_launch_attributes(
                sandbox_id=sandbox_id,
                metadata=metadata,
                extra={"rootfs_path": str(rootfs_path)},
            ),
        )
        try:
            rootfs_path.mkdir(parents=True, exist_ok=True)
            for rel in metadata.get("rootfs_init_dirs", []):
                (rootfs_path / str(rel)).mkdir(parents=True, exist_ok=True)
            for item in metadata.get("rootfs_copy_paths", []):
                source = Path(str(item["source"]))
                destination = rootfs_path / str(item["destination"]).lstrip("/")
                if source.is_dir():
                    shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination, follow_symlinks=True)
        except Exception:
            operation.finish(status="failed")
            raise
        operation.finish(status="succeeded")

    def _sync_clone_view_for_test_runner(self, source_rootfs_path: Path, target_rootfs_path: Path) -> None:
        if isinstance(self._runner, SubprocessCommandRunner):
            return
        if not source_rootfs_path.exists():
            return
        target_rootfs_path.mkdir(parents=True, exist_ok=True)
        for child in list(target_rootfs_path.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        for child in source_rootfs_path.iterdir():
            destination = target_rootfs_path / child.name
            if child.is_dir() and not child.is_symlink():
                shutil.copytree(child, destination, symlinks=True, dirs_exist_ok=True)
            elif child.is_symlink():
                destination.symlink_to(os.readlink(child))
            else:
                shutil.copy2(child, destination, follow_symlinks=True)

    def _ensure_shared_rootfs_base(
        self,
        key: str,
        *,
        persist_across_runs: bool,
        metadata: dict[str, object],
    ) -> tuple[str, Path]:
        dataset, mountpoint = self._shared_rootfs_details(key, persist_across_runs=persist_across_runs)
        snapshot = f"{dataset}@{_SHARED_ROOTFS_SNAPSHOT_NAME}"
        lock_path = self._shared_rootfs_lock_path(key, persist_across_runs=persist_across_runs)
        operation = start_operation(
            self._telemetry,
            "sandbox.shared_rootfs_prepare",
            self._prepare_launch_attributes(
                metadata=metadata,
                extra={
                    "dataset": dataset,
                    "mountpoint": str(mountpoint),
                },
            ),
        )
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("w", encoding="utf-8") as lock_fh:
            lock_wait_started = time.perf_counter()
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            lock_wait_ms = (time.perf_counter() - lock_wait_started) * 1000.0
            self._telemetry.emit_metric(
                "sandbox.shared_rootfs_lock_wait_ms",
                lock_wait_ms,
                self._prepare_launch_attributes(
                    metadata=metadata,
                    extra={"dataset": dataset, "mountpoint": str(mountpoint)},
                ),
            )
            try:
                dataset_exists = self._zfs_object_exists(dataset)
                snapshot_exists = self._zfs_object_exists(snapshot)
                if dataset_exists and snapshot_exists:
                    operation.finish(status="succeeded", attributes={"cache_hit": True})
                    return dataset, mountpoint
                if dataset_exists and not snapshot_exists:
                    self._destroy_dataset_by_name(
                        dataset,
                        operation="sandbox.zfs_destroy_incomplete_shared_rootfs",
                    )
                mountpoint.mkdir(parents=True, exist_ok=True)
                self._run_command(
                    [self._zfs_bin, "create", "-o", f"mountpoint={mountpoint}", dataset],
                    operation="sandbox.zfs_create_shared_rootfs",
                    metadata={"dataset": dataset, "mountpoint": str(mountpoint)},
                )
                self._materialize_rootfs(mountpoint, metadata)
                self._run_command(
                    [self._zfs_bin, "snapshot", snapshot],
                    operation="sandbox.zfs_snapshot_shared_rootfs",
                    metadata={"dataset": dataset, "snapshot": snapshot},
                )
            except Exception:
                operation.finish(status="failed")
                raise
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        operation.finish(status="succeeded", attributes={"cache_hit": False})
        return dataset, mountpoint

    def prepare_launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        sandbox_id, md, bundle_path, rootfs_path, dataset = self._resolve_launch_request(runtime_name, metadata)
        shared_rootfs_key = str(md.get(_SHARED_ROOTFS_KEY_METADATA_KEY, "")).strip()
        operation = start_operation(
            self._telemetry,
            "sandbox.runtime_prepare_launch",
            self._prepare_launch_attributes(
                sandbox_id=sandbox_id,
                metadata=md,
                extra={
                    "bundle_path": str(bundle_path),
                    "rootfs_path": str(rootfs_path),
                    "dataset": dataset,
                    "shared_rootfs_enabled": bool(shared_rootfs_key),
                },
            ),
        )
        if metadata is not None and bool(metadata.get(_LAUNCH_PREPARED_METADATA_KEY, False)):
            logger.info(
                "Skipping runtime launch preparation sandbox=%s bundle_path=%s dataset=%s reason=already_prepared",
                sandbox_id,
                bundle_path,
                dataset,
            )
            operation.finish(status="succeeded", attributes={"already_prepared": True})
            return sandbox_id

        logger.info(
            "Preparing runtime launch sandbox=%s bundle_path=%s rootfs_path=%s dataset=%s",
            sandbox_id,
            bundle_path,
            rootfs_path,
            dataset,
        )
        try:
            bundle_path.mkdir(parents=True, exist_ok=True)
            rootfs_path.mkdir(parents=True, exist_ok=True)
            reuse_existing_rootfs = bool(md.get(_LAUNCH_REUSE_EXISTING_ROOTFS_METADATA_KEY, False))
            if reuse_existing_rootfs and self._zfs_dataset_exists(dataset):
                if metadata is not None:
                    metadata["rootfs_path"] = str(rootfs_path)
                    metadata["zfs_dataset"] = dataset
                    metadata[_LAUNCH_PREPARED_METADATA_KEY] = True
                logger.info(
                    "Prepared runtime launch sandbox=%s bundle_path=%s rootfs_path=%s dataset=%s "
                    "reason=reusing_existing_rootfs",
                    sandbox_id,
                    bundle_path,
                    rootfs_path,
                    dataset,
                )
                operation.finish(status="succeeded", attributes={"already_prepared": False, "reused_existing_rootfs": True})
                return sandbox_id
            if shared_rootfs_key:
                shared_dataset, shared_rootfs_path = self._ensure_shared_rootfs_base(
                    shared_rootfs_key,
                    persist_across_runs=bool(md.get(_SHARED_ROOTFS_PERSIST_METADATA_KEY, False)),
                    metadata=md,
                )
                self._destroy_dataset_by_name(
                    dataset,
                    operation="sandbox.zfs_destroy_stale_launch_dataset",
                    sandbox_id=sandbox_id,
                )
                shared_snapshot = f"{shared_dataset}@{_SHARED_ROOTFS_SNAPSHOT_NAME}"
                self._run_command(
                    [self._zfs_bin, "clone", "-o", f"mountpoint={rootfs_path}", shared_snapshot, dataset],
                    operation="sandbox.zfs_clone_launch_rootfs",
                    sandbox_id=sandbox_id,
                    metadata={
                        "source_dataset": shared_dataset,
                        "target_dataset": dataset,
                        "snapshot": shared_snapshot,
                        "mountpoint": str(rootfs_path),
                    },
                )
                self._sync_clone_view_for_test_runner(shared_rootfs_path, rootfs_path)
            else:
                self._destroy_dataset_by_name(
                    dataset,
                    operation="sandbox.zfs_destroy_stale_launch_dataset",
                    sandbox_id=sandbox_id,
                )
                self._run_command(
                    [self._zfs_bin, "create", "-o", f"mountpoint={rootfs_path}", dataset],
                    operation="sandbox.zfs_create",
                    sandbox_id=sandbox_id,
                    metadata={"dataset": dataset, "mountpoint": str(rootfs_path)},
                    timeout_seconds=self._zfs_prepare_timeout_seconds,
                )
                self._materialize_rootfs(rootfs_path, md, sandbox_id=sandbox_id)
            _repair_postfix_rootfs_permissions(rootfs_path)
            if metadata is not None:
                metadata["rootfs_path"] = str(rootfs_path)
                metadata["zfs_dataset"] = dataset
                metadata[_LAUNCH_PREPARED_METADATA_KEY] = True
            logger.info(
                "Prepared runtime launch sandbox=%s bundle_path=%s rootfs_path=%s dataset=%s",
                sandbox_id,
                bundle_path,
                rootfs_path,
                dataset,
            )
        except Exception:
            operation.finish(status="failed", attributes={"already_prepared": False})
            raise
        operation.finish(status="succeeded", attributes={"already_prepared": False})
        return sandbox_id

    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        self.prepare_launch(runtime_name, metadata)
        sandbox_id, md, bundle_path, rootfs_path, dataset = self._resolve_launch_request(runtime_name, metadata)
        description_metadata = {
            key: value
            for key, value in md.items()
            if key not in {_LAUNCH_PREPARED_METADATA_KEY, _LAUNCH_REUSE_EXISTING_ROOTFS_METADATA_KEY}
        }
        logger.info(
            "Launching prepared runtime sandbox=%s bundle_path=%s dataset=%s",
            sandbox_id,
            bundle_path,
            dataset,
        )
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
                **description_metadata,
                "bundle_path": str(bundle_path),
                "rootfs_path": str(rootfs_path),
                "zfs_dataset": dataset,
            },
        )
        with self._lock:
            self._items[sandbox_id] = description
        self._persist(description)
        self._register_with_host_inspector(description)
        logger.info(
            "Launched runtime sandbox=%s bundle_path=%s rootfs_path=%s dataset=%s",
            sandbox_id,
            bundle_path,
            rootfs_path,
            dataset,
        )
        return sandbox_id

    def stop(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        self._run_command(
            [self._runtime_bin, "--root", str(self._paths.state_root), "kill", str(sandbox_id), "TERM"],
            operation="sandbox.runtime_kill",
            sandbox_id=sandbox_id,
        )
        self._update_description(replace(description, status="stopped"))

    def pause(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
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

    def resume(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        try:
            self._run_command(
                [self._runtime_bin, "--root", str(self._paths.state_root), "resume", str(sandbox_id)],
                operation="sandbox.runtime_resume",
                sandbox_id=sandbox_id,
                expected_error_substrings=(
                    "container not paused",
                    "container not running",
                    "container does not exist",
                ),
            )
        except RuntimeError as exc:
            if any(
                fragment in str(exc)
                for fragment in ("container not paused", "container not running", "container does not exist")
            ):
                runtime_state = self.inspect_runtime(sandbox_id)
                if runtime_state.status.lower() == "running":
                    self._update_description(replace(description, status="running"))
                    return
                if runtime_state.status.lower() in {"paused", "created"}:
                    raise
                self.sync_runtime_state(sandbox_id, is_running=False)
                return
            raise
        self._update_description(replace(description, status="running"))

    def sync_runtime_state(self, sandbox_id: SandboxId, *, is_running: bool) -> None:
        description = self.describe(sandbox_id)
        self._update_description(replace(description, status="running" if is_running else "stopped"))

    def prepare_for_restore(self, sandbox_id: SandboxId) -> None:
        self.delete_runtime(sandbox_id, force=True, ignore_missing=True)

    def mark_restored(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        updated = replace(description, status="running")
        self._update_description(updated)
        self._register_with_host_inspector(updated)

    def delete(self, sandbox_id: SandboxId) -> None:
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
            return SandboxRuntimeState(
                sandbox_id=sandbox_id,
                runtime_name=self.name,
                status="missing",
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
            runtime_name=self.name,
            status=str(payload.get("status", "unknown")),
            pid=pid if pid > 0 else None,
            bundle_path=None if description is None else str(description.metadata.get("bundle_path", "")) or None,
            metadata={"payload": payload, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()},
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
        stdout_target = subprocess.PIPE if capture_output else subprocess.DEVNULL
        stderr_target = subprocess.PIPE if capture_output else subprocess.DEVNULL
        operation_context = start_operation(
            self._telemetry,
            "sandbox.runtime_exec",
            self._command_attributes(
                operation="sandbox.runtime_exec",
                command=command,
                sandbox_id=sandbox_id,
                metadata={
                    "cwd": cwd,
                    "user": user,
                    "capture_output": capture_output,
                    "timeout_s": timeout_s,
                },
            ),
        )
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
        success = completed.returncode == 0
        operation_context.finish(
            status="succeeded" if success else "failed",
            attributes=self._command_finish_attributes(
                success=success,
                returncode=completed.returncode,
                stdout=stdout,
                stderr=stderr,
            ),
        )
        self._telemetry.emit_metric(
            "sandbox.command_duration_ms",
            duration_ms,
            self._command_metric_attributes(
                operation="sandbox.runtime_exec",
                command=command,
                sandbox_id=sandbox_id,
                checkpoint_id=None,
                success=success,
                returncode=completed.returncode,
                metadata={
                    "cwd": cwd,
                    "user": user,
                    "capture_output": capture_output,
                    "timeout_s": timeout_s,
                },
                stdout=stdout,
                stderr=stderr,
            ),
        )
        self._telemetry.emit_event(
            "sandbox.command",
            self._command_metric_attributes(
                operation="sandbox.runtime_exec",
                command=command,
                sandbox_id=sandbox_id,
                checkpoint_id=None,
                success=success,
                returncode=completed.returncode,
                metadata={
                    "cwd": cwd,
                    "user": user,
                    "capture_output": capture_output,
                    "timeout_s": timeout_s,
                },
                stdout=stdout,
                stderr=stderr,
            ),
        )
        return SandboxExecResult(args=tuple(command), returncode=int(completed.returncode), stdout=stdout, stderr=stderr)

    def resilient_exec(
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
        if not capture_output:
            raise ValueError("resilient_exec currently only supports attached capture_output=True mode")
        attempt = 0
        while True:
            attempt += 1
            result = self.exec(
                sandbox_id,
                argv,
                cwd=cwd,
                env=env,
                user=user,
                timeout_s=timeout_s,
                capture_output=capture_output,
            )
            if result.returncode == 0:
                return result
            if not self._is_retriable_resilient_exec_failure(sandbox_id, result):
                return result
            logger.warning(
                "Retrying resilient exec after sandbox interruption sandbox=%s attempt=%d command=%s",
                sandbox_id,
                attempt,
                " ".join(argv),
            )
            self._wait_for_runtime_running(sandbox_id, timeout_s=_RESILIENT_EXEC_RECOVERY_TIMEOUT_S)

    def _is_retriable_resilient_exec_failure(self, sandbox_id: SandboxId, result: SandboxExecResult) -> bool:
        stderr = result.stderr.lower()
        stdout = result.stdout.lower()
        if any(fragment in stderr or fragment in stdout for fragment in _RESILIENT_EXEC_RETRYABLE_ERROR_FRAGMENTS):
            return True
        try:
            runtime_state = self.inspect_runtime(sandbox_id)
        except Exception:
            logger.debug(
                "Treating exec failure as retriable because runtime inspection failed sandbox=%s",
                sandbox_id,
                exc_info=True,
            )
            return True
        return runtime_state.status.lower() in {"missing", "stopped"}

    def _wait_for_runtime_running(self, sandbox_id: SandboxId, *, timeout_s: float) -> None:
        deadline = time.monotonic() + max(1.0, timeout_s)
        while time.monotonic() < deadline:
            runtime_state = self.inspect_runtime(sandbox_id)
            if runtime_state.status.lower() == "running" and runtime_state.is_running:
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"timed out waiting for sandbox {sandbox_id} to recover for resilient exec after {timeout_s:.1f}s"
        )

    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ) -> RuntimeOperationStatus:
        image_path = Path(self.process_checkpoint_location(sandbox_id, checkpoint_id) or "")
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
        status = self._run_status(
            command,
            operation="sandbox.checkpoint_process",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={
                "phase": "process_checkpoint",
                "runtime": self.name,
                "image_path": str(image_path),
                "work_path": str(work_path),
                "bundle_path": str(self.bundle_path_for(sandbox_id)),
                "state_root": str(self._paths.state_root),
                "leave_running": leave_running,
            },
        )
        size_bytes, file_count = self._summarize_directory(image_path)
        return replace(
            status,
            metadata={
                **status.metadata,
                "checkpoint_scope": "process_only",
                "process_checkpoint_size_bytes": size_bytes,
                "process_checkpoint_file_count": file_count,
            },
        )

    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        image_path = Path(self.process_checkpoint_location(sandbox_id, checkpoint_id) or "")
        work_path = self.process_work_path(sandbox_id, checkpoint_id)
        image_path.mkdir(parents=True, exist_ok=True)
        work_path.mkdir(parents=True, exist_ok=True)
        command = [self._runtime_bin, "--root", str(self._paths.state_root), "restore"]
        if self._restore_options.detach:
            command.append("-d")
        command.extend(
            ["--bundle", str(self.bundle_path_for(sandbox_id)), "--image-path", str(image_path), "--work-path", str(work_path)]
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
                "runtime": self.name,
                "image_path": str(image_path),
                "work_path": str(work_path),
                "bundle_path": str(self.bundle_path_for(sandbox_id)),
                "state_root": str(self._paths.state_root),
            },
        )

    def process_checkpoint_location(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> str | None:
        return str(self._paths.checkpoint_root / str(sandbox_id) / str(checkpoint_id) / "process")

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        dataset = self.dataset_name_for(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        status = self._run_status(
            [self._zfs_bin, "snapshot", snapshot],
            operation="sandbox.checkpoint_filesystem",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=self.filesystem_checkpoint_metadata(sandbox_id, checkpoint_id),
        )
        written_bytes, used_bytes = self._query_zfs_snapshot_sizes(snapshot, sandbox_id=sandbox_id, checkpoint_id=checkpoint_id)
        return replace(
            status,
            metadata={
                **status.metadata,
                "checkpoint_scope": "filesystem_only",
                "filesystem_checkpoint_written_bytes": written_bytes,
                "filesystem_checkpoint_used_bytes": used_bytes,
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
                "runtime": self.name,
                "dataset": dataset,
                "snapshot": snapshot,
                "mountpoint": str(self.rootfs_path_for(sandbox_id)),
            },
        )

    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        dataset = self.dataset_name_for(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        return {
            "phase": "filesystem_checkpoint",
            "runtime": self.name,
            "dataset": dataset,
            "snapshot": snapshot,
            "mountpoint": str(self.rootfs_path_for(sandbox_id)),
        }

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
            if "container init still running" in stderr:
                logger.warning(
                    "Runtime delete reported a still-running init process; sending KILL and retrying delete sandbox=%s",
                    sandbox_id,
                )
                kill_command = [self._runtime_bin, "--root", str(self._paths.state_root), "kill", str(sandbox_id), "KILL"]
                kill_result = self._run_command(
                    kill_command,
                    operation="sandbox.runtime_kill_force",
                    sandbox_id=sandbox_id,
                    check=False,
                )
                kill_stderr = kill_result.stderr.strip()
                if (
                    kill_result.returncode != 0
                    and "does not exist" not in kill_stderr
                    and "container not found" not in kill_stderr
                    and "container not running" not in kill_stderr
                ):
                    raise RuntimeError(
                        f"command failed ({kill_result.returncode}): {' '.join(kill_command)}"
                        f"\nstdout: {kill_result.stdout.strip()}"
                        f"\nstderr: {kill_stderr}"
                    )
                result = self._run_command(
                    command,
                    operation="sandbox.runtime_delete_retry",
                    sandbox_id=sandbox_id,
                    check=False,
                )
                stderr = result.stderr.strip()
            if result.returncode != 0 and "does not exist" not in stderr and "container not found" not in stderr:
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
        result: CommandResult | None = None
        stderr = ""
        for attempt in range(1, _DATASET_DESTROY_BUSY_RETRIES + 1):
            result = self._run_command(
                [self._zfs_bin, "destroy", "-r", dataset],
                operation="sandbox.zfs_destroy",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"dataset": dataset, "attempt": attempt},
            )
            stderr = result.stderr.strip()
            if result.returncode == 0 or "does not exist" in stderr:
                return
            if "dataset is busy" not in stderr:
                break
            if attempt < _DATASET_DESTROY_BUSY_RETRIES:
                logger.warning(
                    "ZFS dataset destroy reported busy; retrying sandbox=%s dataset=%s attempt=%d/%d stderr=%s",
                    sandbox_id,
                    dataset,
                    attempt,
                    _DATASET_DESTROY_BUSY_RETRIES,
                    stderr,
                )
                time.sleep(_DATASET_DESTROY_BUSY_RETRY_DELAY_S)
        assert result is not None
        if result.returncode != 0 and "does not exist" not in stderr:
            raise RuntimeError(
                f"command failed ({result.returncode}): {self._zfs_bin} destroy -r {dataset}"
                f"\nstdout: {result.stdout.strip()}"
                f"\nstderr: {stderr}"
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
            [self._zfs_bin, "clone", "-o", f"mountpoint={target_rootfs_path}", snapshot, target_dataset],
            operation="sandbox.zfs_clone",
            sandbox_id=target_sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={"source_dataset": source_dataset, "target_dataset": target_dataset, "mountpoint": str(target_rootfs_path)},
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

    def _zfs_dataset_exists(self, dataset: str) -> bool:
        result = self._run_command(
            [self._zfs_bin, "list", "-H", "-o", "name", dataset],
            operation="sandbox.zfs_exists",
            check=False,
            metadata={"dataset": dataset},
        )
        return result.returncode == 0

    def process_work_path(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> Path:
        return self._paths.checkpoint_root / str(sandbox_id) / str(checkpoint_id) / "work"

    def _persist(self, description: SandboxDescription) -> None:
        path = self._metadata_path(description.sandbox_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
        try:
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
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _register_with_host_inspector(self, description: SandboxDescription) -> None:
        if self._host_inspector_client is None:
            return
        ignore_process_rules = description.metadata.get("host_inspector_ignore_process_rules")
        for attempt in range(1, _HOST_INSPECTOR_REGISTER_ATTEMPTS + 1):
            try:
                self._host_inspector_client.register_sandbox(
                    description.sandbox_id,
                    self.name,
                    str(description.sandbox_id),
                    ignore_process_rules=None if ignore_process_rules is None else list(ignore_process_rules),
                )
                return
            except Exception as exc:
                if attempt >= _HOST_INSPECTOR_REGISTER_ATTEMPTS:
                    logger.exception("Failed to register sandbox %s with host inspector", description.sandbox_id)
                    return
                logger.warning(
                    "Host inspector registration attempt %d/%d failed for sandbox %s; retrying: %s",
                    attempt,
                    _HOST_INSPECTOR_REGISTER_ATTEMPTS,
                    description.sandbox_id,
                    exc,
                )
                time.sleep(_HOST_INSPECTOR_REGISTER_RETRY_DELAY_S)

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
        return RuntimeOperationStatus(executed=True, reason="command_executed", command=result.command, metadata=merged)

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
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        operation_context = start_operation(
            self._telemetry,
            operation,
            self._command_attributes(
                operation=operation,
                command=command,
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                metadata={**(metadata or {}), "cwd": None if cwd is None else str(cwd)},
            ),
        )
        result = self._runner.run(command, cwd=cwd, timeout_seconds=timeout_seconds)
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        success = result.returncode == 0
        duration_ms = operation_context.finish(
            status="succeeded" if success else "failed",
            attributes=self._command_finish_attributes(
                success=success,
                returncode=result.returncode,
                stdout=stdout,
                stderr=stderr,
            ),
        )
        self._telemetry.emit_metric(
            "sandbox.command_duration_ms",
            duration_ms,
            self._command_metric_attributes(
                operation=operation,
                command=command,
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                success=success,
                returncode=result.returncode,
                metadata={**(metadata or {}), "cwd": None if cwd is None else str(cwd)},
                stdout=stdout,
                stderr=stderr,
            ),
        )
        self._telemetry.emit_event(
            "sandbox.command",
            self._command_metric_attributes(
                operation=operation,
                command=command,
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                success=success,
                returncode=result.returncode,
                metadata={**(metadata or {}), "cwd": None if cwd is None else str(cwd)},
                stdout=stdout,
                stderr=stderr,
            ),
        )
        if not success and check:
            expected_error = any(fragment in stderr for fragment in expected_error_substrings)
            log_fn = logger.info if expected_error else logger.error
            message = (
                "Runtime command returned expected non-zero rc=%d command=%s stdout=%s stderr=%s"
                if expected_error
                else "Runtime command failed rc=%d command=%s stdout=%s stderr=%s"
            )
            log_fn(
                message,
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
        return result

    def _command_attributes(
        self,
        *,
        operation: str,
        command: list[str],
        sandbox_id: SandboxId | None,
        checkpoint_id: CheckpointId | None = None,
        metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        attributes = {
            "component": "runtime",
            "operation": operation,
            "command": list(command),
            "sandbox_id": "" if sandbox_id is None else str(sandbox_id),
            "checkpoint_id": "" if checkpoint_id is None else str(checkpoint_id),
            **(metadata or {}),
        }
        return attributes

    def _command_finish_attributes(
        self,
        *,
        success: bool,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> dict[str, object]:
        attributes: dict[str, object] = {
            "success": success,
            "returncode": int(returncode),
        }
        if telemetry_capture_command_output(self._telemetry) or telemetry_is_detailed(self._telemetry):
            attributes["stdout"] = stdout
            attributes["stderr"] = stderr
        return attributes

    def _command_metric_attributes(
        self,
        *,
        operation: str,
        command: list[str],
        sandbox_id: SandboxId | None,
        checkpoint_id: CheckpointId | None,
        success: bool,
        returncode: int,
        metadata: dict[str, object] | None,
        stdout: str,
        stderr: str,
    ) -> dict[str, object]:
        attributes = self._command_attributes(
            operation=operation,
            command=command,
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata,
        )
        attributes.update(self._command_finish_attributes(success=success, returncode=returncode, stdout=stdout, stderr=stderr))
        return attributes

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

    def _summarize_directory(self, path: Path) -> tuple[int, int]:
        size_bytes = 0
        file_count = 0
        try:
            for root, _, files in os.walk(path):
                for name in files:
                    file_path = Path(root) / name
                    try:
                        size_bytes += file_path.stat().st_size
                        file_count += 1
                    except OSError:
                        continue
        except OSError:
            return 0, 0
        return size_bytes, file_count

    def _query_zfs_snapshot_sizes(
        self,
        snapshot: str,
        *,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> tuple[int | None, int | None]:
        result = self._run_command(
            [self._zfs_bin, "get", "-Hp", "-o", "property,value", "written,used", snapshot],
            operation="sandbox.zfs_snapshot_stats",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            check=False,
            metadata={"snapshot": snapshot},
        )
        if result.returncode != 0:
            return None, None
        values: dict[str, int] = {}
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            property_name = parts[0].strip()
            raw_value = parts[1].strip()
            try:
                values[property_name] = int(raw_value)
            except ValueError:
                continue
        return values.get("written"), values.get("used")

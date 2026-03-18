from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..ids import CheckpointId, SandboxId
from ..sandbox_manager import RuncSandboxManager, RuncSandboxManagerPaths
from .base import CommandRunner, CommandRuntimeAdapter


@dataclass(frozen=True)
class RuncRuntimePaths:
    state_root: Path = Path("/run/agent-cr/runc")
    bundle_root: Path = Path("/var/lib/agent-cr/bundles")
    checkpoint_root: Path = Path("/var/lib/agent-cr/checkpoints")
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


class RuncRuntimeAdapter(CommandRuntimeAdapter):
    def __init__(
        self,
        version: str | None = None,
        *,
        sandbox_manager: RuncSandboxManager | None = None,
        paths: RuncRuntimePaths | None = None,
        options: RuncRuntimeOptions | None = None,
        command_runner: CommandRunner | None = None,
        runtime_bin: str = "runc",
        zfs_bin: str = "zfs",
    ):
        super().__init__(name="runc", version=version, command_runner=command_runner)
        self._options = options or RuncRuntimeOptions()
        self._runtime_bin = runtime_bin
        self._zfs_bin = zfs_bin
        self._sandbox_manager = sandbox_manager
        if self._sandbox_manager is None:
            resolved_paths = paths or RuncRuntimePaths()
            self._sandbox_manager = RuncSandboxManager(
                paths=RuncSandboxManagerPaths(
                    state_root=resolved_paths.state_root,
                    bundle_root=resolved_paths.bundle_root,
                    checkpoint_root=resolved_paths.checkpoint_root,
                    zfs_dataset_prefix=resolved_paths.zfs_dataset_prefix,
                ),
                command_runner=command_runner,
                runtime_bin=runtime_bin,
                zfs_bin=zfs_bin,
                checkpoint_options=self._options.checkpoint,
                restore_options=self._options.restore,
            )
            self._paths = resolved_paths
        else:
            self._paths = paths or RuncRuntimePaths(
                state_root=self._sandbox_manager.paths.state_root,
                bundle_root=self._sandbox_manager.paths.bundle_root,
                checkpoint_root=self._sandbox_manager.paths.checkpoint_root,
                zfs_dataset_prefix=self._sandbox_manager.paths.zfs_dataset_prefix,
            )

    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ):
        meta = self._process_metadata(
            sandbox_id,
            checkpoint_id,
            phase="process_checkpoint",
            leave_running=leave_running,
        )
        _ = meta
        return self._sandbox_manager.checkpoint_process(
            sandbox_id,
            checkpoint_id,
            leave_running=leave_running,
        )

    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ):
        meta = self._process_metadata(sandbox_id, checkpoint_id, phase="process_restore")
        _ = meta
        return self._sandbox_manager.restore_process(sandbox_id, checkpoint_id)

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ):
        return self._sandbox_manager.checkpoint_filesystem(sandbox_id, checkpoint_id)

    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ):
        return self._sandbox_manager.restore_filesystem(sandbox_id, checkpoint_id)

    def _process_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        phase: str,
        leave_running: bool | None = None,
    ) -> dict[str, object]:
        image_path = self._checkpoint_image_path(sandbox_id, checkpoint_id)
        work_path = self._checkpoint_work_path(sandbox_id, checkpoint_id)
        metadata = {
            "phase": phase,
            "runtime": self.name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "bundle_path": str(self._bundle_path(sandbox_id)),
            "image_path": str(image_path),
            "work_path": str(work_path),
            "state_root": str(self._paths.state_root),
        }
        if leave_running is not None:
            metadata["leave_running"] = leave_running
        return metadata

    def _filesystem_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        phase: str,
    ) -> dict[str, object]:
        dataset = self._dataset_name(sandbox_id)
        snapshot = self._snapshot_name(sandbox_id, checkpoint_id)
        return {
            "phase": phase,
            "runtime": self.name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "dataset": dataset,
            "snapshot": snapshot,
            "mountpoint": str(self._bundle_path(sandbox_id) / "rootfs"),
        }

    def _checkpoint_cmd(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ) -> list[str]:
        _ = (sandbox_id, checkpoint_id, leave_running)
        raise NotImplementedError("RuncRuntimeAdapter delegates process checkpoint execution to RuncSandboxManager")

    def _restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        _ = (sandbox_id, checkpoint_id)
        raise NotImplementedError("RuncRuntimeAdapter delegates process restore execution to RuncSandboxManager")

    def _filesystem_checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        _ = (sandbox_id, checkpoint_id)
        raise NotImplementedError("RuncRuntimeAdapter delegates filesystem checkpoint execution to RuncSandboxManager")

    def _filesystem_restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        _ = (sandbox_id, checkpoint_id)
        raise NotImplementedError("RuncRuntimeAdapter delegates filesystem restore execution to RuncSandboxManager")

    def _bundle_path(self, sandbox_id: SandboxId) -> Path:
        return self._paths.bundle_root / str(sandbox_id)

    def _checkpoint_image_path(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> Path:
        return self._paths.checkpoint_root / str(sandbox_id) / str(checkpoint_id) / "process"

    def _checkpoint_work_path(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> Path:
        return self._paths.checkpoint_root / str(sandbox_id) / str(checkpoint_id) / "work"

    def _dataset_name(self, sandbox_id: SandboxId) -> str:
        return f"{self._paths.zfs_dataset_prefix}/{sandbox_id}"

    def _snapshot_name(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> str:
        return f"{self._dataset_name(sandbox_id)}@{checkpoint_id}"

    def _ensure_dir(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _optional_args(options: RuncCheckpointOptions) -> list[str]:
        args: list[str] = []
        if options.tcp_established:
            args.append("--tcp-established")
        if options.shell_job:
            args.append("--shell-job")
        if options.tcp_skip_in_flight:
            args.append("--tcp-skip-in-flight")
        args.extend(options.extra_args)
        return args

    @staticmethod
    def _restore_optional_args(options: RuncRestoreOptions) -> list[str]:
        args: list[str] = []
        if options.tcp_established:
            args.append("--tcp-established")
        if options.shell_job:
            args.append("--shell-job")
        args.extend(options.extra_args)
        return args

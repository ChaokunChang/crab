from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..ids import CheckpointId, SandboxId
from .base import CommandRunner, CommandRuntimeAdapter


@dataclass(frozen=True)
class RuncRuntimePaths:
    state_root: Path = Path("/run/agent-cr/runc")
    bundle_root: Path = Path("/var/lib/agent-cr/bundles")
    checkpoint_root: Path = Path("/var/lib/agent-cr/checkpoints")
    zfs_dataset_prefix: str = "agentcr/sandboxes"


class RuncRuntimeAdapter(CommandRuntimeAdapter):
    def __init__(
        self,
        version: str | None = None,
        *,
        paths: RuncRuntimePaths | None = None,
        command_runner: CommandRunner | None = None,
        runtime_bin: str = "runc",
        zfs_bin: str = "zfs",
    ):
        super().__init__(name="runc", version=version, command_runner=command_runner)
        self._paths = paths or RuncRuntimePaths()
        self._runtime_bin = runtime_bin
        self._zfs_bin = zfs_bin

    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ):
        meta = self._process_metadata(sandbox_id, checkpoint_id, phase="process_checkpoint")
        self._ensure_dir(Path(str(meta["image_path"])))
        self._ensure_dir(Path(str(meta["work_path"])))
        return super().checkpoint_process(sandbox_id, checkpoint_id)

    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ):
        meta = self._process_metadata(sandbox_id, checkpoint_id, phase="process_restore")
        self._ensure_dir(Path(str(meta["image_path"])))
        self._ensure_dir(Path(str(meta["work_path"])))
        return super().restore_process(sandbox_id, checkpoint_id)

    def _process_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        phase: str,
    ) -> dict[str, object]:
        image_path = self._checkpoint_image_path(sandbox_id, checkpoint_id)
        work_path = self._checkpoint_work_path(sandbox_id, checkpoint_id)
        return {
            "phase": phase,
            "runtime": self.name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "bundle_path": str(self._bundle_path(sandbox_id)),
            "image_path": str(image_path),
            "work_path": str(work_path),
            "state_root": str(self._paths.state_root),
        }

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

    def _checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        image_path = self._checkpoint_image_path(sandbox_id, checkpoint_id)
        work_path = self._checkpoint_work_path(sandbox_id, checkpoint_id)
        return [
            self._runtime_bin,
            "--root",
            str(self._paths.state_root),
            "checkpoint",
            str(sandbox_id),
            "--image-path",
            str(image_path),
            "--work-path",
            str(work_path),
            "--leave-running=false",
            "--tcp-established",
        ]

    def _restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        image_path = self._checkpoint_image_path(sandbox_id, checkpoint_id)
        work_path = self._checkpoint_work_path(sandbox_id, checkpoint_id)
        bundle_path = self._bundle_path(sandbox_id)
        return [
            self._runtime_bin,
            "--root",
            str(self._paths.state_root),
            "restore",
            "-d",
            "--bundle",
            str(bundle_path),
            "--image-path",
            str(image_path),
            "--work-path",
            str(work_path),
            str(sandbox_id),
        ]

    def _filesystem_checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            self._zfs_bin,
            "snapshot",
            self._snapshot_name(sandbox_id, checkpoint_id),
        ]

    def _filesystem_restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            self._zfs_bin,
            "rollback",
            "-r",
            self._snapshot_name(sandbox_id, checkpoint_id),
        ]

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

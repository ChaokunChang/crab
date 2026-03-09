from __future__ import annotations

from ..ids import CheckpointId, SandboxId
from .base import DryRunRuntimeAdapter


class RuncRuntimeAdapter(DryRunRuntimeAdapter):
    def __init__(self, version: str | None = None):
        super().__init__(name="runc", version=version, runtime_bin="runc")

    def _checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            "runc",
            "checkpoint",
            str(sandbox_id),
            "--image-path",
            f"/tmp/{checkpoint_id}",
            "--work-path",
            f"/tmp/{checkpoint_id}-work",
        ]

    def _restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            "runc",
            "restore",
            "-d",
            "--image-path",
            f"/tmp/{checkpoint_id}",
            "--work-path",
            f"/tmp/{checkpoint_id}-work",
            str(sandbox_id),
        ]

    def _filesystem_checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            "tar",
            "-C",
            f"/sandboxes/{sandbox_id}",
            "-cf",
            f"/tmp/{checkpoint_id}-rootfs.tar",
            ".",
        ]

    def _filesystem_restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            "tar",
            "-C",
            f"/sandboxes/{sandbox_id}",
            "-xf",
            f"/tmp/{checkpoint_id}-rootfs.tar",
        ]

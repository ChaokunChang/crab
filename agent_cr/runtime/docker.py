from __future__ import annotations

from ..ids import CheckpointId, SandboxId
from .base import DryRunRuntimeAdapter


class DockerRuntimeAdapter(DryRunRuntimeAdapter):
    def __init__(self, version: str | None = None):
        super().__init__(name="docker", version=version, runtime_bin="docker")

    def _checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            "docker",
            "checkpoint",
            "create",
            "--leave-running=false",
            str(sandbox_id),
            str(checkpoint_id),
        ]

    def _restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            "docker",
            "start",
            "--checkpoint",
            str(checkpoint_id),
            str(sandbox_id),
        ]

    def _filesystem_checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            "docker",
            "export",
            str(sandbox_id),
            "#=>",
            f"{checkpoint_id}.tar",
        ]

    def _filesystem_restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        return [
            "docker",
            "import",
            f"{checkpoint_id}.tar",
            str(sandbox_id),
        ]

from __future__ import annotations

from ..ids import CheckpointId, SandboxId
from ..models import RuntimeCapabilities, RuntimeOperationStatus
from .base import CommandRuntimeAdapter


class DockerRuntimeAdapter(CommandRuntimeAdapter):
    def __init__(self, version: str | None = None):
        super().__init__(name="docker", version=version)

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_process_checkpoint=True,
            supports_filesystem_checkpoint=True,
            supports_incremental_filesystem=False,
            supports_custom_checkpoint_dir=False,
        )

    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ) -> RuntimeOperationStatus:
        return RuntimeOperationStatus(
            executed=False,
            reason="docker_runtime_not_implemented",
            command=tuple(self._checkpoint_cmd(sandbox_id, checkpoint_id, leave_running=leave_running)),
            metadata=self._process_metadata(
                sandbox_id,
                checkpoint_id,
                phase="process_checkpoint",
                leave_running=leave_running,
            ),
        )

    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        return RuntimeOperationStatus(
            executed=False,
            reason="docker_runtime_not_implemented",
            command=tuple(self._restore_cmd(sandbox_id, checkpoint_id)),
            metadata=self._process_metadata(sandbox_id, checkpoint_id, phase="process_restore"),
        )

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        return RuntimeOperationStatus(
            executed=False,
            reason="docker_runtime_not_implemented",
            command=tuple(self._filesystem_checkpoint_cmd(sandbox_id, checkpoint_id)),
            metadata=self._filesystem_metadata(sandbox_id, checkpoint_id, phase="filesystem_checkpoint"),
        )

    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        return RuntimeOperationStatus(
            executed=False,
            reason="docker_runtime_not_implemented",
            command=tuple(self._filesystem_restore_cmd(sandbox_id, checkpoint_id)),
            metadata=self._filesystem_metadata(sandbox_id, checkpoint_id, phase="filesystem_restore"),
        )

    def _process_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        phase: str,
        leave_running: bool | None = None,
    ) -> dict[str, object]:
        metadata = {
            "phase": phase,
            "runtime": self.name,
            "sandbox_id": str(sandbox_id),
            "checkpoint_id": str(checkpoint_id),
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
        return self._process_metadata(sandbox_id, checkpoint_id, phase=phase)

    def _checkpoint_cmd(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ) -> list[str]:
        return [
            "docker",
            "checkpoint",
            "create",
            f"--leave-running={'true' if leave_running else 'false'}",
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

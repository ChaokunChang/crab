from __future__ import annotations

from ..contracts import SandboxRuntimeAdapter
from ..ids import CheckpointId, SandboxId
from ..models import DryRunStatus, RuntimeCapabilities


class DryRunRuntimeAdapter(SandboxRuntimeAdapter):
    def __init__(self, name: str, version: str | None, runtime_bin: str):
        self._name = name
        self._version = version
        self._runtime_bin = runtime_bin

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str | None:
        return self._version

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_process_checkpoint=True,
            supports_filesystem_checkpoint=True,
            supports_incremental_filesystem=False,
            supports_custom_checkpoint_dir=True,
        )

    def plan_process_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> DryRunStatus:
        return DryRunStatus(
            executed=False,
            reason="dry_run_runtime_adapter",
            planned_command=tuple(
                self._checkpoint_cmd(sandbox_id=sandbox_id, checkpoint_id=checkpoint_id)
            ),
            metadata={"phase": "process_checkpoint", "runtime": self.name},
        )

    def plan_process_restore(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> DryRunStatus:
        return DryRunStatus(
            executed=False,
            reason="dry_run_runtime_adapter",
            planned_command=tuple(
                self._restore_cmd(sandbox_id=sandbox_id, checkpoint_id=checkpoint_id)
            ),
            metadata={"phase": "process_restore", "runtime": self.name},
        )

    def plan_filesystem_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> DryRunStatus:
        return DryRunStatus(
            executed=False,
            reason="dry_run_runtime_adapter",
            planned_command=tuple(
                self._filesystem_checkpoint_cmd(
                    sandbox_id=sandbox_id,
                    checkpoint_id=checkpoint_id,
                )
            ),
            metadata={"phase": "filesystem_checkpoint", "runtime": self.name},
        )

    def plan_filesystem_restore(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> DryRunStatus:
        return DryRunStatus(
            executed=False,
            reason="dry_run_runtime_adapter",
            planned_command=tuple(
                self._filesystem_restore_cmd(sandbox_id=sandbox_id, checkpoint_id=checkpoint_id)
            ),
            metadata={"phase": "filesystem_restore", "runtime": self.name},
        )

    def _checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        raise NotImplementedError

    def _restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        raise NotImplementedError

    def _filesystem_checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        raise NotImplementedError

    def _filesystem_restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        raise NotImplementedError

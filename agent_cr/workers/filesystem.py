from __future__ import annotations

import json

from ..contracts import FileSystemCWorker, FileSystemRWorker, SandboxRuntimeAdapter
from ..ids import CheckpointId
from ..models import ArtifactKind, ArtifactPayload, CheckpointJob, CheckpointManifest, RestoreJob, WorkerStepResult


class AdapterFileSystemCWorker(FileSystemCWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        status = self._adapter.checkpoint_filesystem(job.sandbox_id, checkpoint_id)
        metadata = self._adapter.filesystem_checkpoint_metadata(job.sandbox_id, checkpoint_id)
        payload = {
            "sandbox_id": str(job.sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "status": {
                "executed": status.executed,
                "reason": status.reason,
                "command": list(status.command),
                "metadata": status.metadata,
            },
            "filesystem": metadata,
        }
        artifact = ArtifactPayload(
            kind=ArtifactKind.FILESYSTEM,
            name="filesystem_checkpoint.json",
            data=json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"),
            metadata={"adapter": self._adapter.name},
        )
        return WorkerStepResult(success=True, artifacts=[artifact], operation_status=status)


class AdapterFileSystemRWorker(FileSystemRWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        _ = manifest
        status = self._adapter.restore_filesystem(job.sandbox_id, job.checkpoint_id)
        return WorkerStepResult(success=True, artifacts=[], operation_status=status)

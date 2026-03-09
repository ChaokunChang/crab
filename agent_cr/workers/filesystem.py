from __future__ import annotations

import json

from ..contracts import FileSystemCWorker, FileSystemRWorker, SandboxRuntimeAdapter
from ..ids import CheckpointId
from ..models import ArtifactKind, ArtifactPayload, CheckpointJob, CheckpointManifest, RestoreJob, WorkerStepResult


class AdapterFileSystemCWorker(FileSystemCWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        dry = self._adapter.plan_filesystem_checkpoint(job.sandbox_id, checkpoint_id)
        payload = {
            "sandbox_id": str(job.sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "dry_run": {
                "executed": dry.executed,
                "reason": dry.reason,
                "planned_command": list(dry.planned_command),
                "metadata": dry.metadata,
            },
        }
        artifact = ArtifactPayload(
            kind=ArtifactKind.FILESYSTEM,
            name="filesystem_plan.json",
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            metadata={"adapter": self._adapter.name},
        )
        return WorkerStepResult(success=True, artifacts=[artifact], dry_run_status=dry)


class AdapterFileSystemRWorker(FileSystemRWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        _ = manifest
        dry = self._adapter.plan_filesystem_restore(job.sandbox_id, job.checkpoint_id)
        return WorkerStepResult(success=True, artifacts=[], dry_run_status=dry)

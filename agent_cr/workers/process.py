from __future__ import annotations

import json

from ..contracts import ProcessCWorker, ProcessRWorker, SandboxRuntimeAdapter
from ..ids import CheckpointId
from ..models import ArtifactKind, ArtifactPayload, CheckpointJob, CheckpointManifest, RestoreJob, WorkerStepResult


class AdapterProcessCWorker(ProcessCWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        dry = self._adapter.plan_process_checkpoint(job.sandbox_id, checkpoint_id)
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
            kind=ArtifactKind.PROCESS,
            name="process_plan.json",
            data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            metadata={"adapter": self._adapter.name},
        )
        return WorkerStepResult(success=True, artifacts=[artifact], dry_run_status=dry)


class AdapterProcessRWorker(ProcessRWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        _ = manifest
        dry = self._adapter.plan_process_restore(job.sandbox_id, job.checkpoint_id)
        return WorkerStepResult(success=True, artifacts=[], dry_run_status=dry)

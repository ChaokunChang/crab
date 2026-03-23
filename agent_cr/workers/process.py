from __future__ import annotations

import json
import logging
from pathlib import Path

from ..contracts import ProcessCWorker, ProcessRWorker, Runtime
from ..ids import CheckpointId
from ..models import ArtifactKind, ArtifactPayload, CheckpointJob, CheckpointManifest, RestoreJob, WorkerStepResult

logger = logging.getLogger(__name__)


def _metadata_artifact(name: str, payload: dict[str, object], *, runtime_name: str) -> ArtifactPayload:
    return ArtifactPayload(
        kind=ArtifactKind.PROCESS,
        name=name,
        data=json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"),
        metadata={"runtime": runtime_name},
    )


class AdapterProcessCWorker(ProcessCWorker):
    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        logger.debug(
            "Running process checkpoint worker for sandbox=%s checkpoint=%s runtime=%s",
            job.sandbox_id,
            checkpoint_id,
            self._runtime.name,
        )
        status = self._runtime.checkpoint_process(
            job.sandbox_id,
            checkpoint_id,
            leave_running=job.leave_running,
        )
        image_path = self._runtime.process_checkpoint_location(job.sandbox_id, checkpoint_id)
        payload = {
            "sandbox_id": str(job.sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "leave_running": job.leave_running,
            "process_storage_mode": "runtime_reference" if image_path else "adapter_default",
            "process_checkpoint_location": image_path,
            "status": {
                "executed": status.executed,
                "reason": status.reason,
                "command": list(status.command),
                "metadata": status.metadata,
            },
        }
        artifacts = [_metadata_artifact("process_checkpoint.json", payload, runtime_name=self._runtime.name)]
        logger.debug(
            "Process checkpoint worker finished for sandbox=%s checkpoint=%s executed=%s artifacts=%d",
            job.sandbox_id,
            checkpoint_id,
            status.executed,
            len(artifacts),
        )
        return WorkerStepResult(success=True, artifacts=artifacts, operation_status=status)


class AdapterProcessRWorker(ProcessRWorker):
    def __init__(self, runtime: Runtime):
        self._runtime = runtime

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        restore_checkpoint_id = _restore_checkpoint_id(
            manifest,
            metadata_key="process_restore_checkpoint_id",
            default=job.checkpoint_id,
        )
        if self._runtime.capabilities().supports_custom_checkpoint_dir:
            image_path = self._runtime.process_checkpoint_location(job.sandbox_id, restore_checkpoint_id)
            if image_path is None:
                raise ValueError("runtime did not provide process checkpoint location for restore")
            checkpoint_dir = Path(str(image_path))
            if not checkpoint_dir.exists():
                raise FileNotFoundError(f"process checkpoint directory not found: {checkpoint_dir}")
        status = self._runtime.restore_process(job.sandbox_id, restore_checkpoint_id)
        logger.debug(
            "Process restore worker finished for sandbox=%s checkpoint=%s restore_checkpoint=%s executed=%s",
            job.sandbox_id,
            job.checkpoint_id,
            restore_checkpoint_id,
            status.executed,
        )
        return WorkerStepResult(success=True, artifacts=[], operation_status=status)


def _restore_checkpoint_id(
    manifest: CheckpointManifest,
    *,
    metadata_key: str,
    default: CheckpointId,
) -> CheckpointId:
    raw = manifest.metadata.get(metadata_key)
    if raw is None:
        return default
    return CheckpointId(str(raw))

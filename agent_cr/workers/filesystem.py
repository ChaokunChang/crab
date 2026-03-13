from __future__ import annotations

import json
import logging

from ..contracts import FileSystemCWorker, FileSystemRWorker, SandboxRuntimeAdapter
from ..ids import CheckpointId
from ..models import ArtifactKind, ArtifactPayload, CheckpointJob, CheckpointManifest, RestoreJob, WorkerStepResult

logger = logging.getLogger(__name__)


class AdapterFileSystemCWorker(FileSystemCWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        logger.debug(
            "Running filesystem checkpoint worker for sandbox=%s checkpoint=%s adapter=%s",
            job.sandbox_id,
            checkpoint_id,
            self._adapter.name,
        )
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
        logger.debug(
            "Filesystem checkpoint worker finished for sandbox=%s checkpoint=%s executed=%s",
            job.sandbox_id,
            checkpoint_id,
            status.executed,
        )
        return WorkerStepResult(success=True, artifacts=[artifact], operation_status=status)


class AdapterFileSystemRWorker(FileSystemRWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        restore_checkpoint_id = _restore_checkpoint_id(
            manifest,
            metadata_key="filesystem_restore_checkpoint_id",
            default=job.checkpoint_id,
        )
        status = self._adapter.restore_filesystem(job.sandbox_id, restore_checkpoint_id)
        logger.debug(
            "Filesystem restore worker finished for sandbox=%s checkpoint=%s restore_checkpoint=%s executed=%s",
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

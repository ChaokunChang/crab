from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

from ..contracts import ProcessCWorker, ProcessRWorker, SandboxRuntimeAdapter
from ..ids import CheckpointId
from ..models import ArtifactKind, ArtifactPayload, CheckpointJob, CheckpointManifest, RestoreJob, WorkerStepResult


def _metadata_artifact(name: str, payload: dict[str, object], *, adapter_name: str) -> ArtifactPayload:
    return ArtifactPayload(
        kind=ArtifactKind.PROCESS,
        name=name,
        data=json.dumps(payload, sort_keys=True, indent=2).encode("utf-8"),
        metadata={"adapter": adapter_name},
    )


def _tar_directory(source: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(source, arcname=".")
    return buf.getvalue()


def _extract_tarball(payload: bytes, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tf:
        try:
            tf.extractall(destination, filter="data")
        except TypeError:
            tf.extractall(destination)


class AdapterProcessCWorker(ProcessCWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        status = self._adapter.checkpoint_process(job.sandbox_id, checkpoint_id)
        payload = {
            "sandbox_id": str(job.sandbox_id),
            "checkpoint_id": str(checkpoint_id),
            "status": {
                "executed": status.executed,
                "reason": status.reason,
                "command": list(status.command),
                "metadata": status.metadata,
            },
        }
        artifacts = [_metadata_artifact("process_checkpoint.json", payload, adapter_name=self._adapter.name)]
        image_path = self._adapter.process_checkpoint_location(job.sandbox_id, checkpoint_id)
        if status.executed and image_path:
            artifacts.append(
                ArtifactPayload(
                    kind=ArtifactKind.PROCESS,
                    name="process_state.tar.gz",
                    data=_tar_directory(Path(str(image_path))),
                    metadata={"adapter": self._adapter.name, "source_path": image_path},
                )
            )
        return WorkerStepResult(success=True, artifacts=artifacts, operation_status=status)


class AdapterProcessRWorker(ProcessRWorker):
    def __init__(self, adapter: SandboxRuntimeAdapter):
        self._adapter = adapter

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        process_state_ref = next(
            (x for x in manifest.process_artifacts if x.name == "process_state.tar.gz"),
            None,
        )
        if process_state_ref is not None:
            payload = job.metadata.get("process_state_payload")
            if not isinstance(payload, (bytes, bytearray)):
                raise ValueError("restore job missing process_state_payload bytes")
            image_path = self._adapter.process_checkpoint_location(job.sandbox_id, job.checkpoint_id)
            if image_path is None:
                raise ValueError("runtime adapter did not provide image_path for restore")
            _extract_tarball(bytes(payload), Path(str(image_path)))
        status = self._adapter.restore_process(job.sandbox_id, job.checkpoint_id)
        return WorkerStepResult(success=True, artifacts=[], operation_status=status)

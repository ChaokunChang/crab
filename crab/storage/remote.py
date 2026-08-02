from __future__ import annotations

from ..ids import CheckpointId, SandboxId
from ..models import ArtifactReference, CheckpointManifest
from ..contracts import RemoteCheckpointBackend


class RemoteCheckpointBackendStub(RemoteCheckpointBackend):
    """Stub contract implementation for future remote checkpoint storage backends."""

    def upload_manifest(self, manifest: CheckpointManifest) -> None:
        raise NotImplementedError("remote backend is not implemented in v1")

    def download_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        raise NotImplementedError("remote backend is not implemented in v1")

    def upload_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
        data: bytes,
    ) -> None:
        raise NotImplementedError("remote backend is not implemented in v1")

    def download_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bytes:
        raise NotImplementedError("remote backend is not implemented in v1")

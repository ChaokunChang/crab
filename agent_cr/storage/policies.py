from __future__ import annotations

from ..contracts import CheckpointManager
from ..ids import CheckpointId, SandboxId
from ..models import ArtifactPayload, ArtifactReference, CheckpointManifest


class DelegatingCheckpointManager(CheckpointManager):
    def __init__(self, delegate: CheckpointManager):
        self._delegate = delegate

    def put_manifest(self, manifest: CheckpointManifest) -> None:
        self._delegate.put_manifest(manifest)

    def get_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        return self._delegate.get_manifest(sandbox_id, checkpoint_id)

    def put_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        artifact: ArtifactPayload,
    ) -> ArtifactReference:
        return self._delegate.put_artifact(sandbox_id, checkpoint_id, artifact)

    def get_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bytes:
        return self._delegate.get_artifact(sandbox_id, checkpoint_id, reference)

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        return self._delegate.list_checkpoints(sandbox_id)

    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        self._delegate.delete_checkpoint(sandbox_id, checkpoint_id)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        self._delegate.delete_all_checkpoints(sandbox_id)

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        self._delegate.handle_checkpoint_complete(manifest)

    def handle_restore_complete(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        self._delegate.handle_restore_complete(sandbox_id, checkpoint_id)

    def _manifests(self, sandbox_id: SandboxId) -> list[CheckpointManifest]:
        manifests: list[CheckpointManifest] = []
        for checkpoint_id in self.list_checkpoints(sandbox_id):
            manifests.append(self.get_manifest(sandbox_id, checkpoint_id))
        return manifests

    def _protected_checkpoint_ids(self, sandbox_id: SandboxId) -> set[CheckpointId]:
        latest_checkpoint: CheckpointId | None = None
        latest_process: CheckpointId | None = None
        latest_filesystem: CheckpointId | None = None
        for manifest in self._manifests(sandbox_id):
            latest_checkpoint = manifest.checkpoint_id
            if manifest.process_artifacts:
                latest_process = manifest.checkpoint_id
            if manifest.filesystem_artifacts:
                latest_filesystem = manifest.checkpoint_id
        protected: set[CheckpointId] = set()
        if latest_checkpoint is not None:
            protected.add(latest_checkpoint)
        if latest_process is not None:
            protected.add(latest_process)
        if latest_filesystem is not None:
            protected.add(latest_filesystem)
        return protected

    def _prune_unprotected(self, sandbox_id: SandboxId) -> None:
        protected = self._protected_checkpoint_ids(sandbox_id)
        for checkpoint_id in self.list_checkpoints(sandbox_id):
            if checkpoint_id not in protected:
                self.delete_checkpoint(sandbox_id, checkpoint_id)


class LatestOnlyCheckpointManager(DelegatingCheckpointManager):
    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        super().handle_checkpoint_complete(manifest)
        self._prune_unprotected(manifest.sandbox_id)


class DeleteAfterRestoreCheckpointManager(DelegatingCheckpointManager):
    def handle_restore_complete(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        super().handle_restore_complete(sandbox_id, checkpoint_id)
        for candidate in reversed(self.list_checkpoints(sandbox_id)):
            self.delete_checkpoint(sandbox_id, candidate)


class KeepAllCheckpointManager(DelegatingCheckpointManager):
    pass

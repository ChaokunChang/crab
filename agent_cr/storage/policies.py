from __future__ import annotations

from collections import defaultdict
from threading import RLock

from ..contracts import CheckpointManager
from ..ids import CheckpointId, SandboxId
from ..models import ArtifactPayload, ArtifactReference, CheckpointManifest


class DelegatingCheckpointManager(CheckpointManager):
    def __init__(self, delegate: CheckpointManager):
        self._delegate = delegate
        self._pin_lock = RLock()
        self._pinned_checkpoint_counts: dict[SandboxId, dict[CheckpointId, int]] = defaultdict(dict)

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
        if latest_checkpoint is not None:
            protected.update(self._restore_dependency_checkpoint_ids(sandbox_id, latest_checkpoint))
        with self._pin_lock:
            protected.update(self._pinned_checkpoint_counts.get(sandbox_id, {}).keys())
        return protected

    def _restore_dependency_checkpoint_ids(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> set[CheckpointId]:
        try:
            from ..workers.composite import resolve_restore_manifest

            resolved = resolve_restore_manifest(self, self.get_manifest(sandbox_id, checkpoint_id))
        except Exception:
            return set()
        protected = {checkpoint_id}
        for metadata_key in ("process_restore_checkpoint_id", "filesystem_restore_checkpoint_id"):
            raw_value = resolved.metadata.get(metadata_key)
            if raw_value is not None:
                protected.add(CheckpointId(str(raw_value)))
        return protected

    def _prune_unprotected(self, sandbox_id: SandboxId) -> None:
        protected = self._protected_checkpoint_ids(sandbox_id)
        for checkpoint_id in self.list_checkpoints(sandbox_id):
            if checkpoint_id not in protected:
                self.delete_checkpoint(sandbox_id, checkpoint_id)

    def pin_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> bool:
        try:
            self.get_manifest(sandbox_id, checkpoint_id)
        except FileNotFoundError:
            return False
        with self._pin_lock:
            sandbox_pins = self._pinned_checkpoint_counts[sandbox_id]
            sandbox_pins[checkpoint_id] = sandbox_pins.get(checkpoint_id, 0) + 1
        return True

    def unpin_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        with self._pin_lock:
            sandbox_pins = self._pinned_checkpoint_counts.get(sandbox_id)
            if not sandbox_pins:
                return
            current = sandbox_pins.get(checkpoint_id, 0)
            if current <= 1:
                sandbox_pins.pop(checkpoint_id, None)
            else:
                sandbox_pins[checkpoint_id] = current - 1
            if not sandbox_pins:
                self._pinned_checkpoint_counts.pop(sandbox_id, None)


class LatestOnlyCheckpointManager(DelegatingCheckpointManager):
    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        super().handle_checkpoint_complete(manifest)
        with self._pin_lock:
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

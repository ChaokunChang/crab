from __future__ import annotations

import atexit
from collections import defaultdict, deque
import logging
from threading import Condition, RLock, Thread
import time

from ..contracts import CheckpointManager
from ..ids import CheckpointId, SandboxId
from ..models import ArtifactPayload, ArtifactReference, CheckpointManifest

logger = logging.getLogger(__name__)
_RETENTION_DEBOUNCE_SECONDS = 0.01
_RETENTION_DISPATCHER_LOCK = RLock()
_RETENTION_DISPATCHER_CONDITION = Condition(_RETENTION_DISPATCHER_LOCK)
_RETENTION_DISPATCHER_QUEUE: deque[tuple["LatestOnlyCheckpointManager", SandboxId]] = deque()
_RETENTION_DISPATCHER_THREAD: Thread | None = None
_RETENTION_DISPATCHER_STOP = False


def _ensure_retention_dispatcher() -> None:
    global _RETENTION_DISPATCHER_THREAD
    with _RETENTION_DISPATCHER_CONDITION:
        thread = _RETENTION_DISPATCHER_THREAD
        if thread is not None and thread.is_alive():
            return
        thread = Thread(target=_run_retention_dispatcher, name="agent-cr-retention", daemon=True)
        _RETENTION_DISPATCHER_THREAD = thread
        thread.start()


def _run_retention_dispatcher() -> None:
    while True:
        with _RETENTION_DISPATCHER_CONDITION:
            while not _RETENTION_DISPATCHER_QUEUE and not _RETENTION_DISPATCHER_STOP:
                _RETENTION_DISPATCHER_CONDITION.wait()
            if _RETENTION_DISPATCHER_STOP and not _RETENTION_DISPATCHER_QUEUE:
                return
            manager, sandbox_id = _RETENTION_DISPATCHER_QUEUE.popleft()
        manager._run_prune_task(sandbox_id)


def _shutdown_retention_dispatcher() -> None:
    global _RETENTION_DISPATCHER_STOP
    thread: Thread | None = None
    with _RETENTION_DISPATCHER_CONDITION:
        _RETENTION_DISPATCHER_STOP = True
        _RETENTION_DISPATCHER_CONDITION.notify_all()
        thread = _RETENTION_DISPATCHER_THREAD
    if thread is not None:
        thread.join(timeout=1.0)


atexit.register(_shutdown_retention_dispatcher)


class DelegatingCheckpointManager(CheckpointManager):
    def __init__(self, delegate: CheckpointManager):
        self._delegate = delegate
        self._pin_locks_guard = RLock()
        self._pin_locks: dict[SandboxId, RLock] = {}
        self._pinned_checkpoint_counts: dict[SandboxId, dict[CheckpointId, int]] = defaultdict(dict)

    def _pin_lock_for(self, sandbox_id: SandboxId) -> RLock:
        with self._pin_locks_guard:
            lock = self._pin_locks.get(sandbox_id)
            if lock is None:
                lock = RLock()
                self._pin_locks[sandbox_id] = lock
            return lock

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
            try:
                manifests.append(self.get_manifest(sandbox_id, checkpoint_id))
            except FileNotFoundError:
                continue
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
        with self._pin_lock_for(sandbox_id):
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
        with self._pin_lock_for(sandbox_id):
            sandbox_pins = self._pinned_checkpoint_counts[sandbox_id]
            sandbox_pins[checkpoint_id] = sandbox_pins.get(checkpoint_id, 0) + 1
        return True

    def unpin_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        with self._pin_lock_for(sandbox_id):
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
    def __init__(
        self,
        delegate: CheckpointManager,
        *,
        delete_filesystem_checkpoints: bool = True,
    ):
        super().__init__(delegate)
        self._delete_filesystem_checkpoints = bool(delete_filesystem_checkpoints)
        self._retention_lock = RLock()
        self._retention_condition = Condition(self._retention_lock)
        self._retention_pending: set[SandboxId] = set()
        self._retention_running: set[SandboxId] = set()
        self._retention_dirty: set[SandboxId] = set()
        self._retention_touched: set[SandboxId] = set()

    @property
    def delete_filesystem_checkpoints(self) -> bool:
        return self._delete_filesystem_checkpoints

    def _protected_checkpoint_ids(self, sandbox_id: SandboxId) -> set[CheckpointId]:
        protected = super()._protected_checkpoint_ids(sandbox_id)
        if self._delete_filesystem_checkpoints:
            return protected
        for manifest in self._manifests(sandbox_id):
            if manifest.filesystem_artifacts:
                protected.add(manifest.checkpoint_id)
        return protected

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        super().handle_checkpoint_complete(manifest)
        self._schedule_prune(manifest.sandbox_id)

    def flush(self) -> None:
        with self._retention_condition:
            while self._retention_pending or self._retention_running or self._retention_dirty:
                self._retention_condition.wait(timeout=0.05)
            sandboxes = list(self._retention_touched)
        for sandbox_id in sandboxes:
            with self._pin_lock_for(sandbox_id):
                self._prune_unprotected(sandbox_id)

    def close(self) -> None:
        self.flush()

    def _schedule_prune(self, sandbox_id: SandboxId) -> None:
        with self._retention_condition:
            self._retention_touched.add(sandbox_id)
            if sandbox_id in self._retention_pending or sandbox_id in self._retention_running:
                self._retention_dirty.add(sandbox_id)
                return
            self._retention_pending.add(sandbox_id)
            self._retention_condition.notify_all()
        _ensure_retention_dispatcher()
        with _RETENTION_DISPATCHER_CONDITION:
            _RETENTION_DISPATCHER_QUEUE.append((self, sandbox_id))
            _RETENTION_DISPATCHER_CONDITION.notify()

    def _run_prune_task(self, sandbox_id: SandboxId) -> None:
        while True:
            time.sleep(_RETENTION_DEBOUNCE_SECONDS)
            with self._retention_condition:
                if sandbox_id in self._retention_dirty:
                    self._retention_dirty.discard(sandbox_id)
                    continue
                self._retention_pending.discard(sandbox_id)
                self._retention_running.add(sandbox_id)
                self._retention_condition.notify_all()
                break
        try:
            with self._pin_lock_for(sandbox_id):
                self._prune_unprotected(sandbox_id)
        except Exception:
            logger.exception("Background checkpoint retention failed for sandbox=%s", sandbox_id)
        finally:
            rerun = False
            with self._retention_condition:
                self._retention_running.discard(sandbox_id)
                if sandbox_id in self._retention_dirty:
                    self._retention_dirty.discard(sandbox_id)
                    self._retention_pending.add(sandbox_id)
                    rerun = True
                self._retention_condition.notify_all()
            if rerun:
                _ensure_retention_dispatcher()
                with _RETENTION_DISPATCHER_CONDITION:
                    _RETENTION_DISPATCHER_QUEUE.append((self, sandbox_id))
                    _RETENTION_DISPATCHER_CONDITION.notify()


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

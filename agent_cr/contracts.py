from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable

from .ids import CheckpointId, JobId, SandboxId
from .models import (
    ArtifactPayload,
    ArtifactReference,
    CheckpointJob,
    CheckpointManifest,
    CheckpointResult,
    RequestContext,
    RequestStateChange,
    RestoreJob,
    RestoreResult,
    EBPFEvent,
    RuntimeOperationStatus,
    RuntimeCapabilities,
    SandboxDescription,
    SandboxExecResult,
    SandboxRuntimeState,
    SandboxSnapshot,
    SchedulerCheckpointDecision,
    WorkerStepResult,
)


class Runtime(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    def version(self) -> str | None:
        return None

    @abstractmethod
    def capabilities(self) -> RuntimeCapabilities:
        raise NotImplementedError

    @abstractmethod
    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        raise NotImplementedError

    @abstractmethod
    def stop(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def pause(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def resume(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def sync_runtime_state(self, sandbox_id: SandboxId, *, is_running: bool) -> None:
        raise NotImplementedError

    @abstractmethod
    def prepare_for_restore(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_restored(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        raise NotImplementedError

    @abstractmethod
    def write_bundle_spec(self, bundle_dir) -> None:
        raise NotImplementedError

    @abstractmethod
    def inspect_runtime(self, sandbox_id: SandboxId) -> SandboxRuntimeState:
        raise NotImplementedError

    @abstractmethod
    def exec(
        self,
        sandbox_id: SandboxId,
        argv: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, object] | None = None,
        user: str | None = None,
        timeout_s: float | None = None,
        capture_output: bool = True,
    ) -> SandboxExecResult:
        raise NotImplementedError

    @abstractmethod
    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ) -> RuntimeOperationStatus:
        raise NotImplementedError

    @abstractmethod
    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        raise NotImplementedError

    @abstractmethod
    def process_checkpoint_location(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> str | None:
        raise NotImplementedError

    @abstractmethod
    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        raise NotImplementedError

    @abstractmethod
    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        raise NotImplementedError

    @abstractmethod
    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        raise NotImplementedError

    @abstractmethod
    def delete_runtime(
        self,
        sandbox_id: SandboxId,
        *,
        force: bool = True,
        ignore_missing: bool = True,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def destroy_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path,
    ) -> str:
        raise NotImplementedError


class ProcessCWorker(ABC):
    @abstractmethod
    def checkpoint(
        self,
        job: CheckpointJob,
        checkpoint_id: CheckpointId,
    ) -> WorkerStepResult:
        raise NotImplementedError


class ProcessRWorker(ABC):
    @abstractmethod
    def restore(
        self,
        job: RestoreJob,
        manifest: CheckpointManifest,
    ) -> WorkerStepResult:
        raise NotImplementedError


class FileSystemCWorker(ABC):
    @abstractmethod
    def checkpoint(
        self,
        job: CheckpointJob,
        checkpoint_id: CheckpointId,
    ) -> WorkerStepResult:
        raise NotImplementedError


class FileSystemRWorker(ABC):
    @abstractmethod
    def restore(
        self,
        job: RestoreJob,
        manifest: CheckpointManifest,
    ) -> WorkerStepResult:
        raise NotImplementedError


class CheckpointManager(ABC):
    @abstractmethod
    def put_manifest(self, manifest: CheckpointManifest) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        raise NotImplementedError

    @abstractmethod
    def put_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        artifact: ArtifactPayload,
    ) -> ArtifactReference:
        raise NotImplementedError

    @abstractmethod
    def get_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        raise NotImplementedError

    @abstractmethod
    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        raise NotImplementedError

    @abstractmethod
    def handle_restore_complete(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        raise NotImplementedError


class RemoteCheckpointBackend(ABC):
    @abstractmethod
    def upload_manifest(self, manifest: CheckpointManifest) -> None:
        raise NotImplementedError

    @abstractmethod
    def download_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        raise NotImplementedError

    @abstractmethod
    def upload_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
        data: bytes,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def download_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bytes:
        raise NotImplementedError


class SandboxInspector(ABC):
    @abstractmethod
    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        raise NotImplementedError

    @abstractmethod
    def mark_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        *,
        process: bool,
        filesystem: bool,
        at: datetime,
    ) -> None:
        raise NotImplementedError


class EBPFEventCollector(ABC):
    @abstractmethod
    def poll(self, sandbox_id: SandboxId, since: datetime | None = None) -> list[EBPFEvent]:
        raise NotImplementedError


class RequestInterceptorHook(ABC):
    @abstractmethod
    def on_request_start(self, context: RequestContext) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_request_end(self, context: RequestContext) -> None:
        raise NotImplementedError


class TelemetrySink(ABC):
    @abstractmethod
    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        raise NotImplementedError

    @abstractmethod
    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        raise NotImplementedError


class CompositeCheckpointWorker(ABC):
    @abstractmethod
    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        raise NotImplementedError


class CompositeRestoreWorker(ABC):
    @abstractmethod
    def restore(self, job: RestoreJob) -> RestoreResult:
        raise NotImplementedError


class SchedulerStateStore(ABC):
    @abstractmethod
    def set_last_checkpoint(self, sandbox_id: SandboxId, checkpoint_time: datetime) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_last_checkpoint(self, sandbox_id: SandboxId) -> datetime | None:
        raise NotImplementedError

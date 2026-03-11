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
    SandboxSnapshot,
    ScheduleDecision,
    SchedulerCheckpointDecision,
    WorkerStepResult,
)


class SandboxRuntimeAdapter(ABC):
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
    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
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


class CRPolicy(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, snapshot: SandboxSnapshot) -> ScheduleDecision:
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


class SandboxManager(ABC):
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
    def prepare_for_restore(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, sandbox_id: SandboxId) -> None:
        raise NotImplementedError

    @abstractmethod
    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
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

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
    DryRunStatus,
    RequestContext,
    RestoreJob,
    RestoreResult,
    RuntimeCapabilities,
    SandboxDescription,
    SandboxSnapshot,
    ScheduleDecision,
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
    def plan_process_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> DryRunStatus:
        raise NotImplementedError

    @abstractmethod
    def plan_process_restore(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> DryRunStatus:
        raise NotImplementedError

    @abstractmethod
    def plan_filesystem_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> DryRunStatus:
        raise NotImplementedError

    @abstractmethod
    def plan_filesystem_restore(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> DryRunStatus:
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

    @abstractmethod
    def enqueue_checkpoint_job(self, job: CheckpointJob) -> None:
        raise NotImplementedError

    @abstractmethod
    def pop_checkpoint_job(self) -> CheckpointJob | None:
        raise NotImplementedError

    @abstractmethod
    def pending_jobs(self) -> Iterable[CheckpointJob]:
        raise NotImplementedError

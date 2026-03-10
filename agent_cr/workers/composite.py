from __future__ import annotations

from dataclasses import replace

from ..contracts import (
    CheckpointManager,
    CompositeCheckpointWorker,
    CompositeRestoreWorker,
    FileSystemCWorker,
    FileSystemRWorker,
    ProcessCWorker,
    ProcessRWorker,
    SandboxRuntimeAdapter,
)
from ..ids import CheckpointId
from ..models import (
    ArtifactKind,
    CheckpointJob,
    CheckpointManifest,
    CheckpointResult,
    FailureCode,
    JobStatus,
    RestoreJob,
    RestoreResult,
    utc_now,
)


class DefaultCWorker(CompositeCheckpointWorker):
    def __init__(
        self,
        process_worker: ProcessCWorker,
        filesystem_worker: FileSystemCWorker,
        checkpoint_manager: CheckpointManager,
        runtime_adapter: SandboxRuntimeAdapter,
    ):
        self._process_worker = process_worker
        self._filesystem_worker = filesystem_worker
        self._checkpoint_manager = checkpoint_manager
        self._runtime_adapter = runtime_adapter

    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        started = utc_now()
        checkpoint_id = CheckpointId(str(job.metadata.get("checkpoint_id", CheckpointId.new())))

        process_step = self._process_worker.checkpoint(job, checkpoint_id)
        if not process_step.success:
            return CheckpointResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                manifest=None,
                failure_code=process_step.failure_code,
                message=process_step.message,
                operation_statuses=(process_step.operation_status,),
            )

        fs_step = self._filesystem_worker.checkpoint(job, checkpoint_id)
        if not fs_step.success:
            return CheckpointResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                manifest=None,
                failure_code=fs_step.failure_code,
                message=fs_step.message,
                operation_statuses=(process_step.operation_status, fs_step.operation_status),
            )

        refs = []
        for artifact in [*process_step.artifacts, *fs_step.artifacts]:
            refs.append(
                self._checkpoint_manager.put_artifact(
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=checkpoint_id,
                    artifact=artifact,
                )
            )

        process_refs = [x for x in refs if x.kind == ArtifactKind.PROCESS]
        fs_refs = [x for x in refs if x.kind == ArtifactKind.FILESYSTEM]

        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=checkpoint_id,
            sandbox_id=job.sandbox_id,
            created_at=utc_now(),
            runtime_name=self._runtime_adapter.name,
            runtime_version=self._runtime_adapter.version,
            process_artifacts=process_refs,
            filesystem_artifacts=fs_refs,
            metadata={
                "job_id": str(job.job_id),
                "reason": job.reason,
            },
        ).with_integrity()
        try:
            self._checkpoint_manager.put_manifest(manifest)
        except Exception as exc:
            return CheckpointResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                manifest=None,
                failure_code=FailureCode.STORAGE_ERROR,
                message=str(exc),
                operation_statuses=(process_step.operation_status, fs_step.operation_status),
            )

        return CheckpointResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=started,
            finished_at=utc_now(),
            manifest=manifest,
            operation_statuses=(process_step.operation_status, fs_step.operation_status),
        )


class DefaultRWorker(CompositeRestoreWorker):
    def __init__(
        self,
        process_worker: ProcessRWorker,
        filesystem_worker: FileSystemRWorker,
        checkpoint_manager: CheckpointManager,
    ):
        self._process_worker = process_worker
        self._filesystem_worker = filesystem_worker
        self._checkpoint_manager = checkpoint_manager

    def restore(self, job: RestoreJob) -> RestoreResult:
        started = utc_now()
        try:
            manifest = self._checkpoint_manager.get_manifest(
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
            )
        except Exception as exc:
            return RestoreResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=FailureCode.STORAGE_ERROR,
                message=str(exc),
            )

        restore_job = job
        process_state_ref = next(
            (x for x in manifest.process_artifacts if x.name == "process_state.tar.gz"),
            None,
        )
        if process_state_ref is not None:
            try:
                payload = self._checkpoint_manager.get_artifact(
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=job.checkpoint_id,
                    reference=process_state_ref,
                )
            except Exception as exc:
                return RestoreResult(
                    job_id=job.job_id,
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=job.checkpoint_id,
                    status=JobStatus.FAILED,
                    started_at=started,
                    finished_at=utc_now(),
                    failure_code=FailureCode.STORAGE_ERROR,
                    message=str(exc),
                )
            restore_job = replace(job, metadata={**job.metadata, "process_state_payload": payload})

        fs_step = self._filesystem_worker.restore(restore_job, manifest)
        if not fs_step.success:
            return RestoreResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=fs_step.failure_code,
                message=fs_step.message,
                operation_statuses=(fs_step.operation_status,),
            )

        process_step = self._process_worker.restore(restore_job, manifest)
        if not process_step.success:
            return RestoreResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=process_step.failure_code,
                message=process_step.message,
                operation_statuses=(fs_step.operation_status, process_step.operation_status),
            )

        return RestoreResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=job.checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=started,
            finished_at=utc_now(),
            operation_statuses=(fs_step.operation_status, process_step.operation_status),
        )

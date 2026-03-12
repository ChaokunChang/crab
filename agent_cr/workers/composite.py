from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging

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

logger = logging.getLogger(__name__)

_PROCESS_RESTORE_CHECKPOINT_ID = "process_restore_checkpoint_id"
_FILESYSTEM_RESTORE_CHECKPOINT_ID = "filesystem_restore_checkpoint_id"


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
        logger.info(
            "Starting composite checkpoint for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            checkpoint_id,
        )

        process_step = None
        fs_step = None
        if job.checkpoint_process and job.checkpoint_filesystem:
            with ThreadPoolExecutor(max_workers=2) as pool:
                process_future = pool.submit(self._process_worker.checkpoint, job, checkpoint_id)
                fs_future = pool.submit(self._filesystem_worker.checkpoint, job, checkpoint_id)
                process_step = process_future.result()
                fs_step = fs_future.result()
        elif job.checkpoint_process:
            process_step = self._process_worker.checkpoint(job, checkpoint_id)
        elif job.checkpoint_filesystem:
            fs_step = self._filesystem_worker.checkpoint(job, checkpoint_id)

        failed_step = None
        if process_step is not None and not process_step.success:
            failed_step = process_step
            logger.warning(
                "Process checkpoint step failed for job %s sandbox=%s code=%s message=%s",
                job.job_id,
                job.sandbox_id,
                process_step.failure_code.value,
                process_step.message,
            )
        elif fs_step is not None and not fs_step.success:
            failed_step = fs_step
            logger.warning(
                "Filesystem checkpoint step failed for job %s sandbox=%s code=%s message=%s",
                job.job_id,
                job.sandbox_id,
                fs_step.failure_code.value,
                fs_step.message,
            )

        operation_statuses = tuple(
            step.operation_status for step in (process_step, fs_step) if step is not None
        )
        if failed_step is not None:
            return CheckpointResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                manifest=None,
                failure_code=failed_step.failure_code,
                message=failed_step.message,
                operation_statuses=operation_statuses,
            )

        refs = []
        for artifact in [
            *(process_step.artifacts if process_step is not None else []),
            *(fs_step.artifacts if fs_step is not None else []),
        ]:
            refs.append(
                self._checkpoint_manager.put_artifact(
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=checkpoint_id,
                    artifact=artifact,
                )
            )
        logger.debug(
            "Stored %d artifacts for job %s sandbox=%s checkpoint=%s",
            len(refs),
            job.job_id,
            job.sandbox_id,
            checkpoint_id,
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
                "leave_running": job.leave_running,
            },
        ).with_integrity()
        try:
            self._checkpoint_manager.put_manifest(manifest)
            self._checkpoint_manager.handle_checkpoint_complete(manifest)
        except Exception as exc:
            logger.exception(
                "Failed to persist manifest for job %s sandbox=%s checkpoint=%s",
                job.job_id,
                job.sandbox_id,
                checkpoint_id,
            )
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
                operation_statuses=operation_statuses,
            )

        logger.info(
            "Composite checkpoint succeeded for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            checkpoint_id,
        )
        return CheckpointResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=started,
            finished_at=utc_now(),
            manifest=manifest,
            operation_statuses=operation_statuses,
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
        logger.info(
            "Starting composite restore for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
        )
        try:
            manifest = self._checkpoint_manager.get_manifest(
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
            )
            manifest = self._resolve_restore_manifest(manifest)
        except Exception as exc:
            logger.exception(
                "Failed to load manifest for restore job %s sandbox=%s checkpoint=%s",
                job.job_id,
                job.sandbox_id,
                job.checkpoint_id,
            )
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

        operation_statuses = []

        fs_step = None
        if manifest.filesystem_artifacts:
            fs_step = self._filesystem_worker.restore(job, manifest)
        if fs_step is not None and not fs_step.success:
            logger.warning(
                "Filesystem restore step failed for job %s sandbox=%s code=%s message=%s",
                job.job_id,
                job.sandbox_id,
                fs_step.failure_code.value,
                fs_step.message,
            )
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
        if fs_step is not None:
            operation_statuses.append(fs_step.operation_status)

        process_step = None
        if manifest.process_artifacts:
            process_step = self._process_worker.restore(job, manifest)
        if process_step is not None and not process_step.success:
            logger.warning(
                "Process restore step failed for job %s sandbox=%s code=%s message=%s",
                job.job_id,
                job.sandbox_id,
                process_step.failure_code.value,
                process_step.message,
            )
            return RestoreResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.FAILED,
                started_at=started,
                finished_at=utc_now(),
                failure_code=process_step.failure_code,
                message=process_step.message,
                operation_statuses=tuple(operation_statuses + [process_step.operation_status]),
            )
        if process_step is not None:
            operation_statuses.append(process_step.operation_status)

        logger.info(
            "Composite restore succeeded for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
        )
        return RestoreResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=job.checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=started,
            finished_at=utc_now(),
            operation_statuses=tuple(operation_statuses),
        )

    def _resolve_restore_manifest(self, manifest: CheckpointManifest) -> CheckpointManifest:
        process_artifacts = list(manifest.process_artifacts)
        filesystem_artifacts = list(manifest.filesystem_artifacts)
        metadata = dict(manifest.metadata)
        process_checkpoint_id = manifest.checkpoint_id if process_artifacts else None
        filesystem_checkpoint_id = manifest.checkpoint_id if filesystem_artifacts else None

        if process_artifacts and filesystem_artifacts:
            metadata.setdefault(_PROCESS_RESTORE_CHECKPOINT_ID, str(manifest.checkpoint_id))
            metadata.setdefault(_FILESYSTEM_RESTORE_CHECKPOINT_ID, str(manifest.checkpoint_id))
            return replace(manifest, metadata=metadata).with_integrity()

        checkpoints = self._checkpoint_manager.list_checkpoints(manifest.sandbox_id)
        try:
            current_index = checkpoints.index(manifest.checkpoint_id)
            candidate_ids = reversed(checkpoints[:current_index])
        except ValueError:
            candidate_ids = reversed(checkpoints)

        for checkpoint_id in candidate_ids:
            if process_artifacts and filesystem_artifacts:
                break
            candidate = self._checkpoint_manager.get_manifest(manifest.sandbox_id, checkpoint_id)
            if not process_artifacts and candidate.process_artifacts:
                process_artifacts = list(candidate.process_artifacts)
                process_checkpoint_id = candidate.checkpoint_id
            if not filesystem_artifacts and candidate.filesystem_artifacts:
                filesystem_artifacts = list(candidate.filesystem_artifacts)
                filesystem_checkpoint_id = candidate.checkpoint_id

        if process_checkpoint_id is not None:
            metadata[_PROCESS_RESTORE_CHECKPOINT_ID] = str(process_checkpoint_id)
        if filesystem_checkpoint_id is not None:
            metadata[_FILESYSTEM_RESTORE_CHECKPOINT_ID] = str(filesystem_checkpoint_id)

        return replace(
            manifest,
            process_artifacts=process_artifacts,
            filesystem_artifacts=filesystem_artifacts,
            metadata=metadata,
        ).with_integrity()

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import logging
import time
from typing import Callable

from ..contracts import (
    CheckpointManager,
    CompositeCheckpointWorker,
    CompositeRestoreWorker,
    FileSystemCWorker,
    FileSystemRWorker,
    ProcessCWorker,
    ProcessRWorker,
    Runtime,
    TelemetrySink,
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
from ..telemetry import NoopTelemetrySink

logger = logging.getLogger(__name__)

_PROCESS_RESTORE_CHECKPOINT_ID = "process_restore_checkpoint_id"
_PROCESS_RESTORE_REPLAY_ACTION_COUNT = "process_restore_replay_action_count"
_FILESYSTEM_RESTORE_CHECKPOINT_ID = "filesystem_restore_checkpoint_id"
_FILESYSTEM_RESTORE_REPLAY_ACTION_COUNT = "filesystem_restore_replay_action_count"
_CAPTURES_INFLIGHT_LLM = "captures_inflight_llm"
_CAPTURED_REQUEST_ID = "captured_request_id"
_CAPTURED_REQUEST_PROVIDER = "captured_request_provider"
_CAPTURED_REQUEST_STARTED_AT = "captured_request_started_at"


def resolve_restore_manifest(
    checkpoint_manager: CheckpointManager,
    manifest: CheckpointManifest,
) -> CheckpointManifest:
    checkpoints = checkpoint_manager.list_checkpoints(manifest.sandbox_id)
    try:
        current_index = checkpoints.index(manifest.checkpoint_id)
        candidate_ids = checkpoints[: current_index + 1]
    except ValueError:
        candidate_ids = checkpoints

    candidates = [checkpoint_manager.get_manifest(manifest.sandbox_id, checkpoint_id) for checkpoint_id in candidate_ids]
    current_candidate = candidates[-1] if candidates else manifest

    filesystem_manifest = _latest_manifest_with_artifacts(
        candidates,
        include=lambda candidate: bool(candidate.filesystem_artifacts),
    )
    process_manifest = _latest_manifest_with_artifacts(
        candidates,
        include=lambda candidate: bool(candidate.process_artifacts),
    )

    process_artifacts = [] if process_manifest is None else list(process_manifest.process_artifacts)
    filesystem_artifacts = [] if filesystem_manifest is None else list(filesystem_manifest.filesystem_artifacts)
    metadata = dict(current_candidate.metadata)
    if process_manifest is not None:
        metadata[_PROCESS_RESTORE_CHECKPOINT_ID] = str(process_manifest.checkpoint_id)
        metadata[_PROCESS_RESTORE_REPLAY_ACTION_COUNT] = _committed_replay_action_count(
            process_manifest.metadata
        )
    if filesystem_manifest is not None:
        metadata[_FILESYSTEM_RESTORE_CHECKPOINT_ID] = str(filesystem_manifest.checkpoint_id)
        metadata[_FILESYSTEM_RESTORE_REPLAY_ACTION_COUNT] = _committed_replay_action_count(
            filesystem_manifest.metadata
        )
    _copy_process_restore_metadata(metadata, {} if process_manifest is None else process_manifest.metadata)

    return replace(
        current_candidate,
        process_artifacts=process_artifacts,
        filesystem_artifacts=filesystem_artifacts,
        metadata=metadata,
    ).with_integrity()


def _latest_manifest_with_artifacts(
    manifests: list[CheckpointManifest],
    *,
    include: Callable[[CheckpointManifest], bool],
) -> CheckpointManifest | None:
    for candidate in reversed(manifests):
        if include(candidate):
            return candidate
    return None


def _replay_action_count(metadata: dict[str, object]) -> int:
    raw_value = metadata.get("benchmark_replay_action_count", 0)
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def _committed_replay_action_count(metadata: dict[str, object]) -> int:
    replay_action_count = _replay_action_count(metadata)
    if bool(metadata.get(_CAPTURES_INFLIGHT_LLM, False)):
        return max(0, replay_action_count - 1)
    return replay_action_count


def _copy_process_restore_metadata(target: dict[str, object], source: dict[str, object]) -> None:
    target[_CAPTURES_INFLIGHT_LLM] = bool(source.get(_CAPTURES_INFLIGHT_LLM, False))
    for key in (_CAPTURED_REQUEST_ID, _CAPTURED_REQUEST_PROVIDER, _CAPTURED_REQUEST_STARTED_AT):
        value = source.get(key)
        if value is None:
            target.pop(key, None)
        else:
            target[key] = value


class DefaultCWorker(CompositeCheckpointWorker):
    def __init__(
        self,
        process_worker: ProcessCWorker,
        filesystem_worker: FileSystemCWorker,
        checkpoint_manager: CheckpointManager,
        runtime: Runtime,
        checkpoint_guard: Callable[[CheckpointJob], tuple[bool, str | None]] | None = None,
        telemetry: TelemetrySink | None = None,
    ):
        self._process_worker = process_worker
        self._filesystem_worker = filesystem_worker
        self._checkpoint_manager = checkpoint_manager
        self._runtime = runtime
        self._checkpoint_guard = checkpoint_guard
        self._telemetry = telemetry or NoopTelemetrySink()

    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        job = self._ensure_restorable_checkpoint(job)
        started = utc_now()
        checkpoint_id = CheckpointId(str(job.metadata.get("checkpoint_id", CheckpointId.new())))
        if self._checkpoint_guard is not None:
            allowed, message = self._checkpoint_guard(job)
            if not allowed:
                logger.info(
                    "Skipping composite checkpoint for job %s sandbox=%s checkpoint=%s reason=%s",
                    job.job_id,
                    job.sandbox_id,
                    checkpoint_id,
                    "" if message is None else message,
                )
                return CheckpointResult(
                    job_id=job.job_id,
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=checkpoint_id,
                    status=JobStatus.FAILED,
                    started_at=started,
                    finished_at=utc_now(),
                    manifest=None,
                    failure_code=FailureCode.VALIDATION_ERROR,
                    message=message or "checkpoint_rejected",
                )
        logger.info(
            "Starting composite checkpoint for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            checkpoint_id,
        )

        process_step = None
        fs_step = None
        process_duration_ms = 0.0
        filesystem_duration_ms = 0.0
        if job.checkpoint_process and job.checkpoint_filesystem:
            with ThreadPoolExecutor(max_workers=2) as pool:
                process_future = pool.submit(self._timed_checkpoint_process, job, checkpoint_id)
                fs_future = pool.submit(self._timed_checkpoint_filesystem, job, checkpoint_id)
                process_step, process_duration_ms = process_future.result()
                fs_step, filesystem_duration_ms = fs_future.result()
        elif job.checkpoint_process:
            process_step, process_duration_ms = self._timed_checkpoint_process(job, checkpoint_id)
        elif job.checkpoint_filesystem:
            fs_step, filesystem_duration_ms = self._timed_checkpoint_filesystem(job, checkpoint_id)

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

        artifact_started = time.perf_counter()
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
        self._telemetry.emit_metric(
            "checkpoint.persist_artifacts_ms",
            (time.perf_counter() - artifact_started) * 1000.0,
            {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id), "artifact_count": len(refs)},
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
            runtime_name=self._runtime.name,
            runtime_version=self._runtime.version,
            process_artifacts=process_refs,
            filesystem_artifacts=fs_refs,
            metadata={
                **job.metadata,
                "job_id": str(job.job_id),
                "reason": job.reason,
                "leave_running": job.leave_running,
            },
        ).with_integrity()
        manifest_started = time.perf_counter()
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
        self._telemetry.emit_metric(
            "checkpoint.persist_manifest_ms",
            (time.perf_counter() - manifest_started) * 1000.0,
            {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
        )
        if process_step is not None:
            self._telemetry.emit_metric(
                "checkpoint.process_ms",
                process_duration_ms,
                {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
            )
        if fs_step is not None:
            self._telemetry.emit_metric(
                "checkpoint.filesystem_ms",
                filesystem_duration_ms,
                {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
            )
        self._telemetry.emit_metric(
            "checkpoint.total_ms",
            (utc_now() - started).total_seconds() * 1000.0,
            {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(checkpoint_id)},
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

    def _ensure_restorable_checkpoint(self, job: CheckpointJob) -> CheckpointJob:
        if job.checkpoint_process or not job.checkpoint_filesystem:
            return job
        if self._has_process_ancestor(job.sandbox_id):
            return job
        logger.info(
            "Promoting checkpoint scope to include process for sandbox %s because no prior process checkpoint exists",
            job.sandbox_id,
        )
        metadata = dict(job.metadata)
        metadata.setdefault("promoted_process_checkpoint", True)
        metadata.setdefault("promoted_process_checkpoint_reason", "missing_process_ancestor")
        return replace(job, checkpoint_process=True, metadata=metadata)

    def _has_process_ancestor(self, sandbox_id) -> bool:
        try:
            checkpoint_ids = self._checkpoint_manager.list_checkpoints(sandbox_id)
        except Exception:
            logger.warning(
                "Falling back to process checkpoint for sandbox %s because existing checkpoints could not be listed",
                sandbox_id,
            )
            return False

        for checkpoint_id in reversed(checkpoint_ids):
            try:
                manifest = self._checkpoint_manager.get_manifest(sandbox_id, checkpoint_id)
            except FileNotFoundError:
                continue
            if manifest.process_artifacts:
                return True
        return False

    def _timed_checkpoint_process(
        self,
        job: CheckpointJob,
        checkpoint_id: CheckpointId,
    ) -> tuple[WorkerStepResult, float]:
        started = time.perf_counter()
        return self._process_worker.checkpoint(job, checkpoint_id), (time.perf_counter() - started) * 1000.0

    def _timed_checkpoint_filesystem(
        self,
        job: CheckpointJob,
        checkpoint_id: CheckpointId,
    ) -> tuple[WorkerStepResult, float]:
        started = time.perf_counter()
        return self._filesystem_worker.checkpoint(job, checkpoint_id), (time.perf_counter() - started) * 1000.0


class DefaultRWorker(CompositeRestoreWorker):
    def __init__(
        self,
        process_worker: ProcessRWorker,
        filesystem_worker: FileSystemRWorker,
        checkpoint_manager: CheckpointManager,
        telemetry: TelemetrySink | None = None,
    ):
        self._process_worker = process_worker
        self._filesystem_worker = filesystem_worker
        self._checkpoint_manager = checkpoint_manager
        self._telemetry = telemetry or NoopTelemetrySink()

    def restore(self, job: RestoreJob) -> RestoreResult:
        started = utc_now()
        logger.info(
            "Starting composite restore for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
        )
        manifest_started = time.perf_counter()
        try:
            manifest = self._checkpoint_manager.get_manifest(
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
            )
            manifest = resolve_restore_manifest(self._checkpoint_manager, manifest)
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
        self._telemetry.emit_metric(
            "restore.resolve_manifest_ms",
            (time.perf_counter() - manifest_started) * 1000.0,
            {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(job.checkpoint_id)},
        )

        operation_statuses = []

        fs_step = None
        filesystem_duration_ms = 0.0
        if manifest.filesystem_artifacts:
            restore_started = time.perf_counter()
            fs_step = self._filesystem_worker.restore(job, manifest)
            filesystem_duration_ms = (time.perf_counter() - restore_started) * 1000.0
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
            self._telemetry.emit_metric(
                "restore.filesystem_ms",
                filesystem_duration_ms,
                {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(job.checkpoint_id)},
            )

        process_step = None
        process_duration_ms = 0.0
        if manifest.process_artifacts:
            restore_started = time.perf_counter()
            process_step = self._process_worker.restore(job, manifest)
            process_duration_ms = (time.perf_counter() - restore_started) * 1000.0
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
            self._telemetry.emit_metric(
                "restore.process_ms",
                process_duration_ms,
                {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(job.checkpoint_id)},
            )

        logger.info(
            "Composite restore succeeded for job %s sandbox=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
        )
        self._telemetry.emit_metric(
            "restore.total_ms",
            (utc_now() - started).total_seconds() * 1000.0,
            {"sandbox_id": str(job.sandbox_id), "checkpoint_id": str(job.checkpoint_id)},
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

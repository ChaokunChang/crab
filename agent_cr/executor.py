from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock

from .config import ExecutorConfig
from .contracts import CompositeCheckpointWorker, CompositeRestoreWorker, TelemetrySink
from .ids import CheckpointId, JobId
from .models import (
    CheckpointJob,
    CheckpointResult,
    FailureCode,
    JobRecord,
    JobStatus,
    JobType,
    RestoreJob,
    RestoreResult,
    utc_now,
)
from .telemetry import NoopTelemetrySink


class CRExecutor:
    def __init__(
        self,
        config: ExecutorConfig,
        checkpoint_worker: CompositeCheckpointWorker,
        restore_worker: CompositeRestoreWorker,
        telemetry: TelemetrySink | None = None,
    ):
        self._config = config
        self._checkpoint_worker = checkpoint_worker
        self._restore_worker = restore_worker
        self._telemetry = telemetry or NoopTelemetrySink()
        self._pool = ThreadPoolExecutor(max_workers=config.max_workers)
        self._lock = Lock()
        self._records: dict[JobId, JobRecord] = {}

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def run_checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        fut = self._submit_checkpoint(job)
        return fut.result()

    def run_restore(self, job: RestoreJob) -> RestoreResult:
        fut = self._submit_restore(job)
        return fut.result()

    def run_checkpoints(self, jobs: list[CheckpointJob]) -> list[CheckpointResult]:
        futures = [self._submit_checkpoint(job) for job in jobs]
        return [x.result() for x in futures]

    def run_restores(self, jobs: list[RestoreJob]) -> list[RestoreResult]:
        futures = [self._submit_restore(job) for job in jobs]
        return [x.result() for x in futures]

    def get_job_record(self, job_id: JobId) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def _submit_checkpoint(self, job: CheckpointJob) -> Future[CheckpointResult]:
        self._set_record(
            JobRecord(
                job_id=job.job_id,
                job_type=JobType.CHECKPOINT,
                sandbox_id=job.sandbox_id,
                checkpoint_id=None,
                status=JobStatus.PENDING,
                created_at=utc_now(),
            )
        )
        self._telemetry.emit_event(
            "executor.job_pending",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.CHECKPOINT.value,
                "sandbox_id": str(job.sandbox_id),
            },
        )
        return self._pool.submit(self._execute_checkpoint, job)

    def _submit_restore(self, job: RestoreJob) -> Future[RestoreResult]:
        self._set_record(
            JobRecord(
                job_id=job.job_id,
                job_type=JobType.RESTORE,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.PENDING,
                created_at=utc_now(),
            )
        )
        self._telemetry.emit_event(
            "executor.job_pending",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.RESTORE.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(job.checkpoint_id),
            },
        )
        return self._pool.submit(self._execute_restore, job)

    def _execute_checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        started = utc_now()
        self._set_record(
            JobRecord(
                job_id=job.job_id,
                job_type=JobType.CHECKPOINT,
                sandbox_id=job.sandbox_id,
                checkpoint_id=None,
                status=JobStatus.RUNNING,
                created_at=started,
                started_at=started,
            )
        )

        self._telemetry.emit_event(
            "executor.job_running",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.CHECKPOINT.value,
                "sandbox_id": str(job.sandbox_id),
            },
        )

        last_result: CheckpointResult | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                last_result = self._checkpoint_worker.checkpoint(job)
                if last_result.status == JobStatus.SUCCEEDED:
                    break
            except Exception as exc:
                finished = utc_now()
                last_result = CheckpointResult(
                    job_id=job.job_id,
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=CheckpointId.new(prefix="failed"),
                    status=JobStatus.FAILED,
                    started_at=started,
                    finished_at=finished,
                    manifest=None,
                    failure_code=FailureCode.RUNTIME_ERROR,
                    message=str(exc),
                )
            if attempt < self._config.max_retries:
                time.sleep(self._config.retry_backoff_seconds * (attempt + 1))

        assert last_result is not None
        self._set_record(
            JobRecord(
                job_id=job.job_id,
                job_type=JobType.CHECKPOINT,
                sandbox_id=job.sandbox_id,
                checkpoint_id=last_result.checkpoint_id,
                status=last_result.status,
                created_at=started,
                started_at=started,
                finished_at=last_result.finished_at,
                failure_code=last_result.failure_code,
                message=last_result.message,
            )
        )

        duration_ms = (last_result.finished_at - started).total_seconds() * 1000.0
        self._telemetry.emit_metric(
            "executor.job_duration_ms",
            duration_ms,
            {
                "job_type": JobType.CHECKPOINT.value,
                "status": last_result.status.value,
            },
        )
        self._telemetry.emit_event(
            "executor.job_finished",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.CHECKPOINT.value,
                "status": last_result.status.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(last_result.checkpoint_id),
            },
        )
        return last_result

    def _execute_restore(self, job: RestoreJob) -> RestoreResult:
        started = utc_now()
        self._set_record(
            JobRecord(
                job_id=job.job_id,
                job_type=JobType.RESTORE,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=JobStatus.RUNNING,
                created_at=started,
                started_at=started,
            )
        )

        self._telemetry.emit_event(
            "executor.job_running",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.RESTORE.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(job.checkpoint_id),
            },
        )

        last_result: RestoreResult | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                last_result = self._restore_worker.restore(job)
                if last_result.status == JobStatus.SUCCEEDED:
                    break
            except Exception as exc:
                last_result = RestoreResult(
                    job_id=job.job_id,
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=job.checkpoint_id,
                    status=JobStatus.FAILED,
                    started_at=started,
                    finished_at=utc_now(),
                    failure_code=FailureCode.RUNTIME_ERROR,
                    message=str(exc),
                )
            if attempt < self._config.max_retries:
                time.sleep(self._config.retry_backoff_seconds * (attempt + 1))

        assert last_result is not None
        self._set_record(
            JobRecord(
                job_id=job.job_id,
                job_type=JobType.RESTORE,
                sandbox_id=job.sandbox_id,
                checkpoint_id=job.checkpoint_id,
                status=last_result.status,
                created_at=started,
                started_at=started,
                finished_at=last_result.finished_at,
                failure_code=last_result.failure_code,
                message=last_result.message,
            )
        )

        duration_ms = (last_result.finished_at - started).total_seconds() * 1000.0
        self._telemetry.emit_metric(
            "executor.job_duration_ms",
            duration_ms,
            {
                "job_type": JobType.RESTORE.value,
                "status": last_result.status.value,
            },
        )
        self._telemetry.emit_event(
            "executor.job_finished",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.RESTORE.value,
                "status": last_result.status.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(job.checkpoint_id),
            },
        )
        return last_result

    def _set_record(self, record: JobRecord) -> None:
        with self._lock:
            self._records[record.job_id] = record

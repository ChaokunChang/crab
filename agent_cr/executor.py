from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import logging
from threading import BoundedSemaphore, Condition, Lock, Thread
import time

from .config import ExecutorConfig
from .contracts import CompositeCheckpointWorker, CompositeRestoreWorker, TelemetrySink
from .ids import CheckpointId, JobId, SandboxId
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
from .telemetry import NoopTelemetrySink, start_operation

logger = logging.getLogger(__name__)

_CAPTURED_REQUEST_ID = "captured_request_id"


@dataclass
class _CheckpointQueueItem:
    job: CheckpointJob
    future: Future[CheckpointResult]
    queue_class: str
    request_key: tuple[SandboxId, str] | None = None


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
        self._restore_pool = ThreadPoolExecutor(max_workers=config.resolved_restore_workers)
        self._checkpoint_slots = BoundedSemaphore(
            config.resolved_checkpoint_workers + config.max_checkpoint_queue_size
        )
        self._lock = Lock()
        self._records: dict[JobId, JobRecord] = {}

        self._checkpoint_condition = Condition()
        self._checkpoint_shutdown = False
        self._checkpoint_queue: deque[_CheckpointQueueItem] = deque()
        self._checkpoint_urgent_queue: deque[_CheckpointQueueItem] = deque()
        self._checkpoint_items: dict[JobId, _CheckpointQueueItem] = {}
        self._checkpoint_request_items: dict[tuple[SandboxId, str], list[JobId]] = defaultdict(list)
        self._exposed_live_requests: set[tuple[SandboxId, str]] = set()
        self._urgent_dequeue_streak = 0
        self._checkpoint_threads = [
            Thread(
                target=self._checkpoint_worker_loop,
                name=f"agent-cr-ckpt-{index}",
                daemon=True,
            )
            for index in range(config.resolved_checkpoint_workers)
        ]
        for thread in self._checkpoint_threads:
            thread.start()

    @property
    def config(self) -> ExecutorConfig:
        return self._config

    def shutdown(self) -> None:
        logger.info("Shutting down executor worker pool")
        with self._checkpoint_condition:
            self._checkpoint_shutdown = True
            self._checkpoint_condition.notify_all()
        for thread in self._checkpoint_threads:
            thread.join()
        self._restore_pool.shutdown(wait=True)
        for worker in (self._checkpoint_worker, self._restore_worker):
            close = getattr(worker, "close", None)
            if callable(close):
                close()

    def run_checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        fut = self.submit_checkpoint(job)
        return fut.result()

    def run_restore(self, job: RestoreJob) -> RestoreResult:
        fut = self._submit_restore(job)
        return fut.result()

    def run_checkpoints(self, jobs: list[CheckpointJob]) -> list[CheckpointResult]:
        futures = [self.submit_checkpoint(job) for job in jobs]
        return [x.result() for x in futures]

    def run_restores(self, jobs: list[RestoreJob]) -> list[RestoreResult]:
        futures = [self._submit_restore(job) for job in jobs]
        return [x.result() for x in futures]

    def get_job_record(self, job_id: JobId) -> JobRecord | None:
        with self._lock:
            return self._records.get(job_id)

    def has_active_checkpoint(self, sandbox_id) -> bool:
        with self._lock:
            return any(
                record.job_type == JobType.CHECKPOINT
                and record.sandbox_id == sandbox_id
                and record.status in {JobStatus.PENDING, JobStatus.RUNNING}
                for record in self._records.values()
            )

    def has_active_job(self, sandbox_id) -> bool:
        with self._lock:
            return any(
                record.sandbox_id == sandbox_id
                and record.status in {JobStatus.PENDING, JobStatus.RUNNING}
                for record in self._records.values()
            )

    def submit_checkpoint(self, job: CheckpointJob) -> Future[CheckpointResult]:
        with self._checkpoint_condition:
            if self._checkpoint_shutdown:
                raise RuntimeError("cannot schedule new checkpoint jobs after shutdown")

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
            "executor.job_submitted",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.CHECKPOINT.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_scheduling_policy": self._config.checkpoint_scheduling_policy,
            },
        )
        logger.info("Queued checkpoint job %s for sandbox %s", job.job_id, job.sandbox_id)
        if not self._checkpoint_slots.acquire(blocking=False):
            logger.warning(
                "Checkpoint queue is full; rejecting job %s for sandbox %s",
                job.job_id,
                job.sandbox_id,
            )
            self._set_record(
                JobRecord(
                    job_id=job.job_id,
                    job_type=JobType.CHECKPOINT,
                    sandbox_id=job.sandbox_id,
                    checkpoint_id=None,
                    status=JobStatus.FAILED,
                    created_at=utc_now(),
                    finished_at=utc_now(),
                    failure_code=FailureCode.RUNTIME_ERROR,
                    message="checkpoint queue is full",
                )
            )
            raise RuntimeError("checkpoint queue is full")

        future: Future[CheckpointResult] = Future()
        future.add_done_callback(lambda _: self._checkpoint_slots.release())
        item = _CheckpointQueueItem(
            job=job,
            future=future,
            queue_class="fifo"
            if self._config.checkpoint_scheduling_policy == "fifo"
            else "normal",
            request_key=self._checkpoint_request_key(job),
        )
        try:
            with self._checkpoint_condition:
                if self._checkpoint_shutdown:
                    raise RuntimeError("cannot schedule new checkpoint jobs after shutdown")
                self._checkpoint_items[job.job_id] = item
                self._register_checkpoint_request_locked(item)
                if (
                    self._config.checkpoint_scheduling_policy == "reactive"
                    and item.request_key is not None
                    and item.request_key in self._exposed_live_requests
                ):
                    item.queue_class = "urgent"
                    self._checkpoint_urgent_queue.append(item)
                    self._exposed_live_requests.discard(item.request_key)
                else:
                    self._checkpoint_queue.append(item)
                self._checkpoint_condition.notify()
        except Exception:
            self._finish_checkpoint_item(item)
            self._checkpoint_slots.release()
            raise
        return future

    def notify_live_response_ready(
        self,
        sandbox_id: SandboxId,
        request_id: str,
        *,
        generation: int | None = None,
    ) -> bool:
        if self._config.checkpoint_scheduling_policy != "reactive":
            return False
        request_id = request_id.strip()
        if not request_id:
            return False
        request_key = (sandbox_id, request_id)
        promoted_item: _CheckpointQueueItem | None = None
        was_promoted = False
        with self._checkpoint_condition:
            promoted_item, was_promoted = self._promote_checkpoint_request_locked(request_key)
            if promoted_item is None:
                self._exposed_live_requests.add(request_key)
            self._checkpoint_condition.notify_all()
        if promoted_item is None:
            logger.debug(
                "Marked live request as exposed for reactive scheduling sandbox=%s request_id=%s generation=%s",
                sandbox_id,
                request_id,
                generation,
            )
            return False
        if was_promoted:
            logger.info(
                "Promoted checkpoint job %s to urgent queue sandbox=%s request_id=%s generation=%s",
                promoted_item.job.job_id,
                sandbox_id,
                request_id,
                generation,
            )
            self._telemetry.emit_event(
                "executor.job_promoted",
                {
                    "job_id": str(promoted_item.job.job_id),
                    "job_type": JobType.CHECKPOINT.value,
                    "sandbox_id": str(sandbox_id),
                    "request_id": request_id,
                    "request_generation": generation,
                },
            )
        return True

    def clear_live_response_ready(self, sandbox_id: SandboxId, request_id: str | None = None) -> None:
        if self._config.checkpoint_scheduling_policy != "reactive":
            return
        with self._checkpoint_condition:
            if request_id is None:
                to_clear = [key for key in self._exposed_live_requests if key[0] == sandbox_id]
                for key in to_clear:
                    self._exposed_live_requests.discard(key)
                return
            self._exposed_live_requests.discard((sandbox_id, request_id))

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
            "executor.job_submitted",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.RESTORE.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(job.checkpoint_id),
            },
        )
        logger.info(
            "Queued restore job %s for sandbox %s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
        )
        return self._restore_pool.submit(self._execute_restore, job)

    def _checkpoint_worker_loop(self) -> None:
        while True:
            item = self._dequeue_checkpoint_item()
            if item is None:
                return
            if not item.future.set_running_or_notify_cancel():
                self._finish_checkpoint_item(item)
                continue
            try:
                result = self._execute_checkpoint(item.job, queue_class=item.queue_class)
            except Exception as exc:
                self._finish_checkpoint_item(item)
                item.future.set_exception(exc)
            else:
                self._finish_checkpoint_item(item)
                item.future.set_result(result)

    def _dequeue_checkpoint_item(self) -> _CheckpointQueueItem | None:
        with self._checkpoint_condition:
            while True:
                item = self._take_next_checkpoint_item_locked()
                if item is not None:
                    self._unregister_checkpoint_request_locked(item)
                    return item
                if self._checkpoint_shutdown:
                    return None
                self._checkpoint_condition.wait()

    def _take_next_checkpoint_item_locked(self) -> _CheckpointQueueItem | None:
        if self._config.checkpoint_scheduling_policy == "fifo":
            if not self._checkpoint_queue:
                return None
            return self._checkpoint_queue.popleft()
        if self._checkpoint_urgent_queue:
            if (
                self._checkpoint_queue
                and self._urgent_dequeue_streak >= self._config.reactive_checkpoint_urgent_quota
            ):
                self._urgent_dequeue_streak = 0
                return self._checkpoint_queue.popleft()
            self._urgent_dequeue_streak += 1
            return self._checkpoint_urgent_queue.popleft()
        if not self._checkpoint_queue:
            return None
        self._urgent_dequeue_streak = 0
        return self._checkpoint_queue.popleft()

    def _checkpoint_request_key(self, job: CheckpointJob) -> tuple[SandboxId, str] | None:
        raw_request_id = job.metadata.get(_CAPTURED_REQUEST_ID)
        if raw_request_id is None:
            return None
        request_id = str(raw_request_id).strip()
        if not request_id:
            return None
        return (job.sandbox_id, request_id)

    def _register_checkpoint_request_locked(self, item: _CheckpointQueueItem) -> None:
        if item.request_key is None:
            return
        request_jobs = self._checkpoint_request_items[item.request_key]
        if item.job.job_id not in request_jobs:
            request_jobs.append(item.job.job_id)

    def _unregister_checkpoint_request_locked(self, item: _CheckpointQueueItem) -> None:
        if item.request_key is None:
            return
        request_jobs = self._checkpoint_request_items.get(item.request_key)
        if request_jobs is None:
            return
        self._checkpoint_request_items[item.request_key] = [
            job_id for job_id in request_jobs if job_id != item.job.job_id
        ]
        if not self._checkpoint_request_items[item.request_key]:
            self._checkpoint_request_items.pop(item.request_key, None)

    def _promote_checkpoint_request_locked(
        self,
        request_key: tuple[SandboxId, str],
    ) -> tuple[_CheckpointQueueItem | None, bool]:
        request_jobs = list(self._checkpoint_request_items.get(request_key, ()))
        for job_id in request_jobs:
            item = self._checkpoint_items.get(job_id)
            if item is None or item.future.done():
                continue
            if item.queue_class == "urgent":
                self._exposed_live_requests.discard(request_key)
                return item, False
            try:
                self._checkpoint_queue.remove(item)
            except ValueError:
                continue
            item.queue_class = "urgent"
            self._checkpoint_urgent_queue.append(item)
            self._exposed_live_requests.discard(request_key)
            return item, True
        return None, False

    def _finish_checkpoint_item(self, item: _CheckpointQueueItem) -> None:
        with self._checkpoint_condition:
            self._checkpoint_items.pop(item.job.job_id, None)
            self._unregister_checkpoint_request_locked(item)

    def _execute_checkpoint(self, job: CheckpointJob, *, queue_class: str) -> CheckpointResult:
        started = utc_now()
        queue_wait_ms = max(0.0, (started - job.requested_at).total_seconds() * 1000.0)
        self._telemetry.emit_event(
            "executor.job_dequeued",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.CHECKPOINT.value,
                "sandbox_id": str(job.sandbox_id),
                "queue_wait_ms": queue_wait_ms,
                "queue_class": queue_class,
                "checkpoint_scheduling_policy": self._config.checkpoint_scheduling_policy,
            },
        )
        self._telemetry.emit_metric(
            "executor.job_queue_wait_ms",
            queue_wait_ms,
            {
                "job_id": str(job.job_id),
                "job_type": JobType.CHECKPOINT.value,
                "sandbox_id": str(job.sandbox_id),
                "queue_class": queue_class,
            },
        )
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

        operation = start_operation(
            self._telemetry,
            "executor.job",
            {
                "component": "executor",
                "job_id": str(job.job_id),
                "job_type": JobType.CHECKPOINT.value,
                "sandbox_id": str(job.sandbox_id),
                "queue_class": queue_class,
            },
        )
        logger.info(
            "Starting checkpoint job %s for sandbox %s queue_class=%s",
            job.job_id,
            job.sandbox_id,
            queue_class,
        )

        last_result: CheckpointResult | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                logger.debug(
                    "Running checkpoint job %s attempt=%d",
                    job.job_id,
                    attempt + 1,
                )
                last_result = self._checkpoint_worker.checkpoint(job)
                if (
                    last_result.status == JobStatus.SUCCEEDED
                    or last_result.failure_code == FailureCode.VALIDATION_ERROR
                ):
                    break
            except Exception as exc:
                logger.exception(
                    "Checkpoint job %s failed with an unhandled exception on attempt %d",
                    job.job_id,
                    attempt + 1,
                )
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
            if last_result.status != JobStatus.SUCCEEDED and attempt < self._config.max_retries:
                logger.warning(
                    "Checkpoint job %s failed on attempt %d with status=%s; retrying",
                    job.job_id,
                    attempt + 1,
                    last_result.status.value,
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
        operation.finish(
            status=last_result.status.value,
            attributes={
                "checkpoint_id": str(last_result.checkpoint_id),
                "attempt_count": (
                    self._config.max_retries + 1
                    if last_result.status != JobStatus.SUCCEEDED
                    else attempt + 1
                ),
            },
        )
        log_fn = logger.info if (
            last_result.status == JobStatus.SUCCEEDED
            or last_result.failure_code == FailureCode.VALIDATION_ERROR
        ) else logger.error
        log_fn(
            "Finished checkpoint job %s for sandbox %s with status=%s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            last_result.status.value,
            last_result.checkpoint_id,
        )
        return last_result

    def _execute_restore(self, job: RestoreJob) -> RestoreResult:
        started = utc_now()
        queue_wait_ms = max(0.0, (started - job.requested_at).total_seconds() * 1000.0)
        self._telemetry.emit_event(
            "executor.job_dequeued",
            {
                "job_id": str(job.job_id),
                "job_type": JobType.RESTORE.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(job.checkpoint_id),
                "queue_wait_ms": queue_wait_ms,
            },
        )
        self._telemetry.emit_metric(
            "executor.job_queue_wait_ms",
            queue_wait_ms,
            {
                "job_id": str(job.job_id),
                "job_type": JobType.RESTORE.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(job.checkpoint_id),
            },
        )
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

        operation = start_operation(
            self._telemetry,
            "executor.job",
            {
                "component": "executor",
                "job_id": str(job.job_id),
                "job_type": JobType.RESTORE.value,
                "sandbox_id": str(job.sandbox_id),
                "checkpoint_id": str(job.checkpoint_id),
            },
        )
        logger.info(
            "Starting restore job %s for sandbox %s checkpoint=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
        )

        last_result: RestoreResult | None = None
        for attempt in range(self._config.max_retries + 1):
            try:
                logger.debug(
                    "Running restore job %s attempt=%d",
                    job.job_id,
                    attempt + 1,
                )
                last_result = self._restore_worker.restore(job)
                if last_result.status == JobStatus.SUCCEEDED:
                    break
            except Exception as exc:
                logger.exception(
                    "Restore job %s failed with an unhandled exception on attempt %d",
                    job.job_id,
                    attempt + 1,
                )
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
            if last_result.status != JobStatus.SUCCEEDED and attempt < self._config.max_retries:
                logger.warning(
                    "Restore job %s failed on attempt %d with status=%s; retrying",
                    job.job_id,
                    attempt + 1,
                    last_result.status.value,
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
        operation.finish(
            status=last_result.status.value,
            attributes={
                "attempt_count": (
                    self._config.max_retries + 1
                    if last_result.status != JobStatus.SUCCEEDED
                    else attempt + 1
                )
            },
        )
        log_fn = logger.info if last_result.status == JobStatus.SUCCEEDED else logger.error
        log_fn(
            "Finished restore job %s for sandbox %s checkpoint=%s with status=%s",
            job.job_id,
            job.sandbox_id,
            job.checkpoint_id,
            last_result.status.value,
        )
        return last_result

    def _set_record(self, record: JobRecord) -> None:
        with self._lock:
            self._records[record.job_id] = record
        logger.debug(
            "Updated job record job_id=%s type=%s status=%s",
            record.job_id,
            record.job_type.value,
            record.status.value,
        )

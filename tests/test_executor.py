from __future__ import annotations

import time
import unittest

from agent_cr import CRExecutor, ExecutorConfig
from agent_cr.ids import CheckpointId, JobId, SandboxId
from agent_cr.models import (
    CheckpointJob,
    CheckpointManifest,
    CheckpointResult,
    JobStatus,
    RestoreJob,
    RestoreResult,
    utc_now,
)


class SlowCheckpointWorker:
    def __init__(self, sleep_s: float = 0.05):
        self.sleep_s = sleep_s

    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        time.sleep(self.sleep_s)
        now = utc_now()
        ckpt_id = CheckpointId(f"ckpt-{job.job_id.value}")
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=ckpt_id,
            sandbox_id=job.sandbox_id,
            created_at=now,
            runtime_name="test",
            runtime_version="1",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        return CheckpointResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=ckpt_id,
            status=JobStatus.SUCCEEDED,
            started_at=now,
            finished_at=utc_now(),
            manifest=manifest,
        )


class FlakyCheckpointWorker:
    def __init__(self):
        self.attempts = 0

    def checkpoint(self, job: CheckpointJob) -> CheckpointResult:
        self.attempts += 1
        now = utc_now()
        ckpt_id = CheckpointId("ckpt-flaky")
        if self.attempts == 1:
            return CheckpointResult(
                job_id=job.job_id,
                sandbox_id=job.sandbox_id,
                checkpoint_id=ckpt_id,
                status=JobStatus.FAILED,
                started_at=now,
                finished_at=utc_now(),
                manifest=None,
                message="first attempt failed",
            )
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=ckpt_id,
            sandbox_id=job.sandbox_id,
            created_at=now,
            runtime_name="test",
            runtime_version="1",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        return CheckpointResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=ckpt_id,
            status=JobStatus.SUCCEEDED,
            started_at=now,
            finished_at=utc_now(),
            manifest=manifest,
        )


class PassThroughRestoreWorker:
    def restore(self, job: RestoreJob) -> RestoreResult:
        now = utc_now()
        return RestoreResult(
            job_id=job.job_id,
            sandbox_id=job.sandbox_id,
            checkpoint_id=job.checkpoint_id,
            status=JobStatus.SUCCEEDED,
            started_at=now,
            finished_at=utc_now(),
        )


class ExecutorTests(unittest.TestCase):
    def test_executor_parallel_checkpoint_batches(self) -> None:
        executor = CRExecutor(
            ExecutorConfig(max_workers=3, max_retries=0),
            checkpoint_worker=SlowCheckpointWorker(sleep_s=0.05),
            restore_worker=PassThroughRestoreWorker(),
        )
        jobs = [
            CheckpointJob(
                job_id=JobId.new(prefix="job"),
                sandbox_id=SandboxId("sbx-1"),
                requested_at=utc_now(),
            )
            for _ in range(6)
        ]

        t0 = time.perf_counter()
        results = executor.run_checkpoints(jobs)
        elapsed = time.perf_counter() - t0
        executor.shutdown()

        self.assertEqual(len(results), 6)
        self.assertTrue(all(r.status == JobStatus.SUCCEEDED for r in results))
        self.assertLess(elapsed, 0.25)

    def test_executor_retry_on_failed_checkpoint_result(self) -> None:
        worker = FlakyCheckpointWorker()
        executor = CRExecutor(
            ExecutorConfig(max_workers=1, max_retries=1, retry_backoff_seconds=0.001),
            checkpoint_worker=worker,
            restore_worker=PassThroughRestoreWorker(),
        )
        job = CheckpointJob(
            job_id=JobId("job-1"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
        )

        result = executor.run_checkpoint(job)
        record = executor.get_job_record(job.job_id)
        executor.shutdown()

        self.assertEqual(result.status, JobStatus.SUCCEEDED)
        self.assertEqual(worker.attempts, 2)
        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record.status, JobStatus.SUCCEEDED)


if __name__ == "__main__":
    unittest.main()

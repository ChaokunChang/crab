from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_cr import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CheckpointId,
    CheckpointJob,
    CheckpointManager,
    DockerRuntimeAdapter,
    JobId,
    LocalCheckpointManager,
    RestoreJob,
    RuncRuntimeAdapter,
    SandboxId,
    StorageConfig,
)
from agent_cr.contracts import SandboxRuntimeAdapter
from agent_cr.models import CheckpointManifest, utc_now


class ContractTests(unittest.TestCase):
    def test_runtime_adapters_are_contract_compatible(self) -> None:
        for adapter in (DockerRuntimeAdapter(), RuncRuntimeAdapter()):
            self.assertIsInstance(adapter, SandboxRuntimeAdapter)
            dry = adapter.plan_process_checkpoint(SandboxId("sbx-1"), CheckpointId("ckpt-1"))
            self.assertFalse(dry.executed)
            self.assertTrue(len(dry.planned_command) > 0)

    def test_workers_return_typed_dry_run_results(self) -> None:
        adapter = DockerRuntimeAdapter()
        process_c = AdapterProcessCWorker(adapter)
        process_r = AdapterProcessRWorker(adapter)
        fs_c = AdapterFileSystemCWorker(adapter)
        fs_r = AdapterFileSystemRWorker(adapter)

        cjob = CheckpointJob(
            job_id=JobId("job-1"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
        )
        ckpt_id = CheckpointId("ckpt-1")

        c_process_result = process_c.checkpoint(cjob, ckpt_id)
        c_fs_result = fs_c.checkpoint(cjob, ckpt_id)
        self.assertTrue(c_process_result.success)
        self.assertTrue(c_fs_result.success)
        self.assertFalse(c_process_result.dry_run_status.executed)
        self.assertFalse(c_fs_result.dry_run_status.executed)

        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=ckpt_id,
            sandbox_id=cjob.sandbox_id,
            created_at=utc_now(),
            runtime_name="docker",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        rjob = RestoreJob(
            job_id=JobId("job-2"),
            sandbox_id=SandboxId("sbx-1"),
            checkpoint_id=ckpt_id,
            requested_at=utc_now(),
        )
        r_process_result = process_r.restore(rjob, manifest)
        r_fs_result = fs_r.restore(rjob, manifest)
        self.assertTrue(r_process_result.success)
        self.assertTrue(r_fs_result.success)
        self.assertFalse(r_process_result.dry_run_status.executed)
        self.assertFalse(r_fs_result.dry_run_status.executed)

    def test_local_storage_implements_checkpoint_manager_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_contract_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            self.assertIsInstance(mgr, CheckpointManager)


if __name__ == "__main__":
    unittest.main()

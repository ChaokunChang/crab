from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_cr import JobId, PolicyConfig, RestoreJob, SandboxId, SandboxSnapshot, StorageConfig, build_default_system
from agent_cr.models import JobStatus, utc_now


class SimulatedE2ETests(unittest.TestCase):
    def test_checkpoint_restore_flow_with_scheduler_and_telemetry(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_e2e_") as tmp:
            system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
                policy_config=PolicyConfig(
                    min_checkpoint_interval_seconds=0.0,
                    force_checkpoint_after_seconds=0.0,
                    require_change_signal=True,
                ),
            )

            sandbox_id = system.sandbox_manager.launch("docker")
            system.inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="docker",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )

            ckpt_result = system.checkpoint_if_due(sandbox_id)
            self.assertIsNotNone(ckpt_result)
            assert ckpt_result is not None
            self.assertEqual(ckpt_result.status, JobStatus.SUCCEEDED)
            self.assertIsNotNone(ckpt_result.manifest)

            restore_job = RestoreJob(
                job_id=JobId.new(prefix="restore"),
                sandbox_id=sandbox_id,
                checkpoint_id=ckpt_result.checkpoint_id,
                requested_at=utc_now(),
            )
            restore_result = system.executor.run_restore(restore_job)
            self.assertEqual(restore_result.status, JobStatus.SUCCEEDED)

            loaded_manifest = system.storage.get_manifest(sandbox_id, ckpt_result.checkpoint_id)
            self.assertEqual(str(loaded_manifest.checkpoint_id), str(ckpt_result.checkpoint_id))

            event_names = [x[0] for x in system.telemetry.events]
            self.assertIn("scheduler.evaluate", event_names)
            self.assertIn("executor.job_finished", event_names)
            system.executor.shutdown()


if __name__ == "__main__":
    unittest.main()

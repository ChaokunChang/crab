from __future__ import annotations

import json
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
    DefaultRWorker,
    DockerRuntimeAdapter,
    EBPFSandboxInspector,
    EBPFEvent,
    EBPFEventKind,
    InMemoryEBPFEventCollector,
    JobId,
    LocalCheckpointManager,
    RestoreJob,
    RuncRuntimeAdapter,
    RuncRuntimePaths,
    SandboxId,
    SandboxSnapshot,
    StorageConfig,
)
from agent_cr.contracts import SandboxRuntimeAdapter
from agent_cr.models import CheckpointManifest, RuntimeOperationStatus, WorkerStepResult, utc_now
from agent_cr.runtime import CommandRunner


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None):
        _ = cwd
        self.commands.append(tuple(command))
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class RecordingRestoreWorker:
    def __init__(self) -> None:
        self.jobs = []

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        _ = manifest
        self.jobs.append(job)
        return WorkerStepResult(
            success=True,
            operation_status=RuntimeOperationStatus(executed=False, reason="recorded"),
        )


class NoArtifactCheckpointManager:
    def __init__(self, manifest: CheckpointManifest) -> None:
        self.manifest = manifest

    def get_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> CheckpointManifest:
        _ = (sandbox_id, checkpoint_id)
        return self.manifest

    def get_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, reference) -> bytes:
        _ = (sandbox_id, checkpoint_id, reference)
        raise AssertionError("process restore should not fetch process artifacts from checkpoint storage")


class ContractTests(unittest.TestCase):
    def test_runtime_adapters_are_contract_compatible(self) -> None:
        docker = DockerRuntimeAdapter()
        self.assertIsInstance(docker, SandboxRuntimeAdapter)
        self.assertTrue(len(docker.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1")).command) > 0)

        with tempfile.TemporaryDirectory(prefix="agent_cr_runtime_contract_") as tmp:
            adapter = RuncRuntimeAdapter(
                command_runner=FakeCommandRunner(),
                paths=RuncRuntimePaths(
                    state_root=Path(tmp) / "state",
                    bundle_root=Path(tmp) / "bundles",
                    checkpoint_root=Path(tmp) / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            self.assertIsInstance(adapter, SandboxRuntimeAdapter)
            self.assertTrue(
                len(adapter.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1")).command) > 0
            )

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
        self.assertFalse(c_process_result.operation_status.executed)
        self.assertFalse(c_fs_result.operation_status.executed)

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
        self.assertFalse(r_process_result.operation_status.executed)
        self.assertFalse(r_fs_result.operation_status.executed)

    def test_runc_process_checkpoint_emits_only_metadata_artifact(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_process_contract_") as tmp:
            base = Path(tmp)
            adapter = RuncRuntimeAdapter(
                command_runner=FakeCommandRunner(),
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            process_c = AdapterProcessCWorker(adapter)
            sandbox_id = SandboxId("sbx-1")
            checkpoint_id = CheckpointId("ckpt-1")

            result = process_c.checkpoint(
                CheckpointJob(
                    job_id=JobId("job-1"),
                    sandbox_id=sandbox_id,
                    requested_at=utc_now(),
                ),
                checkpoint_id,
            )

            self.assertTrue(result.success)
            self.assertEqual(len(result.artifacts), 1)
            self.assertEqual(result.artifacts[0].name, "process_checkpoint.json")
            payload = json.loads(result.artifacts[0].data.decode("utf-8"))
            self.assertEqual(payload["process_storage_mode"], "runtime_reference")
            self.assertEqual(
                payload["process_checkpoint_location"],
                str(base / "checkpoints" / str(sandbox_id) / str(checkpoint_id) / "process"),
            )

    def test_runc_process_restore_requires_existing_checkpoint_directory(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_process_restore_") as tmp:
            base = Path(tmp)
            adapter = RuncRuntimeAdapter(
                command_runner=FakeCommandRunner(),
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )
            process_r = AdapterProcessRWorker(adapter)
            sandbox_id = SandboxId("sbx-1")
            checkpoint_id = CheckpointId("ckpt-1")
            manifest = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=checkpoint_id,
                sandbox_id=sandbox_id,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            job = RestoreJob(
                job_id=JobId("job-2"),
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                requested_at=utc_now(),
            )

            with self.assertRaisesRegex(FileNotFoundError, "process checkpoint directory not found"):
                process_r.restore(job, manifest)

            checkpoint_dir = base / "checkpoints" / str(sandbox_id) / str(checkpoint_id) / "process"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            result = process_r.restore(job, manifest)
            self.assertTrue(result.success)
            self.assertTrue(result.operation_status.executed)

    def test_default_restore_worker_skips_process_artifact_downloads(self) -> None:
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=SandboxId("sbx-1"),
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        fs_worker = RecordingRestoreWorker()
        process_worker = RecordingRestoreWorker()
        restore_worker = DefaultRWorker(
            process_worker=process_worker,
            filesystem_worker=fs_worker,
            checkpoint_manager=NoArtifactCheckpointManager(manifest),
        )
        job = RestoreJob(
            job_id=JobId("job-restore"),
            sandbox_id=SandboxId("sbx-1"),
            checkpoint_id=CheckpointId("ckpt-1"),
            requested_at=utc_now(),
            metadata={"keep": "me"},
        )

        result = restore_worker.restore(job)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(fs_worker.jobs, [job])
        self.assertEqual(process_worker.jobs, [job])

    def test_runc_runtime_executes_real_commands_via_runner(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_runc_runtime_") as tmp:
            runner = FakeCommandRunner()
            base = Path(tmp)
            adapter = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            process_status = adapter.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"))
            fs_status = adapter.checkpoint_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

            self.assertTrue(process_status.executed)
            self.assertTrue(fs_status.executed)
            self.assertEqual(runner.commands[0][0:3], ("runc", "--root", str(base / "state")))
            self.assertEqual(runner.commands[1], ("zfs", "snapshot", "pool/agent-cr/sbx-1@ckpt-1"))

    def test_ebpf_inspector_uses_recorded_events(self) -> None:
        collector = InMemoryEBPFEventCollector()
        inspector = EBPFSandboxInspector(collector)
        sandbox_id = SandboxId("sbx-1")
        inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )
        collector.record(
            EBPFEvent(
                sandbox_id=sandbox_id,
                kind=EBPFEventKind.FILE_WRITE,
                observed_at=utc_now(),
                metadata={"path": "/tmp/x"},
            )
        )

        snapshot = inspector.inspect(sandbox_id)
        self.assertFalse(snapshot.process_changed)
        self.assertTrue(snapshot.filesystem_changed)
        self.assertEqual(snapshot.metadata["ebpf_event_count"], 1)

    def test_local_storage_implements_checkpoint_manager_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_contract_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            self.assertIsInstance(mgr, CheckpointManager)


if __name__ == "__main__":
    unittest.main()

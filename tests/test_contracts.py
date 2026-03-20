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
    ArtifactKind,
    ArtifactPayload,
    ArtifactReference,
    CheckpointId,
    CheckpointJob,
    CheckpointManager,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    EBPFEvent,
    EBPFEventKind,
    FailureCode,
    InMemoryRuntime,
    InMemoryEBPFEventCollector,
    JobId,
    LocalCheckpointManager,
    RestoreJob,
    RuncCheckpointOptions,
    RuncRuntime,
    RuncRuntimeOptions,
    RuncRuntimePaths,
    RuncRestoreOptions,
    Runtime,
    SandboxId,
    SandboxSnapshot,
    StorageConfig,
)
from agent_cr.models import CheckpointManifest, RuntimeOperationStatus, WorkerStepResult, utc_now
from agent_cr.runtime import CommandRunner

DockerRuntimeAdapter = InMemoryRuntime
RuncRuntimeAdapter = RuncRuntime
SandboxRuntimeAdapter = Runtime


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
        self.manifests: list[CheckpointManifest] = []

    def restore(self, job: RestoreJob, manifest: CheckpointManifest) -> WorkerStepResult:
        self.jobs.append(job)
        self.manifests.append(manifest)
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

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        _ = sandbox_id
        return [self.manifest.checkpoint_id]

    def delete_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        _ = (sandbox_id, checkpoint_id)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        _ = sandbox_id

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        _ = manifest

    def handle_restore_complete(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        _ = (sandbox_id, checkpoint_id)


class ManifestCheckpointManager:
    def __init__(self, manifests: list[CheckpointManifest]) -> None:
        self._manifests = {(manifest.sandbox_id, manifest.checkpoint_id): manifest for manifest in manifests}
        self._ordered: dict[SandboxId, list[CheckpointId]] = {}
        for manifest in manifests:
            self._ordered.setdefault(manifest.sandbox_id, []).append(manifest.checkpoint_id)

    def get_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> CheckpointManifest:
        return self._manifests[(sandbox_id, checkpoint_id)]

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        return list(self._ordered.get(sandbox_id, []))

    def put_manifest(self, manifest: CheckpointManifest) -> None:
        self._manifests[(manifest.sandbox_id, manifest.checkpoint_id)] = manifest

    def put_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, artifact):
        raise NotImplementedError

    def get_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, reference) -> bytes:
        raise NotImplementedError

    def delete_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        self._manifests.pop((sandbox_id, checkpoint_id), None)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        for checkpoint_id in list(self._ordered.get(sandbox_id, [])):
            self.delete_checkpoint(sandbox_id, checkpoint_id)

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        _ = manifest

    def handle_restore_complete(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        _ = (sandbox_id, checkpoint_id)


class RecordingCheckpointWorker:
    def __init__(self, artifact_kind: str) -> None:
        self.calls: list[tuple[CheckpointJob, CheckpointId]] = []
        self.artifact_kind = artifact_kind

    def checkpoint(self, job: CheckpointJob, checkpoint_id: CheckpointId) -> WorkerStepResult:
        self.calls.append((job, checkpoint_id))
        return WorkerStepResult(
            success=True,
            artifacts=[],
            operation_status=RuntimeOperationStatus(executed=False, reason=self.artifact_kind),
        )


class RecordingCheckpointManager:
    def __init__(self, existing_manifests: list[CheckpointManifest] | None = None) -> None:
        self.manifest: CheckpointManifest | None = None
        self.completed: list[CheckpointManifest] = []
        self._manifests: dict[tuple[SandboxId, CheckpointId], CheckpointManifest] = {}
        self._ordered: dict[SandboxId, list[CheckpointId]] = {}
        for manifest in existing_manifests or []:
            self._manifests[(manifest.sandbox_id, manifest.checkpoint_id)] = manifest
            self._ordered.setdefault(manifest.sandbox_id, []).append(manifest.checkpoint_id)

    def put_manifest(self, manifest: CheckpointManifest) -> None:
        self.manifest = manifest
        self._manifests[(manifest.sandbox_id, manifest.checkpoint_id)] = manifest
        self._ordered.setdefault(manifest.sandbox_id, []).append(manifest.checkpoint_id)

    def get_manifest(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> CheckpointManifest:
        manifest = self._manifests.get((sandbox_id, checkpoint_id))
        if manifest is not None:
            return manifest
        assert self.manifest is not None
        return self.manifest

    def put_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, artifact):
        _ = artifact
        return type(
            "ArtifactReference",
            (),
            {
                "kind": None,
                "name": "noop",
            },
        )()

    def get_artifact(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId, reference) -> bytes:
        _ = (sandbox_id, checkpoint_id, reference)
        return b""

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        return list(self._ordered.get(sandbox_id, []))

    def delete_checkpoint(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        _ = (sandbox_id, checkpoint_id)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        _ = sandbox_id

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        self.completed.append(manifest)

    def handle_restore_complete(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> None:
        _ = (sandbox_id, checkpoint_id)


class ContractTests(unittest.TestCase):
    def test_runtime_adapters_are_contract_compatible(self) -> None:
        docker = DockerRuntimeAdapter()
        self.assertIsInstance(docker, SandboxRuntimeAdapter)
        self.assertTrue(
            len(docker.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"), leave_running=False).command)
            > 0
        )

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
                len(
                    adapter.checkpoint_process(
                        SandboxId("sbx-1"),
                        CheckpointId("ckpt-1"),
                        leave_running=False,
                    ).command
                )
                > 0
            )

    def test_runc_runtime_uses_default_optional_checkpoint_and_restore_args(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_runtime_contract_") as tmp:
            base = Path(tmp)
            runner = FakeCommandRunner()
            adapter = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            adapter.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"), leave_running=False)
            adapter.restore_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

            self.assertEqual(
                runner.commands[0],
                (
                    "runc",
                    "--root",
                    str(base / "state"),
                    "checkpoint",
                    "--image-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"),
                    "--work-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "work"),
                    "--leave-running=false",
                    "--tcp-established",
                    "--shell-job",
                    "--tcp-skip-in-flight",
                    "sbx-1",
                ),
            )
            self.assertEqual(
                runner.commands[1],
                (
                    "runc",
                    "--root",
                    str(base / "state"),
                    "restore",
                    "-d",
                    "--bundle",
                    str(base / "bundles" / "sbx-1"),
                    "--image-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"),
                    "--work-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "work"),
                    "--tcp-established",
                    "--shell-job",
                    "sbx-1",
                ),
            )

    def test_runc_runtime_options_allow_overriding_optional_args(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_runtime_contract_") as tmp:
            base = Path(tmp)
            runner = FakeCommandRunner()
            adapter = RuncRuntimeAdapter(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=base / "state",
                    bundle_root=base / "bundles",
                    checkpoint_root=base / "checkpoints",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
                options=RuncRuntimeOptions(
                    checkpoint=RuncCheckpointOptions(
                        shell_job=False,
                        tcp_skip_in_flight=False,
                        extra_args=("--manage-cgroups-mode=soft",),
                    ),
                    restore=RuncRestoreOptions(
                        detach=False,
                        tcp_established=False,
                        extra_args=("--manage-cgroups-mode=soft",),
                    ),
                ),
            )

            adapter.checkpoint_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"), leave_running=True)
            adapter.restore_process(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

            self.assertEqual(
                runner.commands[0],
                (
                    "runc",
                    "--root",
                    str(base / "state"),
                    "checkpoint",
                    "--image-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"),
                    "--work-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "work"),
                    "--leave-running=true",
                    "--tcp-established",
                    "--manage-cgroups-mode=soft",
                    "sbx-1",
                ),
            )
            self.assertEqual(
                runner.commands[1],
                (
                    "runc",
                    "--root",
                    str(base / "state"),
                    "restore",
                    "--bundle",
                    str(base / "bundles" / "sbx-1"),
                    "--image-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "process"),
                    "--work-path",
                    str(base / "checkpoints" / "sbx-1" / "ckpt-1" / "work"),
                    "--shell-job",
                    "--manage-cgroups-mode=soft",
                    "sbx-1",
                ),
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
        self.assertEqual(fs_worker.jobs, [])
        self.assertEqual(process_worker.jobs, [])

    def test_default_checkpoint_worker_honors_scoped_checkpoint_flags(self) -> None:
        manager = RecordingCheckpointManager()
        process_worker = RecordingCheckpointWorker("process")
        filesystem_worker = RecordingCheckpointWorker("filesystem")
        worker = DefaultCWorker(
            process_worker=process_worker,
            filesystem_worker=filesystem_worker,
            checkpoint_manager=manager,
            runtime=DockerRuntimeAdapter(),
        )
        job = CheckpointJob(
            job_id=JobId("job-scoped"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
            checkpoint_process=True,
            checkpoint_filesystem=False,
        )

        result = worker.checkpoint(job)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(process_worker.calls), 1)
        self.assertEqual(len(filesystem_worker.calls), 0)
        self.assertEqual(len(manager.completed), 1)
        assert result.manifest is not None
        self.assertEqual(result.manifest.filesystem_artifacts, [])

    def test_default_checkpoint_worker_rejects_guarded_job_before_workers_run(self) -> None:
        manager = RecordingCheckpointManager()
        process_worker = RecordingCheckpointWorker("process")
        filesystem_worker = RecordingCheckpointWorker("filesystem")
        worker = DefaultCWorker(
            process_worker=process_worker,
            filesystem_worker=filesystem_worker,
            checkpoint_manager=manager,
            runtime=DockerRuntimeAdapter(),
            checkpoint_guard=lambda job: (False, f"{job.sandbox_id}:sandbox_not_running"),
        )
        job = CheckpointJob(
            job_id=JobId("job-guarded"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
            checkpoint_process=True,
            checkpoint_filesystem=True,
        )

        result = worker.checkpoint(job)

        self.assertEqual(result.status.value, "failed")
        self.assertEqual(result.failure_code, FailureCode.VALIDATION_ERROR)
        self.assertEqual(result.message, "sbx-1:sandbox_not_running")
        self.assertEqual(process_worker.calls, [])
        self.assertEqual(filesystem_worker.calls, [])
        self.assertEqual(manager.completed, [])

    def test_default_checkpoint_worker_promotes_first_filesystem_only_checkpoint(self) -> None:
        manager = RecordingCheckpointManager()
        process_worker = RecordingCheckpointWorker("process")
        filesystem_worker = RecordingCheckpointWorker("filesystem")
        worker = DefaultCWorker(
            process_worker=process_worker,
            filesystem_worker=filesystem_worker,
            checkpoint_manager=manager,
            runtime=DockerRuntimeAdapter(),
        )
        job = CheckpointJob(
            job_id=JobId("job-fs-first"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=utc_now(),
            checkpoint_process=False,
            checkpoint_filesystem=True,
        )

        result = worker.checkpoint(job)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(process_worker.calls), 1)
        self.assertEqual(len(filesystem_worker.calls), 1)
        assert result.manifest is not None
        self.assertTrue(result.manifest.metadata["promoted_process_checkpoint"])
        self.assertEqual(result.manifest.metadata["promoted_process_checkpoint_reason"], "missing_process_ancestor")

    def test_default_checkpoint_worker_keeps_filesystem_only_scope_with_process_ancestor(self) -> None:
        sid = SandboxId("sbx-1")
        prior_process = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[ArtifactReference(kind=ArtifactKind.PROCESS, name="process.json", relative_path="p", size_bytes=1, sha256="0" * 64, metadata={})],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        manager = RecordingCheckpointManager(existing_manifests=[prior_process])
        process_worker = RecordingCheckpointWorker("process")
        filesystem_worker = RecordingCheckpointWorker("filesystem")
        worker = DefaultCWorker(
            process_worker=process_worker,
            filesystem_worker=filesystem_worker,
            checkpoint_manager=manager,
            runtime=DockerRuntimeAdapter(),
        )
        job = CheckpointJob(
            job_id=JobId("job-fs-next"),
            sandbox_id=sid,
            requested_at=utc_now(),
            checkpoint_process=False,
            checkpoint_filesystem=True,
        )

        result = worker.checkpoint(job)

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(process_worker.calls), 0)
        self.assertEqual(len(filesystem_worker.calls), 1)
        assert result.manifest is not None
        self.assertNotIn("promoted_process_checkpoint", result.manifest.metadata)

    def test_default_restore_worker_backfills_missing_process_from_previous_checkpoint(self) -> None:
        sid = SandboxId("sbx-1")
        process_ref = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp()))).put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap"),
            ArtifactPayload(kind=ArtifactKind.PROCESS, name="process.json", data=b"{}"),
        )
        previous = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[process_ref],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        current = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
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
            checkpoint_manager=ManifestCheckpointManager([previous, current]),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-process"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-2"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(process_worker.jobs), 1)
        self.assertEqual(len(fs_worker.jobs), 0)
        self.assertEqual(process_worker.manifests[0].metadata["process_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(len(process_worker.manifests[0].process_artifacts), 1)

    def test_default_restore_worker_backfills_missing_filesystem_from_previous_checkpoint(self) -> None:
        sid = SandboxId("sbx-1")
        fs_ref = LocalCheckpointManager(StorageConfig(root_dir=Path(tempfile.mkdtemp()))).put_artifact(
            sid,
            CheckpointId("ckpt-bootstrap"),
            ArtifactPayload(kind=ArtifactKind.FILESYSTEM, name="filesystem.json", data=b"{}"),
        )
        previous = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=sid,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[fs_ref],
            metadata={},
        ).with_integrity()
        current = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=sid,
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
            checkpoint_manager=ManifestCheckpointManager([previous, current]),
        )

        result = restore_worker.restore(
            RestoreJob(
                job_id=JobId("job-restore-fs"),
                sandbox_id=sid,
                checkpoint_id=CheckpointId("ckpt-2"),
                requested_at=utc_now(),
            )
        )

        self.assertEqual(result.status.value, "succeeded")
        self.assertEqual(len(fs_worker.jobs), 1)
        self.assertEqual(len(process_worker.jobs), 0)
        self.assertEqual(fs_worker.manifests[0].metadata["filesystem_restore_checkpoint_id"], "ckpt-1")
        self.assertEqual(len(fs_worker.manifests[0].filesystem_artifacts), 1)

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

            process_status = adapter.checkpoint_process(
                SandboxId("sbx-1"),
                CheckpointId("ckpt-1"),
                leave_running=True,
            )
            fs_status = adapter.checkpoint_filesystem(SandboxId("sbx-1"), CheckpointId("ckpt-1"))

            self.assertTrue(process_status.executed)
            self.assertTrue(fs_status.executed)
            self.assertEqual(runner.commands[0][0:3], ("runc", "--root", str(base / "state")))
            self.assertIn("--leave-running=true", runner.commands[0])
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

    def test_ebpf_inspector_clears_only_checkpointed_dimension(self) -> None:
        collector = InMemoryEBPFEventCollector()
        inspector = EBPFSandboxInspector(collector)
        sandbox_id = SandboxId("sbx-1")
        base_time = utc_now()
        inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=True,
                observed_at=base_time,
            )
        )

        snapshot = inspector.inspect(sandbox_id)
        self.assertTrue(snapshot.process_changed)
        self.assertTrue(snapshot.filesystem_changed)

        checkpoint_time = utc_now()
        inspector.mark_checkpoint_complete(
            sandbox_id,
            process=True,
            filesystem=False,
            at=checkpoint_time,
        )

        updated = inspector.inspect(sandbox_id)
        self.assertFalse(updated.process_changed)
        self.assertTrue(updated.filesystem_changed)

    def test_local_storage_implements_checkpoint_manager_contract(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_contract_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            self.assertIsInstance(mgr, CheckpointManager)


if __name__ == "__main__":
    unittest.main()

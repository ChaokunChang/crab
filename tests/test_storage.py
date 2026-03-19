from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from agent_cr import (
    ArtifactKind,
    ArtifactPayload,
    ArtifactReference,
    CheckpointId,
    CheckpointManifest,
    DeleteAfterRestoreCheckpointManager,
    KeepAllCheckpointManager,
    LatestOnlyCheckpointManager,
    LocalCheckpointManager,
    SandboxId,
    StorageConfig,
)
from agent_cr.models import utc_now
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


class StorageTests(unittest.TestCase):
    def _reference(self, kind: ArtifactKind, name: str) -> ArtifactReference:
        return ArtifactReference(
            kind=kind,
            name=name,
            relative_path=f"artifacts/sbx/{name}",
            size_bytes=1,
            sha256="0" * 64,
            metadata={},
        )

    def test_local_checkpoint_manager_artifacts_and_manifests(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            sid = SandboxId("sbx-1")
            ckpt = CheckpointId("ckpt-1")

            artifact_ref = mgr.put_artifact(
                sid,
                ckpt,
                ArtifactPayload(
                    kind=ArtifactKind.PROCESS,
                    name="proc.bin",
                    data=b"hello",
                    metadata={"m": 1},
                ),
            )
            loaded = mgr.get_artifact(sid, ckpt, artifact_ref)
            self.assertEqual(loaded, b"hello")

            manifest = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=ckpt,
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="docker",
                runtime_version="stub",
                process_artifacts=[artifact_ref],
                filesystem_artifacts=[],
                metadata={"x": "y"},
            ).with_integrity()
            mgr.put_manifest(manifest)

            loaded_manifest = mgr.get_manifest(sid, ckpt)
            self.assertEqual(loaded_manifest.runtime_name, "docker")
            self.assertEqual(len(loaded_manifest.process_artifacts), 1)

    def test_list_checkpoints_sorted_by_created_at(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            mgr = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            sid = SandboxId("sbx-1")

            c1 = CheckpointId("ckpt-1")
            c2 = CheckpointId("ckpt-2")
            base = utc_now()
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=c1,
                sandbox_id=sid,
                created_at=base,
                runtime_name="docker",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=c2,
                sandbox_id=sid,
                created_at=base + timedelta(seconds=1),
                runtime_name="docker",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            mgr.put_manifest(m1)
            mgr.put_manifest(m2)

            found = mgr.list_checkpoints(sid)
            self.assertEqual(found, [c1, c2])

    def test_delete_checkpoint_removes_manifest_artifacts_and_runtime_refs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            mgr = LocalCheckpointManager(StorageConfig(root_dir=root), command_runner=runner)
            sid = SandboxId("sbx-1")
            ckpt = CheckpointId("ckpt-1")
            process_dir = root / "runtime" / str(sid) / str(ckpt) / "process"
            process_dir.mkdir(parents=True, exist_ok=True)
            (process_dir / "dump.img").write_text("data")

            process_ref = mgr.put_artifact(
                sid,
                ckpt,
                ArtifactPayload(
                    kind=ArtifactKind.PROCESS,
                    name="process_checkpoint.json",
                    data=(
                        '{"process_checkpoint_location": "%s", "status": {"metadata": {"image_path": "%s"}}}'
                        % (process_dir, process_dir)
                    ).encode("utf-8"),
                ),
            )
            fs_ref = mgr.put_artifact(
                sid,
                ckpt,
                ArtifactPayload(
                    kind=ArtifactKind.FILESYSTEM,
                    name="filesystem_checkpoint.json",
                    data=b'{"filesystem": {"snapshot": "pool/agent-cr/sbx-1@ckpt-1"}}',
                ),
            )
            mgr.put_manifest(
                CheckpointManifest(
                    schema_version="v1",
                    checkpoint_id=ckpt,
                    sandbox_id=sid,
                    created_at=utc_now(),
                    runtime_name="runc",
                    runtime_version=None,
                    process_artifacts=[process_ref],
                    filesystem_artifacts=[fs_ref],
                    metadata={},
                ).with_integrity()
            )

            mgr.delete_checkpoint(sid, ckpt)

            self.assertFalse((root / "manifests" / str(sid) / f"{ckpt}.json").exists())
            self.assertFalse((root / "artifacts" / str(sid) / str(ckpt)).exists())
            self.assertFalse(process_dir.parent.exists())
            self.assertEqual(runner.commands, [("zfs", "destroy", "pool/agent-cr/sbx-1@ckpt-1")])

    def test_latest_only_manager_prunes_older_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            for manifest in (m1, m2):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-2")])

    def test_latest_only_manager_keeps_latest_filesystem_ancestor_for_process_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p1")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f1")],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p2")],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            for manifest in (m1, m2):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-1"), CheckpointId("ckpt-2")])

    def test_latest_only_manager_keeps_latest_process_ancestor_for_filesystem_only_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[self._reference(ArtifactKind.PROCESS, "p1")],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f1")],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[self._reference(ArtifactKind.FILESYSTEM, "f2")],
                metadata={},
            ).with_integrity()
            for manifest in (m1, m2):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-1"), CheckpointId("ckpt-2")])

    def test_latest_only_manager_keeps_pinned_checkpoint_during_prune(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = LatestOnlyCheckpointManager(base)
            m1 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m2 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=1),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            m3 = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-3"),
                sandbox_id=sid,
                created_at=utc_now() + timedelta(seconds=2),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            for manifest in (m1, m2):
                mgr.put_manifest(manifest)
                mgr.handle_checkpoint_complete(manifest)

            self.assertTrue(mgr.pin_checkpoint(sid, CheckpointId("ckpt-2")))
            mgr.put_manifest(m3)
            mgr.handle_checkpoint_complete(m3)
            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-2"), CheckpointId("ckpt-3")])

            mgr.unpin_checkpoint(sid, CheckpointId("ckpt-2"))
            mgr.handle_checkpoint_complete(m3)
            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-3")])

    def test_delete_after_restore_manager_removes_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            sid = SandboxId("sbx-1")
            ckpt = CheckpointId("ckpt-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = DeleteAfterRestoreCheckpointManager(base)
            manifest = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=ckpt,
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            mgr.put_manifest(manifest)

            mgr.handle_restore_complete(sid, ckpt)

            self.assertEqual(mgr.list_checkpoints(sid), [])

    def test_keep_all_manager_retains_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_storage_") as tmp:
            sid = SandboxId("sbx-1")
            base = LocalCheckpointManager(StorageConfig(root_dir=Path(tmp)))
            mgr = KeepAllCheckpointManager(base)
            manifest = CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity()
            mgr.put_manifest(manifest)
            mgr.handle_checkpoint_complete(manifest)

            self.assertEqual(mgr.list_checkpoints(sid), [CheckpointId("ckpt-1")])


if __name__ == "__main__":
    unittest.main()

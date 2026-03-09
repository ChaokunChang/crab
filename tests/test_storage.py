from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

from agent_cr import ArtifactKind, ArtifactPayload, CheckpointId, CheckpointManifest, LocalCheckpointManager, SandboxId, StorageConfig
from agent_cr.models import utc_now


class StorageTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from agent_cr.config import StorageConfig
from agent_cr.ids import CheckpointId, SandboxId
from agent_cr.models import CheckpointManifest, MANIFEST_SCHEMA_VERSION, utc_now
from agent_cr.storage.local import LocalCheckpointManager
from agent_cr.storage.policies import LatestOnlyCheckpointManager


def _manifest(
    sbx: SandboxId,
    ckpt: str,
    *,
    parent: str | None = None,
    kind: str = "full",
    has_process_artifacts: bool = True,
) -> CheckpointManifest:
    from agent_cr.models import ArtifactKind, ArtifactReference

    process_artifacts = []
    if has_process_artifacts:
        process_artifacts.append(
            ArtifactReference(
                kind=ArtifactKind.PROCESS,
                name="process_checkpoint.json",
                relative_path=f"artifacts/{sbx}/{ckpt}/process/process_checkpoint.json",
                size_bytes=1,
                sha256="x" * 64,
            )
        )
    return CheckpointManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        checkpoint_id=CheckpointId(ckpt),
        sandbox_id=sbx,
        created_at=utc_now(),
        runtime_name="runc",
        runtime_version="1.3.4",
        process_artifacts=process_artifacts,
        filesystem_artifacts=[],
        parent_checkpoint_id=CheckpointId(parent) if parent else None,
        process_kind=kind,
    ).with_integrity()


class LocalDeleteCascadeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.mgr = LocalCheckpointManager(StorageConfig(root_dir=self.root))
        self.sbx = SandboxId("sbx-1")
        self.mgr.put_manifest(_manifest(self.sbx, "c-1", kind="full"))
        self.mgr.put_manifest(_manifest(self.sbx, "c-2", parent="c-1", kind="incremental"))
        self.mgr.put_manifest(_manifest(self.sbx, "c-3", parent="c-2", kind="incremental"))

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_descendants_returns_transitive_chain_newest_first(self) -> None:
        desc = self.mgr.descendants(self.sbx, CheckpointId("c-1"))
        self.assertEqual([str(d) for d in desc], ["c-3", "c-2"])

    def test_default_delete_refuses_parent_with_descendants(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            self.mgr.delete_checkpoint(self.sbx, CheckpointId("c-1"))
        self.assertIn("live incremental descendants", str(ctx.exception))
        # Manifest is still on disk.
        self.assertIn(CheckpointId("c-1"), self.mgr.list_checkpoints(self.sbx))

    def test_cascade_delete_drops_whole_chain(self) -> None:
        self.mgr.delete_checkpoint(self.sbx, CheckpointId("c-1"), cascade=True)
        self.assertEqual(self.mgr.list_checkpoints(self.sbx), [])

    def test_delete_all_uses_reverse_order(self) -> None:
        # delete_all_checkpoints iterates newest-first so the descendants
        # check never trips, even though every checkpoint here is a parent
        # of some other checkpoint in the chain.
        self.mgr.delete_all_checkpoints(self.sbx)
        self.assertEqual(self.mgr.list_checkpoints(self.sbx), [])


class LatestOnlyChainProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.delegate = LocalCheckpointManager(StorageConfig(root_dir=self.root))
        # delete_filesystem_checkpoints=True so retention is aggressive;
        # the chain ancestors must still be protected because the latest
        # checkpoint is incremental.
        self.mgr = LatestOnlyCheckpointManager(self.delegate)
        self.sbx = SandboxId("sbx-1")

    def tearDown(self) -> None:
        try:
            self.mgr.close()
        except Exception:
            pass
        shutil.rmtree(self.root, ignore_errors=True)

    def test_chain_ancestors_are_protected(self) -> None:
        self.delegate.put_manifest(_manifest(self.sbx, "c-1", kind="full"))
        self.delegate.put_manifest(_manifest(self.sbx, "c-2", parent="c-1", kind="incremental"))
        self.delegate.put_manifest(_manifest(self.sbx, "c-3", parent="c-2", kind="incremental"))
        # Latest = c-3 (incremental). Its chain ancestors c-2, c-1 must be
        # protected even though only the latest checkpoint is normally retained.
        protected = self.mgr._protected_checkpoint_ids(self.sbx)
        self.assertIn(CheckpointId("c-1"), protected)
        self.assertIn(CheckpointId("c-2"), protected)
        self.assertIn(CheckpointId("c-3"), protected)


if __name__ == "__main__":
    unittest.main()

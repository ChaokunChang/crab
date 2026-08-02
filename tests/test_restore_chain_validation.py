from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from crab.config import StorageConfig
from crab.ids import CheckpointId, SandboxId
from crab.models import (
    CheckpointManifest,
    MANIFEST_SCHEMA_VERSION,
    RuntimeCapabilities,
    utc_now,
)
from crab.storage.local import LocalCheckpointManager
from crab.workers.composite import _validate_incremental_chain


class _StubRuntime:
    """Implements the subset of Runtime that chain validation needs."""

    name = "stub"
    version = "1.0"

    def __init__(self, image_root: Path, *, supports_incremental: bool = True) -> None:
        self._image_root = image_root
        self._supports = supports_incremental

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_process_checkpoint=True,
            supports_filesystem_checkpoint=True,
            supports_incremental_process=self._supports,
        )

    def pre_dump_location(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> str | None:
        return str(self._image_root / str(sandbox_id) / str(checkpoint_id) / "pre_dump")


def _manifest(sbx: SandboxId, ckpt: str, *, parent: str | None = None, kind: str = "full") -> CheckpointManifest:
    return CheckpointManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        checkpoint_id=CheckpointId(ckpt),
        sandbox_id=sbx,
        created_at=utc_now(),
        runtime_name="runc",
        runtime_version="1.3.4",
        process_artifacts=[],
        filesystem_artifacts=[],
        parent_checkpoint_id=CheckpointId(parent) if parent else None,
        process_kind=kind,
    ).with_integrity()


class IncrementalChainValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.images = self.root / "images"
        self.mgr = LocalCheckpointManager(StorageConfig(root_dir=self.root / "store"))
        self.runtime = _StubRuntime(self.images)
        self.sbx = SandboxId("sbx-1")

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _make_pre_dump_dir(self, ckpt: str) -> Path:
        path = Path(self.runtime.pre_dump_location(self.sbx, CheckpointId(ckpt)))
        path.mkdir(parents=True, exist_ok=True)
        return path

    def test_full_manifest_is_a_noop(self) -> None:
        m = _manifest(self.sbx, "c-1", kind="full")
        self.mgr.put_manifest(m)
        # No pre_dump dirs exist on disk; full manifests don't trigger walks.
        _validate_incremental_chain(self.runtime, self.mgr, m)

    def test_healthy_chain_validates(self) -> None:
        for cid in ["c-1", "c-2"]:
            self._make_pre_dump_dir(cid)
        self.mgr.put_manifest(_manifest(self.sbx, "c-1", kind="full"))
        self.mgr.put_manifest(_manifest(self.sbx, "c-2", parent="c-1", kind="incremental"))
        m3 = _manifest(self.sbx, "c-3", parent="c-2", kind="incremental")
        self.mgr.put_manifest(m3)
        _validate_incremental_chain(self.runtime, self.mgr, m3)

    def test_missing_intermediate_pre_dump_raises_filenotfound(self) -> None:
        self._make_pre_dump_dir("c-1")
        # c-2's pre_dump dir is intentionally NOT created.
        self.mgr.put_manifest(_manifest(self.sbx, "c-1", kind="full"))
        self.mgr.put_manifest(_manifest(self.sbx, "c-2", parent="c-1", kind="incremental"))
        m3 = _manifest(self.sbx, "c-3", parent="c-2", kind="incremental")
        self.mgr.put_manifest(m3)
        with self.assertRaises(FileNotFoundError) as ctx:
            _validate_incremental_chain(self.runtime, self.mgr, m3)
        self.assertIn("c-2", str(ctx.exception))

    def test_missing_parent_manifest_raises_filenotfound(self) -> None:
        self._make_pre_dump_dir("c-1")
        self._make_pre_dump_dir("c-2")
        # c-2 manifest deliberately not stored.
        self.mgr.put_manifest(_manifest(self.sbx, "c-1", kind="full"))
        m3 = _manifest(self.sbx, "c-3", parent="c-2", kind="incremental")
        self.mgr.put_manifest(m3)
        with self.assertRaises(FileNotFoundError) as ctx:
            _validate_incremental_chain(self.runtime, self.mgr, m3)
        self.assertIn("c-2", str(ctx.exception))

    def test_runtime_without_incremental_support_skips_validation(self) -> None:
        runtime = _StubRuntime(self.images, supports_incremental=False)
        # Even with a broken chain on disk, validation is a no-op when the
        # runtime doesn't model incremental dumps.
        m = _manifest(self.sbx, "c-3", parent="c-missing", kind="incremental")
        self.mgr.put_manifest(m)
        _validate_incremental_chain(runtime, self.mgr, m)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

from agent_cr import ArtifactKind, ArtifactReference, CheckpointId, CheckpointManifest, SandboxId
from agent_cr.models import MANIFEST_SCHEMA_VERSION, utc_now


class ManifestTests(unittest.TestCase):
    def test_manifest_integrity_roundtrip(self) -> None:
        manifest = CheckpointManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=SandboxId("sbx-1"),
            created_at=utc_now(),
            runtime_name="docker",
            runtime_version="stub",
            process_artifacts=[
                ArtifactReference(
                    kind=ArtifactKind.PROCESS,
                    name="process_plan.json",
                    relative_path="artifacts/sbx-1/ckpt-1/process/process_plan.json",
                    size_bytes=5,
                    sha256="abcde",
                )
            ],
            filesystem_artifacts=[],
            metadata={"foo": "bar"},
        ).with_integrity()

        parsed = CheckpointManifest.from_dict(manifest.to_dict())
        self.assertEqual(parsed.schema_version, MANIFEST_SCHEMA_VERSION)
        self.assertEqual(parsed.integrity["manifest_sha256"], manifest.integrity["manifest_sha256"])

    def test_manifest_integrity_detects_tamper(self) -> None:
        manifest = CheckpointManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=SandboxId("sbx-1"),
            created_at=utc_now(),
            runtime_name="docker",
            runtime_version="stub",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={"a": 1},
        ).with_integrity()

        raw = manifest.to_dict()
        raw["metadata"] = {"a": 2}
        with self.assertRaises(ValueError):
            CheckpointManifest.from_dict(raw)

    def test_manifest_version_validation(self) -> None:
        manifest = CheckpointManifest(
            schema_version="v0",
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=SandboxId("sbx-1"),
            created_at=utc_now(),
            runtime_name="docker",
            runtime_version="stub",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
            integrity={"manifest_sha256": "abc"},
        )
        with self.assertRaises(ValueError):
            manifest.validate_schema()


if __name__ == "__main__":
    unittest.main()

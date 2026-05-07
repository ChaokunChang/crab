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

    def test_full_manifest_canonical_omits_incremental_fields(self) -> None:
        # A "full" manifest with default incremental fields must serialize
        # identically to a pre-incremental v1 payload so old manifests on
        # disk keep validating against their stored integrity hash.
        manifest = CheckpointManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            checkpoint_id=CheckpointId("ckpt-1"),
            sandbox_id=SandboxId("sbx-1"),
            created_at=utc_now(),
            runtime_name="docker",
            runtime_version="stub",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={"k": "v"},
        ).with_integrity()
        canonical = manifest.to_canonical_json_bytes()
        self.assertNotIn(b"parent_checkpoint_id", canonical)
        self.assertNotIn(b"process_kind", canonical)

    def test_legacy_manifest_payload_loads_with_defaults(self) -> None:
        # Simulate a manifest written before the incremental fields existed:
        # no parent_checkpoint_id / process_kind keys, integrity stamped over
        # the pre-incremental canonical form.
        legacy = CheckpointManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            checkpoint_id=CheckpointId("ckpt-old"),
            sandbox_id=SandboxId("sbx-old"),
            created_at=utc_now(),
            runtime_name="docker",
            runtime_version="stub",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={"m": 1},
        ).with_integrity()
        raw = legacy.to_dict()
        self.assertNotIn("parent_checkpoint_id", raw)
        self.assertNotIn("process_kind", raw)
        parsed = CheckpointManifest.from_dict(raw)
        self.assertIsNone(parsed.parent_checkpoint_id)
        self.assertEqual(parsed.process_kind, "full")

    def test_incremental_manifest_roundtrip(self) -> None:
        manifest = CheckpointManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=SandboxId("sbx-1"),
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version="1.3.4",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
            parent_checkpoint_id=CheckpointId("ckpt-1"),
            process_kind="incremental",
        ).with_integrity()
        parsed = CheckpointManifest.from_dict(manifest.to_dict())
        self.assertEqual(parsed.parent_checkpoint_id, CheckpointId("ckpt-1"))
        self.assertEqual(parsed.process_kind, "incremental")
        # Hash differs from the same manifest without the incremental fields.
        plain = CheckpointManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            checkpoint_id=CheckpointId("ckpt-2"),
            sandbox_id=SandboxId("sbx-1"),
            created_at=manifest.created_at,
            runtime_name="runc",
            runtime_version="1.3.4",
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={},
        ).with_integrity()
        self.assertNotEqual(
            manifest.integrity["manifest_sha256"],
            plain.integrity["manifest_sha256"],
        )


if __name__ == "__main__":
    unittest.main()

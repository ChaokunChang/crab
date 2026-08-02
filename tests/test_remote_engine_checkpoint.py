from __future__ import annotations

import unittest

from crab.ids import CheckpointId, SandboxId
from crab.models import JobStatus
from crab.remote_engine import _SystemShim


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, object]] = []

    def get_json(self, path: str):
        self.calls.append(("GET", path, None))
        return {
            "checkpoints": [
                {
                    "checkpoint_id": "ckpt-one",
                    "created_at": "2026-08-02T12:00:00+00:00",
                    "label": "safe-point",
                    "has_process": True,
                    "has_filesystem": True,
                }
            ]
        }

    def post_json(self, path: str, payload: object, *, timeout_seconds: float):
        self.calls.append(("POST", path, payload))
        if path.endswith("/restore"):
            return {"status": "succeeded"}
        return {"checkpoint_id": "ckpt-two", "status": "succeeded"}

    def _request_json(self, method: str, path: str, *, body: bytes | None):
        self.calls.append((method, path, body))
        return {"ok": True}


class RemoteCheckpointShimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = _FakeClient()
        self.system = _SystemShim(self.client)  # type: ignore[arg-type]
        self.sandbox_id = SandboxId("sandbox-one")

    def test_checkpoint_and_restore_are_routed_to_daemon(self) -> None:
        created = self.system.checkpoint_once(self.sandbox_id)
        self.assertEqual(created.status, JobStatus.SUCCEEDED)
        self.assertEqual(str(created.manifest.checkpoint_id), "ckpt-two")

        restored = self.system.restore_once(self.sandbox_id, CheckpointId("ckpt-two"))
        self.assertEqual(restored.status, JobStatus.SUCCEEDED)

    def test_checkpoint_storage_lists_metadata_and_deletes(self) -> None:
        checkpoint_ids = self.system.storage.list_checkpoints(self.sandbox_id)
        self.assertEqual([str(item) for item in checkpoint_ids], ["ckpt-one"])

        manifest = self.system.storage.get_manifest(self.sandbox_id, checkpoint_ids[0])
        self.assertEqual(manifest.metadata["label"], "safe-point")
        self.assertTrue(manifest.process_artifacts)
        self.assertTrue(manifest.filesystem_artifacts)

        self.system.storage.delete_checkpoint(self.sandbox_id, checkpoint_ids[0], cascade=True)
        self.assertEqual(self.client.calls[-1][0], "DELETE")
        self.assertIn(b'"cascade": true', self.client.calls[-1][2])


if __name__ == "__main__":
    unittest.main()

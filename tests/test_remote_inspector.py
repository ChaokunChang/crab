from __future__ import annotations

import unittest
from datetime import datetime, timezone

from agent_cr import HostInspectorServiceClient, RemoteSandboxInspector, SandboxId, SandboxSnapshot


class RecordingServiceClient(HostInspectorServiceClient):
    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:1")
        self.reset_calls: list[tuple[SandboxId, datetime | None]] = []

    def get_proc_and_fs_status(self, sandbox_id: SandboxId) -> dict[str, object]:
        _ = sandbox_id
        return {
            "ok": True,
            "status": {
                "runtime_name": "docker",
                "is_running": True,
                "process_changed": True,
                "filesystem_changed": False,
                "observed_at": "2026-03-11T12:00:00+00:00",
                "last_reset_at": "2026-03-11T11:55:00+00:00",
                "metadata": {"object_id": "abc"},
            },
        }

    def reset_sandbox(self, sandbox_id: SandboxId, at: datetime | None) -> dict[str, object]:
        self.reset_calls.append((sandbox_id, at))
        return {"ok": True}


class FailingServiceClient(HostInspectorServiceClient):
    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:1")

    def get_proc_and_fs_status(self, sandbox_id: SandboxId) -> dict[str, object]:
        _ = sandbox_id
        raise RuntimeError("boom")

    def reset_sandbox(self, sandbox_id: SandboxId, at: datetime | None) -> dict[str, object]:
        _ = (sandbox_id, at)
        raise RuntimeError("boom")


class DeferredResetServiceClient(HostInspectorServiceClient):
    def __init__(self) -> None:
        super().__init__("http://127.0.0.1:1")
        self.registered = False
        self.reset_calls: list[tuple[SandboxId, datetime | None]] = []
        self._last_reset_at: datetime | None = None

    def get_proc_and_fs_status(self, sandbox_id: SandboxId) -> dict[str, object]:
        if not self.registered:
            raise KeyError(str(sandbox_id))
        return {
            "ok": True,
            "status": {
                "runtime_name": "runc",
                "is_running": True,
                "process_changed": False,
                "filesystem_changed": False,
                "observed_at": "2026-03-11T12:00:01+00:00",
                "last_reset_at": None if self._last_reset_at is None else self._last_reset_at.isoformat(),
                "metadata": {"object_id": str(sandbox_id)},
            },
        }

    def reset_sandbox(self, sandbox_id: SandboxId, at: datetime | None) -> dict[str, object]:
        if not self.registered:
            raise KeyError(str(sandbox_id))
        self.reset_calls.append((sandbox_id, at))
        self._last_reset_at = at
        return {"ok": True}


class RemoteInspectorTests(unittest.TestCase):
    def test_remote_inspector_maps_status_response(self) -> None:
        inspector = RemoteSandboxInspector(RecordingServiceClient())

        snapshot = inspector.inspect(SandboxId("sbx-1"))

        self.assertEqual(snapshot.runtime_name, "docker")
        self.assertTrue(snapshot.is_running)
        self.assertTrue(snapshot.process_changed)
        self.assertFalse(snapshot.filesystem_changed)
        self.assertEqual(snapshot.metadata["object_id"], "abc")
        self.assertEqual(snapshot.metadata["host_last_reset_at"], "2026-03-11T11:55:00+00:00")
        self.assertIsNone(snapshot.last_checkpoint_at)

    def test_remote_inspector_degrades_to_all_true_on_read_failure(self) -> None:
        inspector = RemoteSandboxInspector(FailingServiceClient())

        snapshot = inspector.inspect(SandboxId("sbx-1"))

        self.assertTrue(snapshot.process_changed)
        self.assertTrue(snapshot.filesystem_changed)
        self.assertTrue(snapshot.is_running)
        self.assertEqual(snapshot.runtime_name, "remote-inspector-unavailable")
        self.assertIn("boom", str(snapshot.metadata["inspector_error"]))

    def test_remote_inspector_resets_both_dimensions(self) -> None:
        client = RecordingServiceClient()
        inspector = RemoteSandboxInspector(client)
        at = datetime(2026, 3, 11, 12, 1, tzinfo=timezone.utc)

        inspector.mark_checkpoint_complete(SandboxId("sbx-1"), process=True, filesystem=False, at=at)

        self.assertEqual(client.reset_calls, [(SandboxId("sbx-1"), at)])

    def test_remote_inspector_uses_seed_snapshot_until_remote_registration_then_syncs_reset(self) -> None:
        client = DeferredResetServiceClient()
        inspector = RemoteSandboxInspector(client)
        observed_at = datetime(2026, 3, 11, 11, 59, tzinfo=timezone.utc)
        sandbox_id = SandboxId("sbx-seeded")

        inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=observed_at,
                metadata={"seeded": True},
            )
        )

        seeded = inspector.inspect(sandbox_id)
        self.assertEqual(seeded.runtime_name, "runc")
        self.assertFalse(seeded.process_changed)
        self.assertFalse(seeded.filesystem_changed)
        self.assertEqual(seeded.metadata, {"seeded": True})

        client.registered = True
        synced = inspector.inspect(sandbox_id)

        self.assertEqual(client.reset_calls, [(sandbox_id, observed_at)])
        self.assertFalse(synced.process_changed)
        self.assertFalse(synced.filesystem_changed)
        self.assertTrue(synced.metadata["seeded"])
        self.assertEqual(synced.metadata["object_id"], "sbx-seeded")
        self.assertEqual(synced.metadata["host_last_reset_at"], observed_at.isoformat())
        self.assertIsNone(synced.last_checkpoint_at)

    def test_remote_inspector_preserves_real_checkpoint_timestamp_from_local_state(self) -> None:
        client = DeferredResetServiceClient()
        client.registered = True
        inspector = RemoteSandboxInspector(client)
        sandbox_id = SandboxId("sbx-checkpointed")
        observed_at = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
        checkpoint_at = datetime(2026, 3, 11, 12, 5, tzinfo=timezone.utc)

        inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=True,
                observed_at=observed_at,
                metadata={"seeded": True},
            )
        )

        inspector.mark_checkpoint_complete(sandbox_id, process=True, filesystem=True, at=checkpoint_at)
        snapshot = inspector.inspect(sandbox_id)

        self.assertEqual(snapshot.last_checkpoint_at, checkpoint_at)
        self.assertEqual(snapshot.metadata["host_last_reset_at"], checkpoint_at.isoformat())
        self.assertFalse(snapshot.process_changed)
        self.assertFalse(snapshot.filesystem_changed)


if __name__ == "__main__":
    unittest.main()

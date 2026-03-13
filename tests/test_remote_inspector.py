from __future__ import annotations

import unittest
from datetime import datetime, timezone

from agent_cr import HostInspectorServiceClient, RemoteSandboxInspector, SandboxId


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


class RemoteInspectorTests(unittest.TestCase):
    def test_remote_inspector_maps_status_response(self) -> None:
        inspector = RemoteSandboxInspector(RecordingServiceClient())

        snapshot = inspector.inspect(SandboxId("sbx-1"))

        self.assertEqual(snapshot.runtime_name, "docker")
        self.assertTrue(snapshot.is_running)
        self.assertTrue(snapshot.process_changed)
        self.assertFalse(snapshot.filesystem_changed)
        self.assertEqual(snapshot.metadata["object_id"], "abc")
        self.assertEqual(snapshot.last_checkpoint_at, datetime(2026, 3, 11, 11, 55, tzinfo=timezone.utc))

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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import time
import unittest
import os
import stat as pystat
from unittest.mock import patch

from agent_cr.host_inspector.protocol import HelperEvent
from agent_cr.host_inspector.runtime_resolver import ResolvedSandbox
from agent_cr.host_inspector.server import HostInspectorDaemon


class FakeResolver:
    def __init__(self) -> None:
        self.item = ResolvedSandbox(
            runtime="docker",
            object_id="container-1",
            runtime_name="docker",
            is_running=True,
            init_pid=111,
            cgroup_path="/demo.scope",
            cgroup_id=6869,
        )

    def resolve(self, runtime: str, object_id: str) -> ResolvedSandbox:
        _ = (runtime, object_id)
        return self.item


class FakeFilesystemMonitor:
    def __init__(self) -> None:
        self.started = False
        self.on_event = None
        self.upserts: list[tuple[str, int]] = []
        self.removals: list[str] = []

    def start(self, on_event) -> None:
        self.started = True
        self.on_event = on_event

    def stop(self) -> None:
        self.started = False

    def upsert_sandbox(self, sandbox_id: str, cgroup_id: int) -> None:
        self.upserts.append((sandbox_id, cgroup_id))

    def remove_sandbox(self, sandbox_id: str) -> None:
        self.removals.append(sandbox_id)

class HostInspectorServerTests(unittest.TestCase):
    def test_register_reset_status_and_unregister(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        registered = {"status": daemon.register("sbx-1", "docker", "container-1")}
        self.assertTrue(registered["status"]["process_changed"])
        self.assertTrue(registered["status"]["filesystem_changed"])
        self.assertEqual(fs_monitor.upserts, [("sbx-1", 6869)])

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111, 222}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111, 222},
        ):
            reset = {"status": daemon.reset("sbx-1")}
        self.assertFalse(reset["status"]["process_changed"])
        self.assertFalse(reset["status"]["filesystem_changed"])

        daemon._handle_fs_event(
            HelperEvent(
                sandbox_id="sbx-1",
                kind="filesystem_change",
                syscall="openat",
                fd=3,
                flags=577,
                cgroup_id=6869,
                timestamp="2026-03-11T12:00:00+00:00",
            )
        )
        status = {"status": daemon.status("sbx-1")}
        self.assertFalse(status["status"]["process_changed"])
        self.assertTrue(status["status"]["filesystem_changed"])

        unregistered = daemon.unregister("sbx-1")
        self.assertTrue(unregistered["unregistered"])
        self.assertIn("sbx-1", fs_monitor.removals)

    def test_process_poll_latches_dirty_memory(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1")

        daemon.start()
        self.addCleanup(daemon.stop)

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value={111},
        ):
            time.sleep(0.12)

        status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["dirty_pids"], [111])

    def test_open_with_rdwr_only_does_not_latch_filesystem(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(
            HelperEvent(
                sandbox_id="sbx-1",
                kind="filesystem_change",
                syscall="openat",
                fd=3,
                flags=os.O_RDWR,
                cgroup_id=6869,
                timestamp="2026-03-11T12:00:00+00:00",
            )
        )
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_write_to_non_regular_fd_does_not_latch_filesystem(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        class _FakeStat:
            st_mode = pystat.S_IFIFO

        with patch("agent_cr.host_inspector.server.os.stat", return_value=_FakeStat()):
            daemon._handle_fs_event(
                HelperEvent(
                    sandbox_id="sbx-1",
                    kind="filesystem_change",
                    syscall="write",
                    pid=999,
                    fd=3,
                    flags=0,
                    cgroup_id=6869,
                    timestamp="2026-03-11T12:00:00+00:00",
                )
            )
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])


if __name__ == "__main__":
    unittest.main()

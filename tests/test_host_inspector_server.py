from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent_cr.host_inspector.process_filter import ProcessIdentity
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


def _event(
    *,
    sandbox_id: str = "sbx-1",
    syscall: str,
    pid: int | None = None,
    fd: int | None = None,
    fd_kind: str | None = None,
    flags: int | None = None,
    path: str | None = None,
    path_secondary: str | None = None,
    inode: int | None = None,
    device: int | None = None,
) -> HelperEvent:
    return HelperEvent(
        sandbox_id=sandbox_id,
        kind="filesystem_change",
        syscall=syscall,
        pid=pid,
        fd=fd,
        fd_kind=fd_kind,
        flags=flags,
        path=path,
        path_secondary=path_secondary,
        inode=inode,
        device=device,
        cgroup_id=6869,
        timestamp="2026-03-11T12:00:00+00:00",
    )

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
            _event(
                syscall="openat",
                fd=3,
                fd_kind="regular",
                flags=577,
                path="/tmp/demo.txt",
                inode=10,
                device=1,
            )
        )
        with patch("agent_cr.host_inspector.server.os.lstat"), patch(
            "agent_cr.host_inspector.server.list_cgroup_pids",
            return_value={111, 222},
        ), patch("agent_cr.host_inspector.server.dirty_pids", return_value=set()):
            status = {"status": daemon.status("sbx-1")}
        self.assertFalse(status["status"]["process_changed"])
        self.assertTrue(status["status"]["filesystem_changed"])

        unregistered = daemon.unregister("sbx-1")
        self.assertTrue(unregistered["unregistered"])
        self.assertIn("sbx-1", fs_monitor.removals)

    def test_status_reports_dirty_memory_for_current_live_pid(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1")

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value={111},
        ):
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["dirty_pids"], [111])

    def test_status_returns_false_after_transient_pid_exits(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1")

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            status = daemon.status("sbx-1")
        self.assertFalse(status["process_changed"])
        self.assertEqual(status["metadata"]["current_pids"], [111])

    def test_status_returns_true_for_live_pid_set_difference(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1")
            daemon.reset("sbx-1")

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111, 222}), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["current_pids"], [111, 222])

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

        daemon._handle_fs_event(_event(syscall="openat", fd=3, fd_kind="regular", flags=os.O_RDWR, path="/tmp/ro.txt"))
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

        daemon._handle_fs_event(_event(syscall="write", pid=999, fd=3, fd_kind="fifo", flags=0))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_unresolved_mutating_open_is_diagnostic_only(self) -> None:
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
            _event(
                syscall="openat",
                pid=999,
                fd=3,
                fd_kind="unknown",
                flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
            )
        )
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])
        self.assertTrue(status["metadata"]["unreconciled_fs_events"])

    def test_create_then_delete_same_file_clears_net_effect(self) -> None:
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
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="regular",
                flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                path="/work/tmp.txt",
                inode=101,
                device=1,
            )
        )
        daemon._handle_fs_event(_event(syscall="unlinkat", pid=222, path="/work/tmp.txt"))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])
        self.assertEqual(status["metadata"]["live_dirty_entries"], [])

    def test_touch_file_latches_filesystem_changed(self) -> None:
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
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="regular",
                flags=os.O_CREAT | os.O_WRONLY,
                path="/work/1.txt",
                inode=201,
                device=1,
            )
        )
        with patch("agent_cr.host_inspector.server.os.lstat"):
            status = daemon.status("sbx-1")
        self.assertTrue(status["filesystem_changed"])
        self.assertEqual(status["metadata"]["live_dirty_entries"][0]["path"], "/work/1.txt")

    def test_mkdir_then_rmdir_same_dir_clears_net_effect(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(_event(syscall="mkdirat", pid=222, path="/work/demo", device=1, inode=301))
        daemon._handle_fs_event(_event(syscall="rmdir", pid=222, path="/work/demo"))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_rename_new_file_then_delete_clears_net_effect(self) -> None:
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
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="regular",
                flags=os.O_CREAT | os.O_WRONLY,
                path="/work/a.txt",
                inode=401,
                device=1,
            )
        )
        daemon._handle_fs_event(
            _event(
                syscall="renameat2",
                pid=222,
                path="/work/a.txt",
                path_secondary="/work/b.txt",
                inode=401,
                device=1,
            )
        )
        daemon._handle_fs_event(_event(syscall="unlink", pid=222, path="/work/b.txt"))
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_delete_preexisting_file_remains_sticky(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        daemon.register("sbx-1", "docker", "container-1")

        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ):
            daemon.reset("sbx-1")

        daemon._handle_fs_event(_event(syscall="unlink", pid=222, path="/work/preexisting.txt"))
        status = daemon.status("sbx-1")
        self.assertTrue(status["filesystem_changed"])
        self.assertTrue(status["metadata"]["live_dirty_entries"][0]["deleted"])

    def test_created_file_missing_at_status_is_reconciled_away(self) -> None:
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
            _event(
                syscall="openat",
                pid=222,
                fd=3,
                fd_kind="unknown",
                flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                path="/tmp/short-lived.tmp",
            )
        )
        with patch("agent_cr.host_inspector.server.os.lstat", side_effect=FileNotFoundError), patch(
            "agent_cr.host_inspector.server.list_cgroup_pids",
            return_value={111},
        ), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])
        self.assertEqual(status["metadata"]["live_dirty_entries"], [])

    def test_ignored_pid_does_not_contribute_to_process_changed(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=0.05)
        rules = [{"executable_basename": "node", "cmdline_contains": ["iflow"]}]

        with patch(
            "agent_cr.host_inspector.process_filter.read_process_identity",
            side_effect=[
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "sleep 1")),
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "sleep 1")),
            ],
        ), patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111, 222}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={222},
        ), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")
            status = daemon.status("sbx-1")

        self.assertFalse(status["process_changed"])
        self.assertEqual(status["metadata"]["tracked_pids"], [222])
        self.assertEqual(status["metadata"]["ignored_pids"], [111])
        self.assertEqual(status["metadata"]["current_pids"], [222])

    def test_ignored_pid_filesystem_event_is_dropped(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [{"executable_basename": "node", "cmdline_contains": ["iflow"]}]

        with patch(
            "agent_cr.host_inspector.process_filter.read_process_identity",
            return_value=ProcessIdentity(
                pid=111,
                executable_path="/opt/iflow-runtime/node/bin/node",
                executable_basename="node",
                cmdline=("node", "iflow"),
            ),
        ), patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value=set(),
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")

        with patch(
            "agent_cr.host_inspector.process_filter.read_process_identity",
            return_value=ProcessIdentity(
                pid=111,
                executable_path="/opt/iflow-runtime/node/bin/node",
                executable_basename="node",
                cmdline=("node", "iflow"),
            ),
        ):
            daemon._handle_fs_event(
                _event(
                    syscall="openat",
                    pid=111,
                    fd=3,
                    fd_kind="regular",
                    flags=577,
                    path="/root/.iflow/history.json",
                    inode=501,
                    device=1,
                )
            )
        status = daemon.status("sbx-1")
        self.assertFalse(status["filesystem_changed"])

    def test_non_ignored_child_process_still_counts(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [{"executable_basename": "node", "cmdline_contains": ["iflow"]}]

        with patch(
            "agent_cr.host_inspector.process_filter.read_process_identity",
            side_effect=[
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "echo hi")),
                ProcessIdentity(pid=111, executable_path="/opt/iflow-runtime/node/bin/node", executable_basename="node", cmdline=("node", "iflow")),
                ProcessIdentity(pid=222, executable_path="/bin/sh", executable_basename="sh", cmdline=("sh", "-lc", "echo hi")),
            ],
        ), patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111, 222}), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={222},
        ), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value={222},
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["dirty_pids"], [222])

    def test_unresolved_proc_identity_defaults_to_tracked(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        rules = [{"executable_basename": "node", "cmdline_contains": ["iflow"]}]

        with patch("agent_cr.host_inspector.process_filter.read_process_identity", return_value=None), patch(
            "agent_cr.host_inspector.server.list_cgroup_pids",
            return_value={111},
        ), patch(
            "agent_cr.host_inspector.server.reset_soft_dirty_for_pids",
            return_value={111},
        ), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value={111},
        ):
            daemon.register("sbx-1", "docker", "container-1", ignore_process_rules=rules)
            daemon.reset("sbx-1")
            status = daemon.status("sbx-1")
        self.assertTrue(status["process_changed"])
        self.assertEqual(status["metadata"]["tracked_pids"], [111])
        self.assertEqual(status["metadata"]["ignored_pids"], [])


if __name__ == "__main__":
    unittest.main()

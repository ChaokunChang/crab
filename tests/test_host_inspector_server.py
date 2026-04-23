from __future__ import annotations

import io
import os
import subprocess
import unittest
from unittest.mock import Mock, patch

from agent_cr import HostInspectorServiceClient, SandboxId
from agent_cr.host_inspector.fs_helper import LibbpfFilesystemMonitor
from agent_cr.host_inspector.process_filter import ProcessIdentity
from agent_cr.host_inspector.protocol import HelperEvent
from agent_cr.host_inspector.runtime_resolver import ResolvedSandbox
from agent_cr.host_inspector.server import HostInspectorDaemon, HostInspectorServer


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
        self.sync_calls = 0
        self.ignored_pid_adds: list[int] = []
        self.ignored_pid_removes: list[int] = []

    def start(self, on_event) -> None:
        self.started = True
        self.on_event = on_event

    def stop(self) -> None:
        self.started = False

    def upsert_sandbox(self, sandbox_id: str, cgroup_id: int) -> None:
        self.upserts.append((sandbox_id, cgroup_id))

    def remove_sandbox(self, sandbox_id: str) -> None:
        self.removals.append(sandbox_id)

    def sync(self, timeout_s: float = 2.0) -> bool:
        self.sync_calls += 1
        return True

    def add_ignored_pid(self, pid: int) -> None:
        self.ignored_pid_adds.append(int(pid))

    def remove_ignored_pid(self, pid: int) -> None:
        self.ignored_pid_removes.append(int(pid))


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
    def test_server_closes_idle_connections_so_one_client_does_not_pin_only_worker(self) -> None:
        daemon = HostInspectorDaemon(resolver=FakeResolver(), fs_monitor=FakeFilesystemMonitor(), process_poll_interval_s=60.0)
        server = HostInspectorServer(host="127.0.0.1", port=0, daemon=daemon, max_workers=1)
        server.start()
        base_url = f"http://127.0.0.1:{server.port}"
        client_a = HostInspectorServiceClient(base_url, timeout_s=0.5)
        client_b = HostInspectorServiceClient(base_url, timeout_s=0.5)
        try:
            first = client_a.register_sandbox(SandboxId("sbx-a"), "docker", "container-1")
            second = client_b.register_sandbox(SandboxId("sbx-b"), "docker", "container-1")
        finally:
            client_a.close()
            client_b.close()
            server.stop()

        self.assertTrue(first["ok"])
        self.assertTrue(second["ok"])

    def test_filesystem_monitor_stop_kills_helper_after_terminate_timeout(self) -> None:
        monitor = LibbpfFilesystemMonitor(helper_path="/bin/true")
        stdout = Mock()
        stderr = Mock()
        stdin = Mock()
        process = Mock()
        process.pid = 12345
        process.poll.side_effect = [None, None]
        process.stdin = stdin
        process.stdout = stdout
        process.stderr = stderr
        process.wait.side_effect = [subprocess.TimeoutExpired(["fs_monitor"], 5.0), 0]
        monitor._process = process
        stdout_thread = Mock()
        stderr_thread = Mock()
        monitor._stdout_thread = stdout_thread
        monitor._stderr_thread = stderr_thread

        monitor.stop()

        stdin.close.assert_called_once()
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        self.assertEqual(process.wait.call_count, 2)
        stdout.close.assert_called_once()
        stderr.close.assert_called_once()
        stdout_thread.join.assert_called_once()
        stderr_thread.join.assert_called_once()
        self.assertIsNone(monitor._stdout_thread)
        self.assertIsNone(monitor._stderr_thread)

    def test_filesystem_monitor_reader_ignores_stream_close_during_shutdown(self) -> None:
        class RaisingStream:
            def __iter__(self):
                raise ValueError("I/O operation on closed file")

        monitor = LibbpfFilesystemMonitor(helper_path="/bin/true")
        process = Mock()
        process.stdout = RaisingStream()
        process.stderr = io.StringIO("")
        monitor._process = process

        monitor._read_stdout()

    def test_register_reset_status_and_unregister(self) -> None:
        resolver = FakeResolver()
        fs_monitor = FakeFilesystemMonitor()
        daemon = HostInspectorDaemon(resolver=resolver, fs_monitor=fs_monitor, process_poll_interval_s=60.0)
        registered = {"status": daemon.register("sbx-1", "docker", "container-1")}
        self.assertFalse(registered["status"]["process_changed"])
        self.assertFalse(registered["status"]["filesystem_changed"])
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

    def test_mutating_open_on_real_path_survives_racy_fd_kind(self) -> None:
        """`cat > file << 'EOF'` opens the real file then dup2/close/pipe-reuses
        the fd before the helper can stat /proc/<pid>/fd/<fd>. The helper then
        reports fd_kind=fifo (or socket) even though the syscall actually hit a
        regular file — the path in the BPF event is authoritative, so the
        mutating open must still latch filesystem_changed."""
        for racy_kind in ("fifo", "socket"):
            with self.subTest(racy_kind=racy_kind):
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
                        fd_kind=racy_kind,
                        flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                        path="/testbed/django/db/backends/postgresql/client.py",
                        inode=4242,
                        device=64,
                    )
                )
                status = daemon.status("sbx-1")
                self.assertTrue(status["filesystem_changed"])

    def test_mutating_open_on_dev_null_still_dropped(self) -> None:
        """The racy-fd_kind bypass must not accept writes into pseudo
        filesystems. /dev/null is the common shell-redirection sink and must
        keep being ignored."""
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
                fd_kind="char",
                flags=os.O_CREAT | os.O_WRONLY | os.O_TRUNC,
                path="/dev/null",
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

    def test_created_file_missing_at_status_still_counts_as_change(self) -> None:
        # A file that was created after reset and then disappeared (e.g., a
        # temp file that was unlinked but whose delete event was lost) is still
        # evidence that the sandbox mutated its filesystem.  Kernel tracepoint
        # events are authoritative — we should not second-guess them with a
        # host-side lstat that may fail for multiple valid reasons.
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
        with patch("agent_cr.host_inspector.server.list_cgroup_pids", return_value={111}), patch(
            "agent_cr.host_inspector.server.dirty_pids",
            return_value=set(),
        ):
            status = daemon.status("sbx-1")
        self.assertTrue(status["filesystem_changed"])
        self.assertEqual(len(status["metadata"]["live_dirty_entries"]), 1)

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

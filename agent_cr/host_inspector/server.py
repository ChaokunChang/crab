from __future__ import annotations

import argparse
import json
import logging
import os
import stat
import socket
from dataclasses import replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from typing import Any

from ..models import utc_now
from .fs_helper import LibbpfFilesystemMonitor
from .process_monitor import dirty_pids, list_cgroup_pids, reset_soft_dirty_for_pids
from .runtime_resolver import ResolvedSandbox, RuntimeResolver
from .state import SandboxRecord

logger = logging.getLogger(__name__)

_OPEN_SYSCALLS = {"open", "openat", "openat2", "creat"}
_WRITE_SYSCALLS = {"write", "pwrite64", "writev", "pwritev", "pwritev2"}
_DEVICE_OR_STREAM_FD_KINDS = {"char", "block", "fifo", "socket"}


def _isoformat(ts: datetime | None) -> str | None:
    return None if ts is None else ts.isoformat()


def _parse_ts(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


class HostInspectorDaemon:
    def __init__(
        self,
        *,
        resolver: RuntimeResolver | None = None,
        fs_monitor: LibbpfFilesystemMonitor | None = None,
        process_poll_interval_s: float = 1.0,
    ) -> None:
        self._resolver = resolver or RuntimeResolver()
        self._fs_monitor = fs_monitor or LibbpfFilesystemMonitor()
        self._process_poll_interval_s = process_poll_interval_s
        self._lock = Lock()
        self._records: dict[str, SandboxRecord] = {}

    def start(self) -> None:
        self._fs_monitor.start(self._handle_fs_event)
        logger.info(
            "process change detection is on-demand; process_poll_interval=%s is accepted for compatibility only",
            self._process_poll_interval_s,
        )

    def stop(self) -> None:
        self._fs_monitor.stop()

    def register(self, sandbox_id: str, runtime: str, object_id: str) -> dict[str, object]:
        resolved = self._resolver.resolve(runtime, object_id)
        observed_at = utc_now()
        with self._lock:
            previous = self._records.get(sandbox_id)
        record = SandboxRecord(
            sandbox_id=sandbox_id,
            runtime=runtime,
            object_id=object_id,
            runtime_name=resolved.runtime_name,
            is_running=resolved.is_running,
            init_pid=resolved.init_pid,
            cgroup_path=resolved.cgroup_path,
            cgroup_id=resolved.cgroup_id,
            process_changed=True,
            filesystem_changed=True,
            observed_at=observed_at,
        )
        with self._lock:
            self._records[sandbox_id] = record
        if previous is not None and previous.cgroup_id is not None and previous.cgroup_id != resolved.cgroup_id:
            self._fs_monitor.remove_sandbox(sandbox_id)
        if resolved.cgroup_id is not None:
            self._fs_monitor.upsert_sandbox(sandbox_id, resolved.cgroup_id)
        return self.status(sandbox_id)

    def unregister(self, sandbox_id: str) -> dict[str, object]:
        with self._lock:
            record = self._records.pop(sandbox_id, None)
        if record is None:
            raise KeyError(sandbox_id)
        self._fs_monitor.remove_sandbox(sandbox_id)
        return {"sandbox_id": sandbox_id, "unregistered": True}

    def status(self, sandbox_id: str) -> dict[str, object]:
        with self._lock:
            record = self._records.get(sandbox_id)
            if record is None:
                raise KeyError(sandbox_id)
        if record.last_reset_at is None:
            return self._record_to_response(record)

        resolved = self._resolver.resolve(record.runtime, record.object_id)
        current_pids = list_cgroup_pids(resolved.cgroup_path)
        dirty = dirty_pids(current_pids)
        updated = replace(
            record,
            runtime_name=resolved.runtime_name,
            is_running=resolved.is_running,
            init_pid=resolved.init_pid,
            cgroup_path=resolved.cgroup_path,
            cgroup_id=resolved.cgroup_id,
            current_pids=current_pids,
            dirty_pids=dirty,
            process_changed=(current_pids != record.baseline_pids) or bool(dirty),
            observed_at=max(record.observed_at or utc_now(), utc_now()),
            last_error=None,
        )
        with self._lock:
            self._records[sandbox_id] = updated
        if resolved.cgroup_id != record.cgroup_id:
            self._fs_monitor.remove_sandbox(sandbox_id)
            if resolved.cgroup_id is not None:
                self._fs_monitor.upsert_sandbox(sandbox_id, resolved.cgroup_id)
        return self._record_to_response(updated)

    def reset(self, sandbox_id: str, at: datetime | None = None) -> dict[str, object]:
        when = at or utc_now()
        with self._lock:
            existing = self._records.get(sandbox_id)
        if existing is None:
            raise KeyError(sandbox_id)

        resolved = self._resolver.resolve(existing.runtime, existing.object_id)
        current_pids = list_cgroup_pids(resolved.cgroup_path)
        baseline_pids = reset_soft_dirty_for_pids(current_pids)

        updated = replace(
            existing,
            runtime_name=resolved.runtime_name,
            is_running=resolved.is_running,
            init_pid=resolved.init_pid,
            cgroup_path=resolved.cgroup_path,
            cgroup_id=resolved.cgroup_id,
            baseline_pids=baseline_pids,
            current_pids=current_pids,
            dirty_pids=set(),
            process_changed=False,
            filesystem_changed=False,
            last_reset_at=when,
            observed_at=max(existing.observed_at or when, when),
            fs_event_count_since_reset=0,
            recent_fs_events=[],
            last_error=None,
        )
        with self._lock:
            self._records[sandbox_id] = updated
        if resolved.cgroup_id is not None:
            self._fs_monitor.upsert_sandbox(sandbox_id, resolved.cgroup_id)
        return self._record_to_response(updated)

    def _handle_fs_event(self, event) -> None:
        if not self._is_countable_fs_event(event):
            return
        now = utc_now()
        with self._lock:
            record = self._records.get(event.sandbox_id)
            if record is None:
                return
            self._records[event.sandbox_id] = replace(
                record,
                filesystem_changed=True,
                observed_at=max(record.observed_at or now, _parse_ts(event.timestamp) or now),
                fs_event_count_since_reset=record.fs_event_count_since_reset + 1,
                recent_fs_events=(
                    record.recent_fs_events
                    + [
                        {
                            "syscall": str(event.syscall),
                            "pid": None if event.pid is None else int(event.pid),
                            "fd": None if event.fd is None else int(event.fd),
                            "fd_kind": None if event.fd_kind is None else str(event.fd_kind),
                            "flags": None if event.flags is None else int(event.flags),
                        }
                    ]
                )[-8:],
            )

    def _is_countable_fs_event(self, event) -> bool:
        syscall = str(event.syscall)
        if syscall in _OPEN_SYSCALLS:
            return self._is_mutating_regular_open(event)
        if syscall in _WRITE_SYSCALLS:
            if event.fd_kind in _DEVICE_OR_STREAM_FD_KINDS:
                return False
            fd = int(event.fd or -1)
            if fd in (1, 2) or fd < 0:
                return False
            pid = int(event.pid or -1)
            if pid <= 0:
                return True
            try:
                mode = os.stat(f"/proc/{pid}/fd/{fd}").st_mode
            except OSError:
                # Short-lived exec processes often disappear before fd resolution here.
                # Treat unresolved write targets as non-filesystem writes and rely on
                # open/truncate/rename/unlink-style events for sticky fs changes.
                return False
            return stat.S_ISREG(mode)
        return True

    def _is_mutating_regular_open(self, event) -> bool:
        flags = int(event.flags or 0)
        if not self._has_mutating_open_flags(flags):
            return False
        if event.fd_kind in _DEVICE_OR_STREAM_FD_KINDS:
            return False
        if event.fd_kind == "regular":
            return True
        fd = int(event.fd or -1)
        pid = int(event.pid or -1)
        if fd < 0 or pid <= 0:
            return True
        try:
            mode = os.stat(f"/proc/{pid}/fd/{fd}").st_mode
        except OSError:
            return True
        return stat.S_ISREG(mode)

    def _has_mutating_open_flags(self, flags: int) -> bool:
        if flags & (os.O_CREAT | os.O_TRUNC):
            return True
        tmpfile_flags = getattr(os, "O_TMPFILE", 0)
        return bool(tmpfile_flags) and (flags & tmpfile_flags) == tmpfile_flags

    def _record_to_response(self, record: SandboxRecord) -> dict[str, object]:
        return {
            "sandbox_id": record.sandbox_id,
            "runtime_name": record.runtime_name,
            "is_running": record.is_running,
            "process_changed": record.process_changed,
            "filesystem_changed": record.filesystem_changed,
            "observed_at": _isoformat(record.observed_at),
            "last_reset_at": _isoformat(record.last_reset_at),
            "metadata": record.metadata(),
        }


class HostInspectorServer:
    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        daemon: HostInspectorDaemon | None = None,
    ) -> None:
        self._daemon = daemon or HostInspectorDaemon()
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread: Thread | None = None

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def start(self) -> None:
        self._daemon.start()
        self._thread = Thread(target=self._server.serve_forever, name="host-inspector-http", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._daemon.stop()

    def _build_handler(self):
        daemon = self._daemon

        class Handler(BaseHTTPRequestHandler):
            def _read_json(self) -> dict[str, Any]:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return {}
                payload = self.rfile.read(length)
                return json.loads(payload.decode("utf-8"))

            def _write_json(self, code: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload, sort_keys=True).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/healthz":
                    self._write_json(HTTPStatus.OK, {"ok": True})
                    return
                self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    if self.path == "/register":
                        payload = self._read_json()
                        result = daemon.register(
                            sandbox_id=str(payload["sandbox_id"]),
                            runtime=str(payload["runtime"]),
                            object_id=str(payload["object_id"]),
                        )
                        self._write_json(HTTPStatus.OK, {"ok": True, "status": result})
                        return
                    if self.path == "/get_proc_and_fs_status":
                        payload = self._read_json()
                        result = daemon.status(str(payload["sandbox_id"]))
                        self._write_json(HTTPStatus.OK, {"ok": True, "status": result})
                        return
                    if self.path == "/reset":
                        payload = self._read_json()
                        result = daemon.reset(str(payload["sandbox_id"]), at=_parse_ts(payload.get("at")))
                        self._write_json(HTTPStatus.OK, {"ok": True, "status": result})
                        return
                    if self.path == "/unregister":
                        payload = self._read_json()
                        result = daemon.unregister(str(payload["sandbox_id"]))
                        self._write_json(HTTPStatus.OK, {"ok": True, **result})
                        return
                    self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": "not found"})
                except KeyError as exc:
                    self._write_json(HTTPStatus.NOT_FOUND, {"ok": False, "error": f"unknown sandbox: {exc.args[0]}"})
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Host inspector request failed")
                    self._write_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(exc)})

            def log_message(self, fmt: str, *args) -> None:
                logger.debug("host-inspector: " + fmt, *args)

        return Handler


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="agent_cr host inspector server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9782)
    parser.add_argument("--process-poll-interval", type=float, default=1.0)
    parser.add_argument("--helper-path", default=None)
    parser.add_argument("--runc-state-root", default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    daemon = HostInspectorDaemon(
        resolver=RuntimeResolver(runc_state_root=args.runc_state_root),
        fs_monitor=LibbpfFilesystemMonitor(helper_path=args.helper_path),
        process_poll_interval_s=args.process_poll_interval,
    )
    server = HostInspectorServer(host=args.host, port=args.port, daemon=daemon)
    server.start()
    logger.info("host inspector listening on %s:%s", args.host, server.port)
    stop = Event()

    try:
        while not stop.wait(1.0):
            continue
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

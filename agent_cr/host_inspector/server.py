from __future__ import annotations

import argparse
import json
import logging
import os
import socket
from dataclasses import replace
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Lock, Thread
from typing import Any

from ..models import utc_now
from .fs_helper import LibbpfFilesystemMonitor
from .process_filter import parse_process_ignore_rules, pid_matches_ignore_rules
from .process_monitor import dirty_pids, list_cgroup_pids, reset_soft_dirty_for_pids
from .runtime_resolver import ResolvedSandbox, RuntimeResolver
from .state import SandboxRecord

logger = logging.getLogger(__name__)

_OPEN_SYSCALLS = {"open", "openat", "openat2", "creat"}
_WRITE_SYSCALLS = {"write", "pwrite64", "writev", "pwritev", "pwritev2", "truncate", "ftruncate"}
_METADATA_PATH_SYSCALLS = {
    "chmod",
    "fchmodat",
    "chown",
    "fchownat",
    "lchown",
    "setxattr",
    "lsetxattr",
    "removexattr",
    "lremovexattr",
}
_METADATA_FD_SYSCALLS = {"fchmod", "fchown", "fsetxattr", "fremovexattr"}
_CREATE_PATH_SYSCALLS = {"mkdir", "mkdirat", "mknod", "mknodat"}
_DELETE_SYSCALLS = {"unlink", "unlinkat", "rmdir"}
_RENAME_SYSCALLS = {"rename", "renameat", "renameat2"}
_SECONDARY_TARGET_SYSCALLS = {"link", "linkat", "symlink", "symlinkat"}
_DEVICE_OR_STREAM_FD_KINDS = {"char", "block", "fifo", "socket"}
_MUTATING_FD_KINDS = {"regular", "directory", "symlink", "unknown"}
_RECENT_EVENT_LIMIT = 8


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

    def register(
        self,
        sandbox_id: str,
        runtime: str,
        object_id: str,
        *,
        ignore_process_rules: object | None = None,
    ) -> dict[str, object]:
        resolved = self._resolver.resolve(runtime, object_id)
        observed_at = utc_now()
        parsed_ignore_rules = parse_process_ignore_rules(ignore_process_rules)
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
            ignore_process_rules=parsed_ignore_rules,
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
        all_current_pids = list_cgroup_pids(resolved.cgroup_path)
        tracked_pids, ignored_pids = self._split_pids(all_current_pids, record.ignore_process_rules)
        dirty = dirty_pids(tracked_pids)
        live_dirty_entries = self._reconcile_live_dirty_entries(record.live_dirty_entries, resolved.init_pid)
        filesystem_changed = bool(live_dirty_entries) or any(
            bool(item.get("counts_as_change")) for item in record.unreconciled_fs_events
        )
        updated = replace(
            record,
            runtime_name=resolved.runtime_name,
            is_running=resolved.is_running,
            init_pid=resolved.init_pid,
            cgroup_path=resolved.cgroup_path,
            cgroup_id=resolved.cgroup_id,
            current_pids=tracked_pids,
            dirty_pids=dirty,
            tracked_pids=tracked_pids,
            ignored_pids=ignored_pids,
            process_changed=(tracked_pids != record.baseline_pids) or bool(dirty),
            filesystem_changed=filesystem_changed,
            live_dirty_entries=live_dirty_entries,
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
        all_current_pids = list_cgroup_pids(resolved.cgroup_path)
        tracked_pids, ignored_pids = self._split_pids(all_current_pids, existing.ignore_process_rules)
        baseline_pids = reset_soft_dirty_for_pids(tracked_pids)

        updated = replace(
            existing,
            runtime_name=resolved.runtime_name,
            is_running=resolved.is_running,
            init_pid=resolved.init_pid,
            cgroup_path=resolved.cgroup_path,
            cgroup_id=resolved.cgroup_id,
            baseline_pids=baseline_pids,
            current_pids=tracked_pids,
            dirty_pids=set(),
            tracked_pids=tracked_pids,
            ignored_pids=ignored_pids,
            process_changed=False,
            filesystem_changed=False,
            last_reset_at=when,
            observed_at=max(existing.observed_at or when, when),
            fs_event_count_since_reset=0,
            recent_fs_events=[],
            live_dirty_entries={},
            unreconciled_fs_events=[],
            last_error=None,
        )
        with self._lock:
            self._records[sandbox_id] = updated
        if resolved.cgroup_id is not None:
            self._fs_monitor.upsert_sandbox(sandbox_id, resolved.cgroup_id)
        return self._record_to_response(updated)

    def _handle_fs_event(self, event) -> None:
        now = utc_now()
        with self._lock:
            record = self._records.get(event.sandbox_id)
            if record is None:
                return
            if int(event.pid or 0) > 0 and pid_matches_ignore_rules(int(event.pid or 0), record.ignore_process_rules):
                return
            if not self._is_countable_fs_event(event):
                return
            live_dirty_entries = {key: dict(value) for key, value in record.live_dirty_entries.items()}
            unreconciled_fs_events = list(record.unreconciled_fs_events)
            self._apply_fs_event(event, live_dirty_entries, unreconciled_fs_events)
            filesystem_changed = bool(live_dirty_entries) or any(
                bool(item.get("counts_as_change")) for item in unreconciled_fs_events
            )
            self._records[event.sandbox_id] = replace(
                record,
                filesystem_changed=filesystem_changed,
                observed_at=max(record.observed_at or now, _parse_ts(event.timestamp) or now),
                fs_event_count_since_reset=record.fs_event_count_since_reset + 1,
                recent_fs_events=(record.recent_fs_events + [self._event_to_dict(event)])[-_RECENT_EVENT_LIMIT:],
                live_dirty_entries=live_dirty_entries,
                unreconciled_fs_events=unreconciled_fs_events[-_RECENT_EVENT_LIMIT:],
            )

    def _is_countable_fs_event(self, event) -> bool:
        syscall = str(event.syscall)
        if syscall in _OPEN_SYSCALLS:
            return self._has_mutating_open_flags(int(event.flags or 0)) and event.fd_kind not in _DEVICE_OR_STREAM_FD_KINDS
        if syscall in _WRITE_SYSCALLS or syscall in _METADATA_FD_SYSCALLS:
            fd = int(event.fd or -1)
            if fd in (1, 2) or fd < 0:
                return False
            return str(event.fd_kind or "unknown") in _MUTATING_FD_KINDS
        if syscall in _METADATA_PATH_SYSCALLS | _CREATE_PATH_SYSCALLS | _DELETE_SYSCALLS | _RENAME_SYSCALLS | _SECONDARY_TARGET_SYSCALLS:
            return True
        return bool(event.path or event.path_secondary)

    def _has_mutating_open_flags(self, flags: int) -> bool:
        if flags & (os.O_CREAT | os.O_TRUNC):
            return True
        tmpfile_flags = getattr(os, "O_TMPFILE", 0)
        return bool(tmpfile_flags) and (flags & tmpfile_flags) == tmpfile_flags

    def _apply_fs_event(
        self,
        event,
        live_dirty_entries: dict[str, dict[str, object]],
        unreconciled_fs_events: list[dict[str, object]],
    ) -> None:
        syscall = str(event.syscall)
        if syscall in _RENAME_SYSCALLS:
            self._apply_rename_event(event, live_dirty_entries, unreconciled_fs_events)
            return
        if syscall in _DELETE_SYSCALLS:
            self._apply_delete_event(event, live_dirty_entries, unreconciled_fs_events)
            return
        self._apply_mutation_event(event, live_dirty_entries, unreconciled_fs_events)

    def _apply_mutation_event(
        self,
        event,
        live_dirty_entries: dict[str, dict[str, object]],
        unreconciled_fs_events: list[dict[str, object]],
    ) -> None:
        syscall = str(event.syscall)
        target_path = self._event_target_path(event)
        target_key = self._event_target_key(event, target_path)
        if target_key is None:
            self._append_unreconciled(
                unreconciled_fs_events,
                event,
                reason="missing_identity",
                counts_as_change=False,
            )
            return
        existing_key = self._lookup_entry_key(
            live_dirty_entries,
            preferred_key=target_key,
            path=target_path,
            device=event.device,
            inode=event.inode,
        )
        created_after_reset = self._event_creates_new_entry(event)
        entry = (
            dict(live_dirty_entries.pop(existing_key))
            if existing_key is not None
            else self._new_dirty_entry(
                key=target_key,
                path=target_path,
                device=event.device,
                inode=event.inode,
                syscall=syscall,
                created_after_reset=created_after_reset,
            )
        )
        entry["path"] = target_path
        entry["device"] = event.device
        entry["inode"] = event.inode
        entry["last_syscall"] = syscall
        entry["event_count"] = int(entry.get("event_count", 0)) + 1
        entry["created_after_reset"] = bool(entry.get("created_after_reset")) or created_after_reset
        live_dirty_entries[target_key] = entry

    def _apply_delete_event(
        self,
        event,
        live_dirty_entries: dict[str, dict[str, object]],
        unreconciled_fs_events: list[dict[str, object]],
    ) -> None:
        delete_path = self._event_target_path(event)
        delete_key = self._event_target_key(event, delete_path)
        existing_key = self._lookup_entry_key(
            live_dirty_entries,
            preferred_key=delete_key,
            path=delete_path,
            device=event.device,
            inode=event.inode,
        )
        if existing_key is not None:
            entry = dict(live_dirty_entries[existing_key])
            if bool(entry.get("created_after_reset")):
                live_dirty_entries.pop(existing_key, None)
                return
            entry["path"] = delete_path
            entry["deleted"] = True
            entry["last_syscall"] = str(event.syscall)
            entry["event_count"] = int(entry.get("event_count", 0)) + 1
            live_dirty_entries[existing_key] = entry
            return
        if delete_key is None:
            self._append_unreconciled(
                unreconciled_fs_events,
                event,
                reason="delete_without_identity",
                counts_as_change=True,
            )
            return
        live_dirty_entries[delete_key] = self._new_dirty_entry(
            key=delete_key,
            path=delete_path,
            device=event.device,
            inode=event.inode,
            syscall=str(event.syscall),
            created_after_reset=False,
            deleted=True,
        )

    def _apply_rename_event(
        self,
        event,
        live_dirty_entries: dict[str, dict[str, object]],
        unreconciled_fs_events: list[dict[str, object]],
    ) -> None:
        old_path = self._clean_path(event.path)
        new_path = self._clean_path(event.path_secondary)
        new_key = self._event_secondary_key(event, new_path)
        existing_key = self._lookup_entry_key(
            live_dirty_entries,
            preferred_key=self._path_key(old_path),
            path=old_path,
            device=None,
            inode=None,
        )
        if existing_key is None:
            existing_key = self._lookup_entry_key(
                live_dirty_entries,
                preferred_key=new_key,
                path=new_path,
                device=event.device,
                inode=event.inode,
            )
        if existing_key is None and new_key is None:
            self._append_unreconciled(
                unreconciled_fs_events,
                event,
                reason="rename_without_identity",
                counts_as_change=True,
            )
            return
        if existing_key is None:
            live_dirty_entries[new_key] = self._new_dirty_entry(
                key=new_key,
                path=new_path,
                device=event.device,
                inode=event.inode,
                syscall=str(event.syscall),
                created_after_reset=False,
            )
            live_dirty_entries[new_key]["original_path"] = old_path
            return
        entry = dict(live_dirty_entries.pop(existing_key))
        entry["path"] = new_path or old_path
        entry["device"] = event.device
        entry["inode"] = event.inode
        entry["last_syscall"] = str(event.syscall)
        entry["event_count"] = int(entry.get("event_count", 0)) + 1
        entry.setdefault("original_path", old_path)
        live_dirty_entries[new_key or existing_key] = entry

    def _event_to_dict(self, event) -> dict[str, object]:
        return {
            "syscall": str(event.syscall),
            "pid": None if event.pid is None else int(event.pid),
            "fd": None if event.fd is None else int(event.fd),
            "fd_kind": None if event.fd_kind is None else str(event.fd_kind),
            "flags": None if event.flags is None else int(event.flags),
            "path": self._clean_path(event.path),
            "path_secondary": self._clean_path(event.path_secondary),
            "inode": None if event.inode is None else int(event.inode),
            "device": None if event.device is None else int(event.device),
        }

    def _append_unreconciled(
        self,
        unreconciled_fs_events: list[dict[str, object]],
        event,
        *,
        reason: str,
        counts_as_change: bool,
    ) -> None:
        item = self._event_to_dict(event)
        item["reason"] = reason
        item["counts_as_change"] = counts_as_change
        unreconciled_fs_events.append(item)

    def _event_target_path(self, event) -> str | None:
        syscall = str(event.syscall)
        if syscall in _SECONDARY_TARGET_SYSCALLS:
            return self._clean_path(event.path_secondary)
        return self._clean_path(event.path)

    def _event_target_key(self, event, target_path: str | None) -> str | None:
        syscall = str(event.syscall)
        if syscall in _SECONDARY_TARGET_SYSCALLS:
            return self._identity_key(event.device, event.inode, target_path)
        return self._identity_key(event.device, event.inode, target_path)

    def _event_secondary_key(self, event, target_path: str | None) -> str | None:
        return self._identity_key(event.device, event.inode, target_path)

    def _event_creates_new_entry(self, event) -> bool:
        syscall = str(event.syscall)
        if syscall in _CREATE_PATH_SYSCALLS | _SECONDARY_TARGET_SYSCALLS:
            return True
        if syscall in _OPEN_SYSCALLS:
            flags = int(event.flags or 0)
            if flags & os.O_CREAT:
                return True
            tmpfile_flags = getattr(os, "O_TMPFILE", 0)
            return bool(tmpfile_flags) and (flags & tmpfile_flags) == tmpfile_flags
        return False

    def _new_dirty_entry(
        self,
        *,
        key: str,
        path: str | None,
        device: int | None,
        inode: int | None,
        syscall: str,
        created_after_reset: bool,
        deleted: bool = False,
    ) -> dict[str, object]:
        return {
            "key": key,
            "path": path,
            "device": device,
            "inode": inode,
            "created_after_reset": created_after_reset,
            "deleted": deleted,
            "first_syscall": syscall,
            "last_syscall": syscall,
            "event_count": 1,
        }

    def _lookup_entry_key(
        self,
        live_dirty_entries: dict[str, dict[str, object]],
        *,
        preferred_key: str | None,
        path: str | None,
        device: int | None,
        inode: int | None,
    ) -> str | None:
        if preferred_key is not None and preferred_key in live_dirty_entries:
            return preferred_key
        for key, entry in live_dirty_entries.items():
            if path is not None and entry.get("path") == path:
                return key
            if device is not None and inode is not None:
                if entry.get("device") == device and entry.get("inode") == inode:
                    return key
        return None

    def _identity_key(self, device: int | None, inode: int | None, path: str | None) -> str | None:
        if device is not None and inode is not None:
            return f"ino:{int(device)}:{int(inode)}"
        return self._path_key(path)

    def _path_key(self, path: str | None) -> str | None:
        cleaned = self._clean_path(path)
        if not cleaned:
            return None
        return f"path:{cleaned}"

    def _clean_path(self, path: str | None) -> str | None:
        if path is None:
            return None
        cleaned = str(path).strip()
        return cleaned or None

    def _reconcile_live_dirty_entries(
        self,
        live_dirty_entries: dict[str, dict[str, object]],
        init_pid: int | None,
    ) -> dict[str, dict[str, object]]:
        reconciled: dict[str, dict[str, object]] = {}
        for key, value in live_dirty_entries.items():
            entry = dict(value)
            path = self._clean_path(entry.get("path"))
            if bool(entry.get("created_after_reset")) and not bool(entry.get("deleted")) and path:
                try:
                    os.lstat(self._sandbox_path_for_stat(path, init_pid))
                except OSError:
                    continue
            reconciled[key] = entry
        return reconciled

    def _sandbox_path_for_stat(self, path: str, init_pid: int | None) -> str:
        if init_pid is None or not path.startswith("/"):
            return path
        return f"/proc/{int(init_pid)}/root{path}"

    def _split_pids(
        self,
        pids: set[int],
        ignore_process_rules,
    ) -> tuple[set[int], set[int]]:
        tracked: set[int] = set()
        ignored: set[int] = set()
        for pid in sorted(pids):
            if pid_matches_ignore_rules(pid, ignore_process_rules):
                ignored.add(pid)
            else:
                tracked.add(pid)
        return tracked, ignored

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
                            ignore_process_rules=payload.get("ignore_process_rules"),
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

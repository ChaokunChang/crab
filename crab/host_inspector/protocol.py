from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class HelperEvent:
    sandbox_id: str
    kind: str
    syscall: str
    pid: int | None = None
    fd: int | None = None
    fd_kind: str | None = None
    flags: int | None = None
    path: str | None = None
    path_secondary: str | None = None
    inode: int | None = None
    device: int | None = None
    cgroup_id: int | None = None
    timestamp: str | None = None


def encode_command(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def decode_event(line: str) -> HelperEvent:
    raw = json.loads(line)
    return HelperEvent(
        sandbox_id=str(raw["sandbox_id"]),
        kind=str(raw["kind"]),
        syscall=str(raw["syscall"]),
        pid=None if raw.get("pid") is None else int(raw["pid"]),
        fd=None if raw.get("fd") is None else int(raw["fd"]),
        fd_kind=None if raw.get("fd_kind") is None else str(raw["fd_kind"]),
        flags=None if raw.get("flags") is None else int(raw["flags"]),
        path=None if raw.get("path") is None else str(raw["path"]),
        path_secondary=None if raw.get("path_secondary") is None else str(raw["path_secondary"]),
        inode=None if raw.get("inode") is None else int(raw["inode"]),
        device=None if raw.get("device") is None else int(raw["device"]),
        cgroup_id=None if raw.get("cgroup_id") is None else int(raw["cgroup_id"]),
        timestamp=None if raw.get("timestamp") is None else str(raw["timestamp"]),
    )

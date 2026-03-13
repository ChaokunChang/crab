from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SandboxRecord:
    sandbox_id: str
    runtime: str
    object_id: str
    runtime_name: str
    is_running: bool
    init_pid: int | None = None
    cgroup_path: str | None = None
    cgroup_id: int | None = None
    baseline_pids: set[int] = field(default_factory=set)
    current_pids: set[int] = field(default_factory=set)
    dirty_pids: set[int] = field(default_factory=set)
    last_reset_at: datetime | None = None
    process_changed: bool = True
    filesystem_changed: bool = True
    observed_at: datetime | None = None
    fs_event_count_since_reset: int = 0
    recent_fs_events: list[dict[str, object]] = field(default_factory=list)
    last_error: str | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "cgroup_id": self.cgroup_id,
            "cgroup_path": self.cgroup_path,
            "init_pid": self.init_pid,
            "baseline_pids": sorted(self.baseline_pids),
            "current_pids": sorted(self.current_pids),
            "dirty_pids": sorted(self.dirty_pids),
            "fs_event_count_since_reset": self.fs_event_count_since_reset,
            "recent_fs_events": list(self.recent_fs_events),
            "last_error": self.last_error,
        }

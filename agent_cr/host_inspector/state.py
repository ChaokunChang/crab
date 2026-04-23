from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .process_filter import ProcessIgnoreRule


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
    ignore_process_rules: tuple[ProcessIgnoreRule, ...] = field(default_factory=tuple)
    baseline_pids: set[int] = field(default_factory=set)
    current_pids: set[int] = field(default_factory=set)
    dirty_pids: set[int] = field(default_factory=set)
    tracked_pids: set[int] = field(default_factory=set)
    ignored_pids: set[int] = field(default_factory=set)
    last_reset_at: datetime | None = None
    process_changed: bool = True
    filesystem_changed: bool = True
    observed_at: datetime | None = None
    fs_event_count_since_reset: int = 0
    recent_fs_events: list[dict[str, object]] = field(default_factory=list)
    live_dirty_entries: dict[str, dict[str, object]] = field(default_factory=dict)
    # Reverse indexes into live_dirty_entries, kept in sync by _dirty_put /
    # _dirty_pop under sandbox_lock. Without them _lookup_entry_key had to
    # scan the full dirty set on every event whose preferred key (by inode)
    # wasn't yet registered — O(N) per event, O(N*M) over a burst of new
    # file creations (tar/make install), enough to blow fs_monitor.sync
    # barriers at N~10k.
    dirty_by_path: dict[str, str] = field(default_factory=dict)
    dirty_by_inode: dict[tuple[int, int], str] = field(default_factory=dict)
    unreconciled_fs_events: list[dict[str, object]] = field(default_factory=list)
    last_error: str | None = None

    def metadata(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "cgroup_id": self.cgroup_id,
            "cgroup_path": self.cgroup_path,
            "init_pid": self.init_pid,
            "ignore_process_rules": [rule.to_dict() for rule in self.ignore_process_rules],
            "baseline_pids": sorted(self.baseline_pids),
            "current_pids": sorted(self.current_pids),
            "dirty_pids": sorted(self.dirty_pids),
            "tracked_pids": sorted(self.tracked_pids),
            "ignored_pids": sorted(self.ignored_pids),
            "fs_event_count_since_reset": self.fs_event_count_since_reset,
            "recent_fs_events": list(self.recent_fs_events),
            "live_dirty_entries": [
                dict(entry)
                for _, entry in sorted(self.live_dirty_entries.items(), key=lambda item: item[0])
            ],
            "unreconciled_fs_events": list(self.unreconciled_fs_events),
            "last_error": self.last_error,
        }

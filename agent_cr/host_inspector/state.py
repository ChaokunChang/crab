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
    # Subset of `ignored_pids` that should also be filtered out of fs-event
    # processing (i.e., scope=all matches only). PIDs ignored under
    # scope=process_only are in `ignored_pids` but NOT here, so their fs
    # events still flow through `_handle_fs_event` and the eBPF kernel
    # filter.
    fs_ignored_pids: set[int] = field(default_factory=set)
    # Path-prefix filter for fs events. Drop any event whose `path` (or
    # `path_secondary`) starts with one of these strings. Targets host-side
    # writes that get attributed to the sandbox cgroup because their author
    # (e.g. CRIU writing `dump.log`) transiently joined the cgroup during
    # a checkpoint operation. CRIU's PID is too short-lived for the
    # PID-based ignore rules to catch reliably (the proc identity is gone
    # by the time the event is processed), but its target paths are well-
    # known per-sandbox host paths the runtime can declare up-front.
    ignored_path_prefixes: tuple[str, ...] = ()
    last_reset_at: datetime | None = None
    process_changed: bool = True
    filesystem_changed: bool = True
    # Set when a write/delete FS event lands on a path that is currently
    # mmap-backed by a live tracked process. Such writes silently invalidate
    # any prior CRIU process checkpoint (build-ID mismatch on restore), so the
    # scheduler must promote the next checkpoint to a full one. Stays True
    # until reset() clears the sandbox state.
    mmap_invalidated: bool = False
    mmap_invalidated_path: str | None = None
    # Snapshot of `(deleted)`-suffixed mmap paths captured by the most recent
    # full process checkpoint. CRIU dumps the content of unlinked-but-mmap'd
    # files inline in the process image, so once a path is in this set the
    # process checkpoint already has its content frozen and subsequent
    # filesystem-only checkpoints can restore safely without on-disk file
    # match. mmap_invalidation only fires for paths NOT in this baseline,
    # which is what prevents one libc rewrite from latching every subsequent
    # checkpoint as full forever.
    acknowledged_deleted_mmaps: frozenset[str] = field(default_factory=frozenset)
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
            "fs_ignored_pids": sorted(self.fs_ignored_pids),
            "ignored_path_prefixes": list(self.ignored_path_prefixes),
            "acknowledged_deleted_mmaps": sorted(self.acknowledged_deleted_mmaps),
            "fs_event_count_since_reset": self.fs_event_count_since_reset,
            "recent_fs_events": list(self.recent_fs_events),
            "live_dirty_entries": [
                dict(entry)
                for _, entry in sorted(self.live_dirty_entries.items(), key=lambda item: item[0])
            ],
            "unreconciled_fs_events": list(self.unreconciled_fs_events),
            "last_error": self.last_error,
        }

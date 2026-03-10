from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from threading import Lock

from .contracts import EBPFEventCollector, SandboxInspector
from .ids import SandboxId
from .models import EBPFEvent, EBPFEventKind, SandboxSnapshot, utc_now

_PROCESS_EVENT_KINDS = {EBPFEventKind.PROCESS_EXEC, EBPFEventKind.PROCESS_EXIT}
_FILESYSTEM_EVENT_KINDS = {EBPFEventKind.FILE_WRITE, EBPFEventKind.FILE_DELETE}


class InMemoryEBPFEventCollector(EBPFEventCollector):
    def __init__(self) -> None:
        self._lock = Lock()
        self._events: list[EBPFEvent] = []

    def record(self, event: EBPFEvent) -> None:
        with self._lock:
            self._events.append(event)

    def poll(self, sandbox_id: SandboxId, since: datetime | None = None) -> list[EBPFEvent]:
        with self._lock:
            events = [x for x in self._events if x.sandbox_id == sandbox_id]
        if since is not None:
            events = [x for x in events if x.observed_at > since]
        events.sort(key=lambda x: x.observed_at)
        return events


class EBPFSandboxInspector(SandboxInspector):
    def __init__(self, collector: EBPFEventCollector | None = None) -> None:
        self._collector = collector or InMemoryEBPFEventCollector()
        self._lock = Lock()
        self._snapshots: dict[SandboxId, SandboxSnapshot] = {}
        self._last_cursor: dict[SandboxId, datetime] = {}

    def upsert_snapshot(self, snapshot: SandboxSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.sandbox_id] = snapshot
            self._last_cursor.setdefault(snapshot.sandbox_id, snapshot.observed_at)

    def mark_changed(
        self,
        sandbox_id: SandboxId,
        *,
        process_changed: bool,
        filesystem_changed: bool,
    ) -> None:
        with self._lock:
            current = self._snapshots[sandbox_id]
            self._snapshots[sandbox_id] = replace(
                current,
                process_changed=process_changed,
                filesystem_changed=filesystem_changed,
                observed_at=utc_now(),
            )

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        with self._lock:
            if sandbox_id not in self._snapshots:
                raise KeyError(f"sandbox snapshot not found: {sandbox_id}")
            base = self._snapshots[sandbox_id]
            cursor = self._last_cursor.get(sandbox_id)

        events = self._collector.poll(sandbox_id, since=cursor)
        process_changed = base.process_changed
        filesystem_changed = base.filesystem_changed
        network_event_count = 0
        observed_at = base.observed_at

        for event in events:
            observed_at = max(observed_at, event.observed_at)
            if event.kind in _PROCESS_EVENT_KINDS:
                process_changed = True
            elif event.kind in _FILESYSTEM_EVENT_KINDS:
                filesystem_changed = True
            else:
                network_event_count += 1

        updated = replace(
            base,
            process_changed=process_changed,
            filesystem_changed=filesystem_changed,
            observed_at=observed_at,
            metadata={**base.metadata, "ebpf_event_count": len(events), "network_event_count": network_event_count},
        )
        with self._lock:
            self._snapshots[sandbox_id] = updated
            if events:
                self._last_cursor[sandbox_id] = events[-1].observed_at
        return updated


class InMemorySandboxInspector(EBPFSandboxInspector):
    pass

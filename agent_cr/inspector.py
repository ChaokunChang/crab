from __future__ import annotations

from dataclasses import replace
from threading import Lock

from .contracts import SandboxInspector
from .ids import SandboxId
from .models import SandboxSnapshot, utc_now


class InMemorySandboxInspector(SandboxInspector):
    def __init__(self) -> None:
        self._lock = Lock()
        self._snapshots: dict[SandboxId, SandboxSnapshot] = {}

    def upsert_snapshot(self, snapshot: SandboxSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.sandbox_id] = snapshot

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
            return self._snapshots[sandbox_id]

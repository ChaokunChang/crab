from __future__ import annotations

from dataclasses import replace
from threading import Lock

from .contracts import SandboxManager
from .ids import SandboxId
from .models import SandboxDescription


class InMemorySandboxManager(SandboxManager):
    def __init__(self) -> None:
        self._lock = Lock()
        self._items: dict[SandboxId, SandboxDescription] = {}

    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        with self._lock:
            sandbox_id = SandboxId.new()
            self._items[sandbox_id] = SandboxDescription(
                sandbox_id=sandbox_id,
                runtime_name=runtime_name,
                status="running",
                metadata=dict(metadata or {}),
            )
            return sandbox_id

    def stop(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="stopped")

    def delete(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            self._items.pop(sandbox_id)

    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        with self._lock:
            return self._items[sandbox_id]

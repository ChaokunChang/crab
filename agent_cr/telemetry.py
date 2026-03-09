from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from .contracts import TelemetrySink


class NoopTelemetrySink(TelemetrySink):
    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        return

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        return


@dataclass
class InMemoryTelemetrySink(TelemetrySink):
    events: list[tuple[str, dict[str, object]]] = field(default_factory=list)
    metrics: list[tuple[str, float, dict[str, object]]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._lock = Lock()

    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        with self._lock:
            self.events.append((name, dict(attributes)))

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            self.metrics.append((name, float(value), dict(attributes or {})))

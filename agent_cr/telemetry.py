from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from .contracts import TelemetrySink


def _normalize_value(value: object) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_value(item) for item in value]
    return value


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


class JsonlTelemetrySink(TelemetrySink):
    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        self._write_record(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "kind": "event",
                "name": name,
                "attributes": _normalize_value(attributes),
            }
        )

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        self._write_record(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "kind": "metric",
                "name": name,
                "value": float(value),
                "attributes": _normalize_value(attributes or {}),
            }
        )

    def _write_record(self, payload: dict[str, object]) -> None:
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True))
                handle.write("\n")


class CompositeTelemetrySink(TelemetrySink):
    def __init__(self, sinks: list[TelemetrySink] | None = None) -> None:
        self._sinks = list(sinks or [])

    def add_sink(self, sink: TelemetrySink) -> None:
        self._sinks.append(sink)

    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        for sink in self._sinks:
            sink.emit_event(name, attributes)

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        for sink in self._sinks:
            sink.emit_metric(name, value, attributes)

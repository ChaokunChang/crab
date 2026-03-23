from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
from threading import Lock
from typing import Any
import uuid

from .contracts import TelemetrySink


DEFAULT_TELEMETRY_DETAIL_LEVEL = "basic"
DEFAULT_TELEMETRY_CAPTURE_COMMAND_OUTPUT = False
DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES = 2048


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


def _truncate_text(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    truncated = encoded[: max(0, max_bytes - 3)].decode("utf-8", errors="ignore")
    return truncated + "..."


def _truncate_value(value: object, *, max_bytes: int) -> object:
    if isinstance(value, str):
        return _truncate_text(value, max_bytes)
    if isinstance(value, dict):
        return {str(key): _truncate_value(item, max_bytes=max_bytes) for key, item in value.items()}
    if isinstance(value, list):
        return [_truncate_value(item, max_bytes=max_bytes) for item in value]
    if isinstance(value, tuple):
        return [_truncate_value(item, max_bytes=max_bytes) for item in value]
    return value


def _telemetry_setting(telemetry: TelemetrySink, name: str, default: object) -> object:
    value = getattr(telemetry, name, None)
    if value is not None:
        return value
    wrapped = getattr(telemetry, "_sink", None)
    if wrapped is not None:
        nested_value = getattr(wrapped, name, None)
        if nested_value is not None:
            return nested_value
    return default


def telemetry_detail_level(telemetry: TelemetrySink) -> str:
    value = _telemetry_setting(telemetry, "detail_level", DEFAULT_TELEMETRY_DETAIL_LEVEL)
    return str(value)


def telemetry_capture_command_output(telemetry: TelemetrySink) -> bool:
    value = _telemetry_setting(
        telemetry,
        "capture_command_output",
        DEFAULT_TELEMETRY_CAPTURE_COMMAND_OUTPUT,
    )
    return bool(value)


def telemetry_max_text_attribute_bytes(telemetry: TelemetrySink) -> int:
    value = _telemetry_setting(
        telemetry,
        "max_text_attribute_bytes",
        DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES,
    )
    try:
        return max(32, int(value))
    except (TypeError, ValueError):
        return DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES


def telemetry_is_detailed(telemetry: TelemetrySink) -> bool:
    return telemetry_detail_level(telemetry).lower() == "detailed"


def telemetry_truncate_value(telemetry: TelemetrySink, value: object) -> object:
    return _truncate_value(value, max_bytes=telemetry_max_text_attribute_bytes(telemetry))


@dataclass
class TelemetryOperation:
    telemetry: TelemetrySink
    name: str
    attributes: dict[str, object]
    started_at: datetime = field(default_factory=lambda: datetime.now().astimezone())
    _started_ns: int = field(default_factory=time.perf_counter_ns, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.attributes.setdefault("op_id", uuid.uuid4().hex)
        self.telemetry.emit_event(f"{self.name}.start", dict(self.attributes))

    def finish(
        self,
        *,
        status: str = "succeeded",
        attributes: dict[str, object] | None = None,
        metric_attributes: dict[str, object] | None = None,
    ) -> float:
        if self._finished:
            return 0.0
        self._finished = True
        duration_ms = max(0.0, (time.perf_counter_ns() - self._started_ns) / 1_000_000.0)
        finish_attributes = dict(self.attributes)
        finish_attributes["status"] = status
        if attributes:
            finish_attributes.update(attributes)
        self.telemetry.emit_event(f"{self.name}.finish", finish_attributes)
        metric_payload = dict(finish_attributes)
        if metric_attributes:
            metric_payload.update(metric_attributes)
        self.telemetry.emit_metric(f"{self.name}.duration_ms", duration_ms, metric_payload)
        return duration_ms


def start_operation(
    telemetry: TelemetrySink,
    name: str,
    attributes: dict[str, object] | None = None,
) -> TelemetryOperation:
    return TelemetryOperation(telemetry=telemetry, name=name, attributes=dict(attributes or {}))


class ConfiguredTelemetrySink(TelemetrySink):
    def __init__(
        self,
        sink: TelemetrySink,
        *,
        default_attributes: dict[str, object] | None = None,
        detail_level: str = DEFAULT_TELEMETRY_DETAIL_LEVEL,
        capture_command_output: bool = DEFAULT_TELEMETRY_CAPTURE_COMMAND_OUTPUT,
        max_text_attribute_bytes: int = DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES,
    ) -> None:
        self._sink = sink
        self._default_attributes = dict(default_attributes or {})
        self.detail_level = detail_level
        self.capture_command_output = capture_command_output
        self.max_text_attribute_bytes = max_text_attribute_bytes

    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        payload = {**self._default_attributes, **attributes}
        payload = telemetry_truncate_value(self, payload)
        self._sink.emit_event(name, payload)

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        payload = {**self._default_attributes, **dict(attributes or {})}
        payload = telemetry_truncate_value(self, payload)
        self._sink.emit_metric(name, value, payload)

    def __getattr__(self, name: str):
        return getattr(self._sink, name)


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

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
import time
from threading import Event, Lock, Thread
from typing import Any
import uuid

from .config import TelemetryConfig
from .contracts import TelemetrySink
from .json_codec import get_json_codec


DEFAULT_TELEMETRY_DETAIL_LEVEL = "basic"
DEFAULT_TELEMETRY_CAPTURE_COMMAND_OUTPUT = False
DEFAULT_TELEMETRY_MAX_TEXT_ATTRIBUTE_BYTES = 2048
DEFAULT_TELEMETRY_WRITER_MODE = "async"
DEFAULT_TELEMETRY_QUEUE_CAPACITY = 16384
DEFAULT_TELEMETRY_BATCH_MAX_RECORDS = 256
DEFAULT_TELEMETRY_FLUSH_INTERVAL_MS = 50
DEFAULT_TELEMETRY_OVERFLOW_POLICY = "drop_new"
DEFAULT_TELEMETRY_SERIALIZER = "auto"


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
    _started_ns: int = field(default=0, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.attributes.setdefault("op_id", uuid.uuid4().hex)
        self.telemetry.emit_event(f"{self.name}.start", dict(self.attributes))
        self._started_ns = time.perf_counter_ns()

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
        metric_payload = dict(finish_attributes)
        if metric_attributes:
            metric_payload.update(metric_attributes)
        self.telemetry.emit_event(f"{self.name}.finish", finish_attributes)
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
        self._default_attributes = _normalize_value(dict(default_attributes or {}))
        self.detail_level = detail_level
        self.capture_command_output = capture_command_output
        self.max_text_attribute_bytes = max_text_attribute_bytes

    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        if (
            not telemetry_is_detailed(self)
            and "op_id" in attributes
            and (name.endswith(".start") or name.endswith(".finish"))
        ):
            return
        self._sink.emit_event(name, self._prepare_payload(attributes))

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        self._sink.emit_metric(name, value, self._prepare_payload(attributes or {}))

    def flush(self) -> None:
        self._sink.flush()

    def close(self) -> None:
        self._sink.close()

    def __getattr__(self, name: str):
        return getattr(self._sink, name)

    def _prepare_payload(self, attributes: dict[str, object]) -> dict[str, object]:
        if self._default_attributes:
            payload = dict(self._default_attributes)
            payload.update(attributes)
        else:
            payload = dict(attributes)
        return telemetry_truncate_value(self, payload)


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
    def __init__(self, path: Path, *, serializer: str = DEFAULT_TELEMETRY_SERIALIZER) -> None:
        self._path = path
        self._lock = Lock()
        self._codec = get_json_codec(serializer)
        self.serializer = self._codec.serializer
        self._fd: int | None = None
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        return self._path

    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        self.write_records(
            [
                {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "kind": "event",
                    "name": name,
                    "attributes": _normalize_value(attributes),
                }
            ]
        )

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        self.write_records(
            [
                {
                    "timestamp": datetime.now().astimezone().isoformat(),
                    "kind": "metric",
                    "name": name,
                    "value": float(value),
                    "attributes": _normalize_value(attributes or {}),
                }
            ]
        )

    def write_records(self, records: list[dict[str, object]]) -> None:
        if not records:
            return
        payload = bytearray()
        for record in records:
            payload.extend(self._codec.dumps_bytes(record))
            payload.extend(b"\n")
        with self._lock:
            fd = self._ensure_fd()
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(fd, remaining)
                    remaining = remaining[written:]
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)

    def flush(self) -> None:
        with self._lock:
            if self._fd is None:
                return
            os.fsync(self._fd)

    def close(self) -> None:
        with self._lock:
            if self._fd is None:
                return
            os.close(self._fd)
            self._fd = None

    def _ensure_fd(self) -> int:
        if self._fd is None:
            self._fd = os.open(self._path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        return self._fd


class AsyncJsonlTelemetrySink(TelemetrySink):
    def __init__(
        self,
        sink: JsonlTelemetrySink,
        *,
        queue_capacity: int = DEFAULT_TELEMETRY_QUEUE_CAPACITY,
        batch_max_records: int = DEFAULT_TELEMETRY_BATCH_MAX_RECORDS,
        flush_interval_ms: int = DEFAULT_TELEMETRY_FLUSH_INTERVAL_MS,
        overflow_policy: str = DEFAULT_TELEMETRY_OVERFLOW_POLICY,
    ) -> None:
        self._sink = sink
        self._queue: Queue[dict[str, object]] = Queue(maxsize=max(1, int(queue_capacity)))
        self._batch_max_records = max(1, int(batch_max_records))
        self._flush_interval_s = max(0.001, float(flush_interval_ms) / 1000.0)
        self._overflow_policy = str(overflow_policy)
        self._stop_event = Event()
        self._state_lock = Lock()
        self._drop_lock = Lock()
        self._closed = False
        self._dropped_events = 0
        self._dropped_metrics = 0
        self._writer = Thread(target=self._run, name="crab-telemetry-jsonl", daemon=True)
        self._writer.start()

    @property
    def path(self) -> Path:
        return self._sink.path

    @property
    def serializer(self) -> str:
        return self._sink.serializer

    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        self._enqueue(
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
        self._enqueue(
            {
                "timestamp": datetime.now().astimezone().isoformat(),
                "kind": "metric",
                "name": name,
                "value": float(value),
                "attributes": _normalize_value(attributes or {}),
            }
        )

    def flush(self) -> None:
        self._queue.join()
        self._sink.flush()

    def close(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
        self._stop_event.set()
        self._queue.join()
        self._writer.join(timeout=max(1.0, self._flush_interval_s * 10.0))
        self._emit_drop_summary()
        self._sink.flush()
        self._sink.close()

    def _enqueue(self, record: dict[str, object]) -> None:
        with self._state_lock:
            if self._closed:
                return
        if self._overflow_policy == "block":
            self._queue.put(record)
            return
        try:
            self._queue.put_nowait(record)
        except Full:
            with self._drop_lock:
                if record.get("kind") == "metric":
                    self._dropped_metrics += 1
                else:
                    self._dropped_events += 1

    def _emit_drop_summary(self) -> None:
        with self._drop_lock:
            dropped_events = self._dropped_events
            dropped_metrics = self._dropped_metrics
            self._dropped_events = 0
            self._dropped_metrics = 0
        if dropped_events <= 0 and dropped_metrics <= 0:
            return
        self._sink.emit_event(
            "telemetry.writer.dropped",
            {
                "writer_mode": "async",
                "overflow_policy": self._overflow_policy,
                "path": str(self._sink.path),
                "dropped_event_records": dropped_events,
                "dropped_metric_records": dropped_metrics,
            },
        )

    def _run(self) -> None:
        while True:
            batch: list[dict[str, object]] = []
            try:
                batch.append(self._queue.get(timeout=self._flush_interval_s))
            except Empty:
                if self._stop_event.is_set():
                    break
                continue
            while len(batch) < self._batch_max_records:
                try:
                    batch.append(self._queue.get_nowait())
                except Empty:
                    break
            try:
                self._sink.write_records(batch)
            finally:
                for _ in batch:
                    self._queue.task_done()


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

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self._sinks:
            sink.close()


def build_configured_telemetry_sink(
    config: TelemetryConfig,
    *,
    default_attributes: dict[str, object] | None = None,
    keep_in_memory_fallback: bool = False,
) -> TelemetrySink:
    if not config.enabled:
        return NoopTelemetrySink()

    sinks: list[TelemetrySink] = []
    if config.resolve_keep_in_memory_copy(fallback=keep_in_memory_fallback):
        sinks.append(InMemoryTelemetrySink())
    if config.jsonl_path is not None:
        jsonl_sink = JsonlTelemetrySink(Path(config.jsonl_path), serializer=config.serializer)
        if config.writer_mode == "async":
            sinks.append(
                AsyncJsonlTelemetrySink(
                    jsonl_sink,
                    queue_capacity=config.queue_capacity,
                    batch_max_records=config.batch_max_records,
                    flush_interval_ms=config.flush_interval_ms,
                    overflow_policy=config.overflow_policy,
                )
            )
        else:
            sinks.append(jsonl_sink)
    if not sinks:
        sinks.append(NoopTelemetrySink())
    base_sink = sinks[0] if len(sinks) == 1 else CompositeTelemetrySink(sinks)
    return ConfiguredTelemetrySink(
        base_sink,
        default_attributes=default_attributes,
        detail_level=config.detail_level,
        capture_command_output=config.capture_command_output,
        max_text_attribute_bytes=config.max_text_attribute_bytes,
    )

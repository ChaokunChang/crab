from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import TelemetryRecordRef, TimelineItem, ToolCallSummary


@dataclass(frozen=True)
class ParsedTelemetry:
    records_by_sandbox: dict[str, list[TelemetryRecordRef]]
    records_by_task: dict[str, list[TelemetryRecordRef]]
    timeline_by_sandbox: dict[str, list[TimelineItem]]
    tool_calls_by_sandbox: dict[str, list[ToolCallSummary]]


def _record_ids(attributes: dict[str, Any]) -> tuple[str | None, str | None]:
    sandbox_id = attributes.get("sandbox_id")
    task_id = attributes.get("task_id")
    return (
        str(sandbox_id) if sandbox_id not in (None, "") else None,
        str(task_id) if task_id not in (None, "") else None,
    )


def _timeline_label(name: str, attributes: dict[str, Any]) -> str | None:
    lowered = name.lower()
    event_type = str(attributes.get("event_type", "")).strip().lower()
    if "verify" in lowered:
        return "verification"
    if "checkpoint" in lowered:
        return "checkpoint"
    if "restore" in lowered:
        return "restore"
    if "recovery" in lowered:
        return "recovery"
    if event_type == "fault" or lowered.endswith(".fault") or "fault" in lowered:
        return "fault injected"
    if "request" in lowered or attributes.get("request_id"):
        return "request"
    return None


def _command_preview(attributes: dict[str, Any], *, limit: int) -> str:
    for key in ("command", "args", "argv", "operation"):
        raw = attributes.get(key)
        if isinstance(raw, list):
            value = " ".join(str(item) for item in raw)
            if len(value) <= limit:
                return value
            return f"{value[: max(0, limit - 3)].rstrip()}..."
        if isinstance(raw, str) and raw.strip():
            value = raw.strip()
            if len(value) <= limit:
                return value
            return f"{value[: max(0, limit - 3)].rstrip()}..."
    return ""


def parse_telemetry(
    telemetry_path: Path | None,
    *,
    command_preview_chars: int = 240,
) -> ParsedTelemetry:
    if telemetry_path is None or not telemetry_path.exists():
        return ParsedTelemetry(
            records_by_sandbox={},
            records_by_task={},
            timeline_by_sandbox={},
            tool_calls_by_sandbox={},
        )
    records_by_sandbox: dict[str, list[TelemetryRecordRef]] = defaultdict(list)
    records_by_task: dict[str, list[TelemetryRecordRef]] = defaultdict(list)
    timeline_by_sandbox: dict[str, list[TimelineItem]] = defaultdict(list)
    tool_calls_by_sandbox: dict[str, list[ToolCallSummary]] = defaultdict(list)
    with telemetry_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                continue
            attributes = payload.get("attributes", {})
            if not isinstance(attributes, dict):
                attributes = {}
            sandbox_id, task_id = _record_ids(attributes)
            record = TelemetryRecordRef(
                line_number=line_number,
                timestamp=str(payload.get("timestamp", "")),
                kind=str(payload.get("kind", "")),
                name=str(payload.get("name", "")),
                value=float(payload["value"]) if payload.get("kind") == "metric" and isinstance(payload.get("value"), (int, float)) else None,
                sandbox_id=sandbox_id,
                task_id=task_id,
                checkpoint_id=None if attributes.get("checkpoint_id") in (None, "") else str(attributes.get("checkpoint_id")),
                request_id=None if attributes.get("request_id") in (None, "") else str(attributes.get("request_id")),
                job_id=None if attributes.get("job_id") in (None, "") else str(attributes.get("job_id")),
                attributes=dict(attributes),
                raw=payload,
            )
            if sandbox_id is not None:
                records_by_sandbox[sandbox_id].append(record)
                label = _timeline_label(record.name, attributes)
                if label is not None:
                    timeline_by_sandbox[sandbox_id].append(
                        TimelineItem(
                            source="telemetry",
                            timestamp=record.timestamp,
                            label=label,
                            detail=record.name,
                            sandbox_id=sandbox_id,
                            task_id=task_id,
                            checkpoint_id=record.checkpoint_id,
                            request_id=record.request_id,
                            evidence_ref=f"telemetry:{line_number}",
                        )
                    )
                preview = _command_preview(attributes, limit=command_preview_chars)
                if preview and (
                    "command" in record.name.lower()
                    or attributes.get("component") == "runtime"
                    or attributes.get("operation")
                ):
                    tool_calls_by_sandbox[sandbox_id].append(
                        ToolCallSummary(
                            source="telemetry",
                            timestamp=record.timestamp,
                            tool_name=str(attributes.get("operation", record.name)),
                            description=record.name,
                            arguments_preview=preview,
                            status=None if attributes.get("status") in (None, "") else str(attributes.get("status")),
                            duration_ms=record.value if record.kind == "metric" and "duration" in record.name.lower() else None,
                            has_error_indicators=bool(
                                attributes.get("status") == "failed"
                                or attributes.get("success") is False
                                or attributes.get("returncode") not in (None, "", 0)
                                or str(attributes.get("stderr", "")).strip()
                            ),
                            result_summary=(
                                f"returncode={attributes.get('returncode')}"
                                if attributes.get("returncode") not in (None, "")
                                else (str(attributes.get("status")) if attributes.get("status") not in (None, "") else None)
                            ),
                            raw_arguments=dict(attributes),
                        )
                    )
            if task_id is not None:
                records_by_task[task_id].append(record)
    for values in timeline_by_sandbox.values():
        values.sort(key=lambda item: (item.timestamp or "", item.label, item.detail))
    return ParsedTelemetry(
        records_by_sandbox=dict(records_by_sandbox),
        records_by_task=dict(records_by_task),
        timeline_by_sandbox={key: value for key, value in timeline_by_sandbox.items()},
        tool_calls_by_sandbox={key: value for key, value in tool_calls_by_sandbox.items()},
    )

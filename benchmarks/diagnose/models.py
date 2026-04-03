from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DiagnoseRunContext:
    config_path: Path
    scenario: str
    mode: str
    provider: str
    agent: str
    llm_service: str | None
    task_dataset_path: Path | None
    log_path: Path | None
    csv_path: Path | None
    telemetry_path: Path | None
    configured_benchmark_root: Path | None
    actual_benchmark_root: Path | None
    inferred_run_roots: tuple[str, ...]


@dataclass(frozen=True)
class DatasetTaskInfo:
    dataset_index: int
    task_id: str
    agent_type: str
    llm_service_type: str | None
    trace_path: Path | None
    trace_response_count: int | None
    trace_malformed_line_count: int | None
    task_root: Path | None
    service_name: str | None
    prompt_preview: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class CsvSandboxRow:
    row_index: int
    sandbox_id: str
    task_id: str
    iteration: int | None
    classification: str
    success_ratio: float | None
    verification_status: str | None
    verification_exit_code: int | None
    task_error: str | None
    event_type: str | None
    raw: dict[str, str]


@dataclass(frozen=True)
class MissingTaskRecord:
    dataset_index: int
    task_id: str
    reason: str
    occurrences_expected: int
    occurrences_observed: int
    sandbox_id: str | None = None
    trace_path: Path | None = None


@dataclass(frozen=True)
class TelemetryRecordRef:
    line_number: int
    timestamp: str
    kind: str
    name: str
    value: float | None
    sandbox_id: str | None
    task_id: str | None
    checkpoint_id: str | None
    request_id: str | None
    job_id: str | None
    attributes: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class TimelineItem:
    source: str
    timestamp: str | None
    label: str
    detail: str
    sandbox_id: str | None = None
    task_id: str | None = None
    checkpoint_id: str | None = None
    request_id: str | None = None
    evidence_ref: str | None = None


@dataclass(frozen=True)
class ToolCallSummary:
    source: str
    timestamp: str | None
    tool_name: str
    description: str
    arguments_preview: str
    call_index: int | None = None
    status: str | None = None
    exit_code: int | None = None
    duration_ms: float | None = None
    is_dummy: bool = False
    matched_trace_index: int | None = None
    has_error_indicators: bool = False
    result_summary: str | None = None
    raw_arguments: dict[str, Any] | None = None
    raw_result_preview: str | None = None
    observed_key_events: tuple[TimelineItem, ...] = ()


@dataclass(frozen=True)
class Finding:
    severity: str
    title: str
    summary: str
    category: str
    evidence_refs: tuple[str, ...] = ()


@dataclass
class SandboxDiagnosis:
    sandbox_id: str
    task_id: str
    status: str
    dataset_tasks: list[DatasetTaskInfo]
    csv_rows: list[CsvSandboxRow]
    findings: list[Finding]
    timeline: list[TimelineItem]
    telemetry_records: list[TelemetryRecordRef]
    log_lines: list[str]
    log_excerpt: list[str]
    tool_calls: list[ToolCallSummary]
    trace_tool_calls: list[ToolCallSummary]
    tool_alignment_summary: dict[str, Any]
    replay_trace_summary: dict[str, Any]
    session_summary: dict[str, Any]
    raw_evidence: list[str]
    notes: list[str]


@dataclass
class RunDiagnosis:
    context: DiagnoseRunContext
    dataset_tasks: list[DatasetTaskInfo]
    csv_rows: list[CsvSandboxRow]
    missing_tasks: list[MissingTaskRecord]
    failed_sandboxes: list[str]
    sandboxes: dict[str, SandboxDiagnosis]
    run_findings: list[Finding]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if is_dataclass(value):
        return _jsonable(asdict(value))
    return value


def to_jsonable(value: Any) -> Any:
    return _jsonable(value)

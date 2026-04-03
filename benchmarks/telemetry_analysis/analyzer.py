from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable


DEFAULT_TOP_K = 25

_PREFERRED_METRIC_SOURCES: dict[str, tuple[str, ...]] = {
    "llm.interceptor_total_ms": ("llm.interceptor_total_ms", "llm.request_total_ms"),
    "executor.job.duration_ms": ("executor.job.duration_ms", "executor.job_duration_ms"),
    "checkpoint.flow.duration_ms": ("checkpoint.flow.duration_ms", "checkpoint.total_ms"),
    "checkpoint.process.duration_ms": ("checkpoint.process.duration_ms", "checkpoint.process_ms"),
    "checkpoint.filesystem.duration_ms": ("checkpoint.filesystem.duration_ms", "checkpoint.filesystem_ms"),
    "checkpoint.persist_artifacts.duration_ms": (
        "checkpoint.persist_artifacts.duration_ms",
        "checkpoint.persist_artifacts_ms",
    ),
    "checkpoint.persist_manifest.duration_ms": (
        "checkpoint.persist_manifest.duration_ms",
        "checkpoint.persist_manifest_ms",
    ),
    "restore.resolve_manifest.duration_ms": (
        "restore.resolve_manifest.duration_ms",
        "restore.resolve_manifest_ms",
    ),
}

_DEFAULT_BREAKDOWN_METRICS = (
    "benchmark.task.queue_wait_ms",
    "benchmark.task.run.duration_ms",
    "benchmark.task.duration_ms",
    "benchmark.task.verify.duration_ms",
    "benchmark.checkpoint_ms",
    "benchmark.restore_ms",
    "benchmark.recovery_ms",
    "benchmark.readiness_ms",
    "benchmark.end_to_end_recovery_ms",
    "benchmark.workload_resume_ms",
    "benchmark.lost_actions",
    "benchmark.task.success_ratio",
)

_LLM_BREAKDOWN_METRICS = (
    "llm.service.request.duration_ms",
    "interceptor.request.forward.duration_ms",
    "llm.interceptor_total_ms",
    "llm.gate_wait_ms",
    "llm.agentcr_delay_ms",
)

_OVERHEAD_ANALYSIS_METRICS = (
    "llm.gate_wait_ms",
    "llm.agentcr_delay_ms",
)

_TURN_ANALYSIS_RESPONSE_METRIC_SOURCES = (
    "llm.interceptor_total_ms",
    "llm.request_total_ms",
)
_TURN_ANALYSIS_PURE_LLM_METRIC = "llm.service.request.duration_ms"
_TURN_ANALYSIS_METRIC_ORDER = (
    "llm_response_time",
    "pure_llm_time",
    "action_time",
    "turn_time",
)
_CHECKPOINT_BREAKDOWN_METRICS = (
    "checkpoint.flow.duration_ms",
    "checkpoint.process.duration_ms",
    "checkpoint.filesystem.duration_ms",
    "checkpoint.persist_artifacts.duration_ms",
    "checkpoint.persist_manifest.duration_ms",
    "restore.flow.duration_ms",
    "restore.resolve_manifest.duration_ms",
    "restore.filesystem.duration_ms",
    "restore.process.duration_ms",
)

_CHECKPOINT_SIZE_METRICS = {
    "checkpoint.process.size_bytes",
    "checkpoint.process.file_count",
    "checkpoint.filesystem.written_bytes",
    "checkpoint.filesystem.used_bytes",
    "checkpoint.estimated_io_bytes",
}

_RESTORE_EXTRA_METRICS = {
    "restore.source_gap.turns",
    "restore.source_gap.ms",
    "restore.estimated_io_bytes",
}

_RESOURCE_HOST_PREFIX = "resource.host."
_RESOURCE_SANDBOX_PREFIX = "resource.sandbox."

_EXCLUDED_FROM_TOTAL_TIME = {
    "sandbox.command_duration_ms",
}

_TASK_SUMMARY_SORT_METRICS = (
    "benchmark.task.duration_ms",
    "benchmark.task.run.duration_ms",
    "benchmark.task.queue_wait_ms",
)


def _record_timestamp(payload: dict[str, Any]) -> datetime | None:
    raw = payload.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _maybe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, q)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def _infer_category(metric_name: str, component: str) -> str:
    if metric_name.startswith("benchmark."):
        return "benchmark"
    if metric_name.startswith("llm.") or metric_name.startswith("interceptor."):
        return "llm_path"
    if metric_name.startswith("checkpoint.") or metric_name.startswith("restore.") or metric_name.startswith("recovery."):
        return "checkpoint_restore"
    if metric_name.startswith("resource."):
        return "resource"
    if metric_name.startswith("scheduler.") or metric_name.startswith("executor."):
        return "scheduler_executor"
    if metric_name.startswith("sandbox.") or metric_name.startswith("image."):
        return "runtime"
    if component:
        return component
    return "other"


def _duration_minutes(started_at: datetime | None, finished_at: datetime | None) -> float:
    if started_at is None or finished_at is None:
        return 0.0
    duration = max(0.0, (finished_at - started_at).total_seconds())
    if duration <= 0:
        return 0.0
    return duration / 60.0


def _offset_seconds(timestamp: datetime | None, started_at: datetime | None) -> float:
    if timestamp is None or started_at is None:
        return 0.0
    return max(0.0, (timestamp - started_at).total_seconds())


def _benchmark_phase_name(name: str, attributes: dict[str, Any]) -> str | None:
    if not name.startswith("benchmark.phase."):
        return None
    phase_scope = _maybe_str(attributes.get("phase_scope"))
    if phase_scope and phase_scope != "run":
        return None
    phase = _maybe_str(attributes.get("phase"))
    if phase:
        return phase
    suffix = name[len("benchmark.phase.") :]
    for marker in (".configured_max_workers", ".duration_ms", ".start", ".finish"):
        if suffix.endswith(marker):
            phase = suffix[: -len(marker)]
            break
    if not phase or phase.endswith(".item"):
        return None
    return phase


def _is_detail_record(
    *,
    name: str,
    timestamp: datetime | None,
    detail_window: tuple[datetime, datetime] | None,
) -> bool:
    if name.startswith("benchmark.phase."):
        return False
    if detail_window is None:
        return True
    if timestamp is None:
        return False
    started_at, finished_at = detail_window
    return started_at <= timestamp <= finished_at


def _format_ts(timestamp: datetime | None) -> str:
    return "" if timestamp is None else timestamp.isoformat()


@dataclass(frozen=True)
class SlowRecord:
    metric_name: str
    source_metric_name: str
    timestamp: str
    value_ms: float
    component: str
    category: str
    sandbox_id: str
    task_id: str
    request_id: str
    checkpoint_id: str
    job_id: str
    status: str
    operation: str


@dataclass(frozen=True)
class MetricSummary:
    metric_name: str
    source_metric_name: str
    category: str
    component: str
    count: int
    total_ms: float
    mean_ms: float
    min_ms: float
    p25_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    success_count: int
    failure_count: int
    unique_sandboxes: int
    unique_tasks: int
    top_records: list[SlowRecord]


@dataclass(frozen=True)
class TaskSummary:
    sandbox_id: str
    task_id: str
    agent_type: str
    llm_service_type: str
    metrics: dict[str, float]


@dataclass(frozen=True)
class LifecycleSummary:
    operation_name: str
    start_count: int
    finish_count: int
    missing_finish_count: int
    finish_status_counts: dict[str, int]


@dataclass(frozen=True)
class TimeSeriesPoint:
    timestamp: str
    offset_seconds: float
    value: float


@dataclass(frozen=True)
class LoadSeriesPoint:
    timestamp: str
    offset_seconds: float
    active_jobs: int
    active_process_jobs: int = 0
    active_filesystem_jobs: int = 0
    active_estimated_io_bytes: float = 0.0


@dataclass(frozen=True)
class OperationTaskAnalysis:
    task_id: str
    sandbox_id: str
    total_count: int
    success_count: int
    fail_count: int
    success_rate: float
    per_minute_rate: float
    mean_estimated_io_bytes: float
    skip_count: int = 0
    mean_source_gap_turns: float = 0.0
    mean_source_gap_ms: float = 0.0


@dataclass(frozen=True)
class CheckpointAnalysisSummary:
    total_count: int
    success_count: int
    fail_count: int
    skip_count: int
    fail_rate: float
    skip_rate: float
    overall_rate_per_minute: float
    process_frequency_per_minute: float
    filesystem_frequency_per_minute: float
    scope_counts: dict[str, int]
    scope_rates: dict[str, float]
    filesystem_only_to_full_ratio: float
    promoted_process_checkpoint_count: int
    total_process_size_bytes: float
    total_filesystem_written_bytes: float
    total_estimated_io_bytes: float
    mean_process_size_bytes: float
    mean_filesystem_written_bytes: float
    mean_estimated_io_bytes: float
    p95_process_size_bytes: float
    p95_filesystem_written_bytes: float
    per_task: list[OperationTaskAnalysis]
    load_over_time: list[LoadSeriesPoint]
    latency_over_time: dict[str, list[TimeSeriesPoint]]


@dataclass(frozen=True)
class RestoreAnalysisSummary:
    total_count: int
    success_count: int
    fail_count: int
    skip_count: int
    fail_rate: float
    skip_rate: float
    overall_rate_per_minute: float
    mixed_source_count: int
    mixed_source_rate: float
    mean_estimated_io_bytes: float
    mean_source_gap_turns: float
    p95_source_gap_turns: float
    max_source_gap_turns: float
    mean_source_gap_ms: float
    p95_source_gap_ms: float
    max_source_gap_ms: float
    per_task: list[OperationTaskAnalysis]
    load_over_time: list[LoadSeriesPoint]
    latency_over_time: dict[str, list[TimeSeriesPoint]]


@dataclass(frozen=True)
class ResourceMetricSummary:
    metric_name: str
    sandbox_id: str
    sample_count: int
    mean_value: float
    max_value: float


@dataclass(frozen=True)
class ResourceAnalysisSummary:
    host_metrics: list[ResourceMetricSummary]
    sandbox_metrics: list[ResourceMetricSummary]
    host_series: dict[str, list[TimeSeriesPoint]]
    sandbox_series: dict[str, dict[str, list[TimeSeriesPoint]]]
    coverage: dict[str, bool]


@dataclass(frozen=True)
class OverheadAnalysisSummary:
    metrics: list[MetricSummary]
    time_series: dict[str, list[TimeSeriesPoint]]
    cdf_series: dict[str, list[CDFPoint]] = field(default_factory=dict)


@dataclass(frozen=True)
class CDFPoint:
    value: float
    cumulative_probability: float


@dataclass(frozen=True)
class TurnMetricSummary:
    metric_name: str
    count: int
    mean_ms: float
    min_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    request_kind: str = ""


@dataclass(frozen=True)
class TurnAnalysisSummary:
    metrics: list[TurnMetricSummary]
    cdf_series: dict[str, list[CDFPoint]]
    time_series: dict[str, list[TimeSeriesPoint]]
    metrics_by_request_kind: dict[str, list[TurnMetricSummary]] = field(default_factory=dict)
    cdf_series_by_request_kind: dict[str, dict[str, list[CDFPoint]]] = field(default_factory=dict)
    time_series_by_request_kind: dict[str, dict[str, list[TimeSeriesPoint]]] = field(default_factory=dict)


@dataclass(frozen=True)
class PhaseSummary:
    phase: str
    started_at: str
    finished_at: str
    duration_ms: float
    configured_max_workers: int | None
    sandbox_count: int | None
    status: str


@dataclass(frozen=True)
class TelemetryAnalysis:
    input_path: str
    run_id: str
    total_records: int
    total_events: int
    total_metrics: int
    started_at: str
    finished_at: str
    distinct_sandboxes: int
    distinct_tasks: int
    distinct_requests: int
    distinct_checkpoints: int
    distinct_jobs: int
    event_name_counts: dict[str, int]
    metric_name_counts: dict[str, int]
    operation_summaries: list[MetricSummary]
    top_total_time_operations: list[MetricSummary]
    top_invocation_operations: list[MetricSummary]
    top_tail_latency_operations: list[MetricSummary]
    task_summaries: list[TaskSummary]
    lifecycle_summaries: list[LifecycleSummary]
    lifecycle_gaps: list[LifecycleSummary]
    slowest_records: list[SlowRecord]
    llm_breakdown: dict[str, float]
    checkpoint_breakdown: dict[str, float]
    phase_overview: list[PhaseSummary]
    detail_scope_name: str
    detail_total_records: int
    detail_total_events: int
    detail_total_metrics: int
    detail_started_at: str
    detail_finished_at: str
    checkpoint_analysis: CheckpointAnalysisSummary | None = None
    overhead_analysis: OverheadAnalysisSummary | None = None
    turn_analysis: TurnAnalysisSummary | None = None
    restore_analysis: RestoreAnalysisSummary | None = None
    resource_analysis: ResourceAnalysisSummary | None = None
    exclude_failed_tasks: bool = False
    excluded_sandbox_task_pairs: list[tuple[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "run_id": self.run_id,
            "total_records": self.total_records,
            "total_events": self.total_events,
            "total_metrics": self.total_metrics,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "distinct_sandboxes": self.distinct_sandboxes,
            "distinct_tasks": self.distinct_tasks,
            "distinct_requests": self.distinct_requests,
            "distinct_checkpoints": self.distinct_checkpoints,
            "distinct_jobs": self.distinct_jobs,
            "event_name_counts": dict(self.event_name_counts),
            "metric_name_counts": dict(self.metric_name_counts),
            "operation_summaries": [asdict(item) for item in self.operation_summaries],
            "top_total_time_operations": [asdict(item) for item in self.top_total_time_operations],
            "top_invocation_operations": [asdict(item) for item in self.top_invocation_operations],
            "top_tail_latency_operations": [asdict(item) for item in self.top_tail_latency_operations],
            "task_summaries": [asdict(item) for item in self.task_summaries],
            "lifecycle_summaries": [asdict(item) for item in self.lifecycle_summaries],
            "lifecycle_gaps": [asdict(item) for item in self.lifecycle_gaps],
            "slowest_records": [asdict(item) for item in self.slowest_records],
            "llm_breakdown": dict(self.llm_breakdown),
            "checkpoint_breakdown": dict(self.checkpoint_breakdown),
            "phase_overview": [asdict(item) for item in self.phase_overview],
            "detail_scope_name": self.detail_scope_name,
            "detail_total_records": self.detail_total_records,
            "detail_total_events": self.detail_total_events,
            "detail_total_metrics": self.detail_total_metrics,
            "detail_started_at": self.detail_started_at,
            "detail_finished_at": self.detail_finished_at,
            "checkpoint_analysis": None if self.checkpoint_analysis is None else asdict(self.checkpoint_analysis),
            "overhead_analysis": None if self.overhead_analysis is None else asdict(self.overhead_analysis),
            "turn_analysis": None if self.turn_analysis is None else asdict(self.turn_analysis),
            "restore_analysis": None if self.restore_analysis is None else asdict(self.restore_analysis),
            "resource_analysis": None if self.resource_analysis is None else asdict(self.resource_analysis),
            "exclude_failed_tasks": self.exclude_failed_tasks,
            "excluded_sandbox_task_pairs": [
                {"sandbox_id": sid, "task_id": tid}
                for sid, tid in self.excluded_sandbox_task_pairs
            ],
        }


@dataclass
class _MetricAccumulator:
    name: str
    values: list[float] = field(default_factory=list)
    total: float = 0.0
    minimum: float = math.inf
    maximum: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    component_counts: Counter[str] = field(default_factory=Counter)
    sandbox_ids: set[str] = field(default_factory=set)
    task_ids: set[str] = field(default_factory=set)
    top_records: list[SlowRecord] = field(default_factory=list)

    def add(self, value: float, *, payload: dict[str, Any], attributes: dict[str, Any], top_k: int) -> None:
        self.values.append(value)
        self.total += value
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)
        component = _maybe_str(attributes.get("component"))
        if component:
            self.component_counts[component] += 1
        sandbox_id = _maybe_str(attributes.get("sandbox_id"))
        task_id = _maybe_str(attributes.get("task_id"))
        if sandbox_id:
            self.sandbox_ids.add(sandbox_id)
        if task_id:
            self.task_ids.add(task_id)
        status = _maybe_str(attributes.get("status")).lower()
        if status == "failed":
            self.failure_count += 1
        elif status:
            self.success_count += 1

        record = SlowRecord(
            metric_name=self.name,
            source_metric_name=self.name,
            timestamp=_maybe_str(payload.get("timestamp")),
            value_ms=value,
            component=component,
            category=_infer_category(self.name, component),
            sandbox_id=sandbox_id,
            task_id=task_id,
            request_id=_maybe_str(attributes.get("request_id")),
            checkpoint_id=_maybe_str(attributes.get("checkpoint_id")),
            job_id=_maybe_str(attributes.get("job_id")),
            status=_maybe_str(attributes.get("status")),
            operation=_maybe_str(attributes.get("operation")),
        )
        self.top_records.append(record)
        self.top_records.sort(key=lambda item: item.value_ms, reverse=True)
        if len(self.top_records) > top_k:
            del self.top_records[top_k:]

    def build_summary(self, *, canonical_name: str, source_metric_name: str) -> MetricSummary:
        sorted_values = sorted(self.values)
        component = self.component_counts.most_common(1)[0][0] if self.component_counts else ""
        count = len(sorted_values)
        mean_ms = (self.total / count) if count else 0.0
        return MetricSummary(
            metric_name=canonical_name,
            source_metric_name=source_metric_name,
            category=_infer_category(canonical_name, component),
            component=component,
            count=count,
            total_ms=self.total,
            mean_ms=mean_ms,
            min_ms=0.0 if not count else self.minimum,
            p25_ms=_percentile(sorted_values, 0.25),
            p50_ms=_percentile(sorted_values, 0.50),
            p90_ms=_percentile(sorted_values, 0.90),
            p95_ms=_percentile(sorted_values, 0.95),
            p99_ms=_percentile(sorted_values, 0.99),
            max_ms=0.0 if not count else self.maximum,
            success_count=self.success_count,
            failure_count=self.failure_count,
            unique_sandboxes=len(self.sandbox_ids),
            unique_tasks=len(self.task_ids),
            top_records=[
                SlowRecord(
                    metric_name=canonical_name,
                    source_metric_name=source_metric_name,
                    timestamp=item.timestamp,
                    value_ms=item.value_ms,
                    component=item.component,
                    category=_infer_category(canonical_name, item.component),
                    sandbox_id=item.sandbox_id,
                    task_id=item.task_id,
                    request_id=item.request_id,
                    checkpoint_id=item.checkpoint_id,
                    job_id=item.job_id,
                    status=item.status,
                    operation=item.operation,
                )
                for item in self.top_records
            ],
        )


@dataclass
class _LifecycleAccumulator:
    name: str
    start_count: int = 0
    finish_count: int = 0
    finish_status_counts: Counter[str] = field(default_factory=Counter)

    def build_summary(self) -> LifecycleSummary:
        return LifecycleSummary(
            operation_name=self.name,
            start_count=self.start_count,
            finish_count=self.finish_count,
            missing_finish_count=max(0, self.start_count - self.finish_count),
            finish_status_counts=dict(self.finish_status_counts),
        )


@dataclass
class _TaskAccumulator:
    sandbox_id: str
    task_id: str
    agent_types: Counter[str] = field(default_factory=Counter)
    llm_service_types: Counter[str] = field(default_factory=Counter)
    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, *, metric_name: str, value: float, task_id: str, agent_type: str, llm_service_type: str) -> None:
        if task_id:
            self.task_id = task_id
        if agent_type:
            self.agent_types[agent_type] += 1
        if llm_service_type:
            self.llm_service_types[llm_service_type] += 1
        self.metrics[metric_name].append(value)

    def build_summary(self) -> TaskSummary:
        averaged_metrics = {
            metric_name: (sum(values) / len(values))
            for metric_name, values in sorted(self.metrics.items())
            if values
        }
        return TaskSummary(
            sandbox_id=self.sandbox_id,
            task_id=self.task_id,
            agent_type=self.agent_types.most_common(1)[0][0] if self.agent_types else "",
            llm_service_type=self.llm_service_types.most_common(1)[0][0] if self.llm_service_types else "",
            metrics=averaged_metrics,
        )


@dataclass
class _RequestMetricAccumulator:
    sandbox_id: str
    request_id: str
    task_id: str = ""
    request_kind: str = ""
    llm_response_metric_name: str | None = None
    llm_response_timestamp: datetime | None = None
    llm_response_time_ms: float | None = None
    pure_llm_timestamp: datetime | None = None
    pure_llm_time_ms: float | None = None

    def add(
        self,
        *,
        metric_name: str,
        timestamp: datetime,
        value_ms: float,
        task_id: str,
        request_kind: str,
    ) -> None:
        if task_id:
            self.task_id = task_id
        if request_kind:
            self.request_kind = request_kind
        if metric_name == _TURN_ANALYSIS_PURE_LLM_METRIC:
            self.pure_llm_timestamp = timestamp
            self.pure_llm_time_ms = value_ms
            return
        if metric_name not in _TURN_ANALYSIS_RESPONSE_METRIC_SOURCES:
            return
        current_priority = (
            len(_TURN_ANALYSIS_RESPONSE_METRIC_SOURCES)
            if self.llm_response_metric_name is None
            else _TURN_ANALYSIS_RESPONSE_METRIC_SOURCES.index(self.llm_response_metric_name)
        )
        new_priority = _TURN_ANALYSIS_RESPONSE_METRIC_SOURCES.index(metric_name)
        if new_priority <= current_priority:
            self.llm_response_metric_name = metric_name
            self.llm_response_timestamp = timestamp
            self.llm_response_time_ms = value_ms


@dataclass(frozen=True)
class _ResolvedRequestTiming:
    sandbox_id: str
    request_id: str
    task_id: str
    request_kind: str
    start_time: datetime
    finish_time: datetime
    llm_response_time_ms: float
    pure_llm_time_ms: float | None


def _metric_start_time(timestamp: datetime, duration_ms: float) -> datetime:
    return timestamp - timedelta(milliseconds=duration_ms)


def _build_turn_metric_summary(
    metric_name: str,
    values: list[float],
    *,
    request_kind: str = "",
) -> TurnMetricSummary:
    sorted_values = sorted(values)
    count = len(sorted_values)
    mean_ms = (sum(sorted_values) / count) if count else 0.0
    return TurnMetricSummary(
        metric_name=metric_name,
        count=count,
        mean_ms=mean_ms,
        min_ms=0.0 if not count else sorted_values[0],
        p50_ms=_percentile(sorted_values, 0.50),
        p95_ms=_percentile(sorted_values, 0.95),
        p99_ms=_percentile(sorted_values, 0.99),
        max_ms=0.0 if not count else sorted_values[-1],
        request_kind=request_kind,
    )


def _build_turn_analysis_view(
    *,
    started_at: datetime | None,
    metric_values: dict[str, list[float]],
    metric_points: dict[str, list[tuple[datetime, float]]],
    request_kind: str = "",
) -> tuple[list[TurnMetricSummary], dict[str, list[CDFPoint]], dict[str, list[TimeSeriesPoint]]]:
    metrics = [
        _build_turn_metric_summary(metric_name, metric_values[metric_name], request_kind=request_kind)
        for metric_name in _TURN_ANALYSIS_METRIC_ORDER
        if metric_values.get(metric_name)
    ]
    cdf_series = {
        metric_name: _build_cdf_points(metric_values[metric_name])
        for metric_name in _TURN_ANALYSIS_METRIC_ORDER
        if metric_values.get(metric_name)
    }
    time_series = {
        metric_name: [
            TimeSeriesPoint(
                timestamp=timestamp.isoformat(),
                offset_seconds=_offset_seconds(timestamp, started_at),
                value=value,
            )
            for timestamp, value in sorted(metric_points.get(metric_name, []))
        ]
        for metric_name in _TURN_ANALYSIS_METRIC_ORDER
        if metric_points.get(metric_name)
    }
    return metrics, cdf_series, time_series


def _build_cdf_points(values: list[float]) -> list[CDFPoint]:
    sorted_values = sorted(values)
    total = len(sorted_values)
    if total <= 0:
        return []
    return [
        CDFPoint(value=value, cumulative_probability=(index + 1) / total)
        for index, value in enumerate(sorted_values)
    ]


def _build_turn_analysis(
    *,
    started_at: datetime | None,
    request_metric_records: dict[tuple[str, str], _RequestMetricAccumulator],
) -> TurnAnalysisSummary | None:
    resolved_requests_by_sandbox: dict[str, list[_ResolvedRequestTiming]] = defaultdict(list)
    metric_values: dict[str, list[float]] = defaultdict(list)
    metric_points: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    metric_values_by_request_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    metric_points_by_request_kind: dict[str, dict[str, list[tuple[datetime, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    def _record_metric(metric_name: str, timestamp: datetime, value_ms: float, *, request_kind: str = "") -> None:
        metric_values[metric_name].append(value_ms)
        metric_points[metric_name].append((timestamp, value_ms))
        if request_kind:
            metric_values_by_request_kind[request_kind][metric_name].append(value_ms)
            metric_points_by_request_kind[request_kind][metric_name].append((timestamp, value_ms))

    for item in request_metric_records.values():
        if item.pure_llm_timestamp is not None and item.pure_llm_time_ms is not None:
            pure_llm_start_time = _metric_start_time(item.pure_llm_timestamp, item.pure_llm_time_ms)
            _record_metric(
                "pure_llm_time",
                pure_llm_start_time,
                item.pure_llm_time_ms,
                request_kind=item.request_kind,
            )
        if item.llm_response_timestamp is None or item.llm_response_time_ms is None:
            continue
        start_time = _metric_start_time(item.llm_response_timestamp, item.llm_response_time_ms)
        finish_time = item.llm_response_timestamp
        resolved_requests_by_sandbox[item.sandbox_id].append(
            _ResolvedRequestTiming(
                sandbox_id=item.sandbox_id,
                request_id=item.request_id,
                task_id=item.task_id,
                request_kind=item.request_kind,
                start_time=start_time,
                finish_time=finish_time,
                llm_response_time_ms=item.llm_response_time_ms,
                pure_llm_time_ms=item.pure_llm_time_ms,
            )
        )

    if not resolved_requests_by_sandbox:
        if not metric_values:
            return None
        metrics, cdf_series, time_series = _build_turn_analysis_view(
            started_at=started_at,
            metric_values=metric_values,
            metric_points=metric_points,
        )
        metrics_by_request_kind: dict[str, list[TurnMetricSummary]] = {}
        cdf_series_by_request_kind: dict[str, dict[str, list[CDFPoint]]] = {}
        time_series_by_request_kind: dict[str, dict[str, list[TimeSeriesPoint]]] = {}
        for request_kind in sorted(metric_values_by_request_kind):
            kind_metrics, kind_cdf_series, kind_time_series = _build_turn_analysis_view(
                started_at=started_at,
                metric_values=metric_values_by_request_kind[request_kind],
                metric_points=metric_points_by_request_kind[request_kind],
                request_kind=request_kind,
            )
            if kind_metrics:
                metrics_by_request_kind[request_kind] = kind_metrics
                cdf_series_by_request_kind[request_kind] = kind_cdf_series
                time_series_by_request_kind[request_kind] = kind_time_series
        return TurnAnalysisSummary(
            metrics=metrics,
            cdf_series=cdf_series,
            time_series=time_series,
            metrics_by_request_kind=metrics_by_request_kind,
            cdf_series_by_request_kind=cdf_series_by_request_kind,
            time_series_by_request_kind=time_series_by_request_kind,
        )

    for sandbox_id in sorted(resolved_requests_by_sandbox):
        requests = sorted(
            resolved_requests_by_sandbox[sandbox_id],
            key=lambda item: (item.start_time, item.finish_time, item.request_id),
        )
        for index, request in enumerate(requests):
            _record_metric(
                "llm_response_time",
                request.start_time,
                request.llm_response_time_ms,
                request_kind=request.request_kind,
            )
            if index + 1 >= len(requests):
                continue
            next_request = requests[index + 1]
            action_time_ms = max(0.0, (next_request.start_time - request.finish_time).total_seconds() * 1000.0)
            turn_time_ms = request.llm_response_time_ms + action_time_ms
            _record_metric("action_time", request.start_time, action_time_ms, request_kind=request.request_kind)
            _record_metric("turn_time", request.start_time, turn_time_ms, request_kind=request.request_kind)

    if not metric_values:
        return None

    metrics, cdf_series, time_series = _build_turn_analysis_view(
        started_at=started_at,
        metric_values=metric_values,
        metric_points=metric_points,
    )
    metrics_by_request_kind: dict[str, list[TurnMetricSummary]] = {}
    cdf_series_by_request_kind: dict[str, dict[str, list[CDFPoint]]] = {}
    time_series_by_request_kind: dict[str, dict[str, list[TimeSeriesPoint]]] = {}
    for request_kind in sorted(metric_values_by_request_kind):
        kind_metrics, kind_cdf_series, kind_time_series = _build_turn_analysis_view(
            started_at=started_at,
            metric_values=metric_values_by_request_kind[request_kind],
            metric_points=metric_points_by_request_kind[request_kind],
            request_kind=request_kind,
        )
        if kind_metrics:
            metrics_by_request_kind[request_kind] = kind_metrics
            cdf_series_by_request_kind[request_kind] = kind_cdf_series
            time_series_by_request_kind[request_kind] = kind_time_series
    return TurnAnalysisSummary(
        metrics=metrics,
        cdf_series=cdf_series,
        time_series=time_series,
        metrics_by_request_kind=metrics_by_request_kind,
        cdf_series_by_request_kind=cdf_series_by_request_kind,
        time_series_by_request_kind=time_series_by_request_kind,
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def detect_primary_run_id(path: Path) -> str:
    run_ids: Counter[str] = Counter()
    for payload in _iter_jsonl(path):
        attributes = payload.get("attributes")
        if isinstance(attributes, dict):
            run_id = _maybe_str(attributes.get("run_id"))
            if run_id:
                run_ids[run_id] += 1
    if not run_ids:
        raise ValueError(f"telemetry file {path} does not contain run_id attributes")
    return run_ids.most_common(1)[0][0]


def _detect_failed_sandboxes(path: Path, run_id: str) -> set[str]:
    failed: set[str] = set()
    for payload in _iter_jsonl(path):
        if payload.get("name") != "benchmark.task.success_ratio":
            continue
        attributes = payload.get("attributes")
        if not isinstance(attributes, dict):
            continue
        if _maybe_str(attributes.get("run_id")) != run_id:
            continue
        value = _safe_float(payload.get("value"))
        if value is not None and value == 0.0:
            sandbox_id = _maybe_str(attributes.get("sandbox_id"))
            if sandbox_id:
                failed.add(sandbox_id)
    return failed


def _select_operation_summaries(raw_metrics: dict[str, _MetricAccumulator]) -> list[MetricSummary]:
    used_sources: set[str] = set()
    summaries: list[MetricSummary] = []
    for canonical_name, candidates in _PREFERRED_METRIC_SOURCES.items():
        selected = None
        for candidate in candidates:
            if candidate in raw_metrics:
                selected = candidate
                break
        if selected is None:
            continue
        used_sources.add(selected)
        summaries.append(raw_metrics[selected].build_summary(canonical_name=canonical_name, source_metric_name=selected))
    alias_values = {alias for aliases in _PREFERRED_METRIC_SOURCES.values() for alias in aliases[1:]}
    for metric_name, accumulator in raw_metrics.items():
        if metric_name in used_sources or metric_name in alias_values:
            continue
        summaries.append(accumulator.build_summary(canonical_name=metric_name, source_metric_name=metric_name))
    summaries.sort(key=lambda item: (item.category, item.metric_name))
    return summaries


def _summary_lookup(summaries: Iterable[MetricSummary]) -> dict[str, MetricSummary]:
    return {summary.metric_name: summary for summary in summaries}


def _interval_key(attributes: dict[str, Any], fallback: str) -> str:
    op_id = _maybe_str(attributes.get("op_id"))
    if op_id:
        return op_id
    return fallback


def _pair_intervals(raw: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    paired: list[dict[str, Any]] = []
    for op_id, state in raw.items():
        start_attributes = dict(state.get("start_attributes", {}))
        finish_attributes = dict(state.get("finish_attributes", {}))
        merged_attributes = {**start_attributes, **finish_attributes}
        paired.append(
            {
                "op_id": op_id,
                "start": state.get("start"),
                "finish": state.get("finish"),
                "start_attributes": start_attributes,
                "finish_attributes": finish_attributes,
                "attributes": merged_attributes,
                "status": _maybe_str(finish_attributes.get("status")),
            }
        )
    paired.sort(
        key=lambda item: (
            float("-inf")
            if item["start"] is None
            else item["start"].timestamp()
        )
    )
    return paired


def _build_load_series(
    *,
    started_at: datetime | None,
    flow_intervals: list[dict[str, Any]],
    process_intervals: list[dict[str, Any]] | None = None,
    filesystem_intervals: list[dict[str, Any]] | None = None,
    estimated_io_lookup: Callable[[dict[str, Any]], float | None] | None = None,
) -> list[LoadSeriesPoint]:
    deltas: list[tuple[datetime, str, float]] = []
    for interval in flow_intervals:
        start = interval.get("start")
        finish = interval.get("finish")
        if not isinstance(start, datetime) or not isinstance(finish, datetime):
            continue
        attributes = interval.get("attributes", {})
        estimated_io_bytes = _safe_float(attributes.get("estimated_io_bytes"))
        if estimated_io_bytes is None and estimated_io_lookup is not None:
            estimated_io_bytes = estimated_io_lookup(interval)
        estimated_io_bytes = estimated_io_bytes or 0.0
        deltas.append((start, "jobs", 1.0))
        deltas.append((finish, "jobs", -1.0))
        deltas.append((start, "io", estimated_io_bytes))
        deltas.append((finish, "io", -estimated_io_bytes))
    for kind, intervals in (("process", process_intervals or []), ("filesystem", filesystem_intervals or [])):
        for interval in intervals:
            start = interval.get("start")
            finish = interval.get("finish")
            if not isinstance(start, datetime) or not isinstance(finish, datetime):
                continue
            deltas.append((start, kind, 1.0))
            deltas.append((finish, kind, -1.0))
    deltas.sort(key=lambda item: (item[0], 0 if item[2] > 0 else 1, item[1]))

    active_jobs = 0
    active_process_jobs = 0
    active_filesystem_jobs = 0
    active_estimated_io_bytes = 0.0
    points: list[LoadSeriesPoint] = []
    for timestamp, kind, delta in deltas:
        if kind == "jobs":
            active_jobs += int(delta)
        elif kind == "process":
            active_process_jobs += int(delta)
        elif kind == "filesystem":
            active_filesystem_jobs += int(delta)
        elif kind == "io":
            active_estimated_io_bytes += delta
        points.append(
            LoadSeriesPoint(
                timestamp=timestamp.isoformat(),
                offset_seconds=_offset_seconds(timestamp, started_at),
                active_jobs=max(0, active_jobs),
                active_process_jobs=max(0, active_process_jobs),
                active_filesystem_jobs=max(0, active_filesystem_jobs),
                active_estimated_io_bytes=max(0.0, active_estimated_io_bytes),
            )
        )
    return points


def _build_latency_series(
    *,
    started_at: datetime | None,
    intervals: list[dict[str, Any]],
) -> list[TimeSeriesPoint]:
    points: list[TimeSeriesPoint] = []
    for interval in intervals:
        start = interval.get("start")
        finish = interval.get("finish")
        if not isinstance(start, datetime) or not isinstance(finish, datetime):
            continue
        points.append(
            TimeSeriesPoint(
                timestamp=start.isoformat(),
                offset_seconds=_offset_seconds(start, started_at),
                value=max(0.0, (finish - start).total_seconds() * 1000.0),
            )
        )
    points.sort(key=lambda item: (item.offset_seconds, item.timestamp))
    return points


def _lookup_interval_metric_value(
    *,
    interval: dict[str, Any],
    metric_records: dict[tuple[str, str, str], dict[str, Any]],
    metric_name: str,
) -> float | None:
    attributes = interval.get("attributes", {})
    if not isinstance(attributes, dict):
        return None
    sandbox_id = _maybe_str(attributes.get("sandbox_id"))
    checkpoint_id = _maybe_str(attributes.get("checkpoint_id"))
    job_id = _maybe_str(attributes.get("job_id"))
    if not sandbox_id or not checkpoint_id or not job_id:
        return None
    metric_record = metric_records.get((sandbox_id, checkpoint_id, job_id), {})
    return _safe_float(metric_record.get(metric_name))


def _build_checkpoint_analysis(
    *,
    started_at: datetime | None,
    finished_at: datetime | None,
    flow_intervals: list[dict[str, Any]],
    process_intervals: list[dict[str, Any]],
    filesystem_intervals: list[dict[str, Any]],
    checkpoint_metrics: dict[tuple[str, str, str], dict[str, Any]],
) -> CheckpointAnalysisSummary:
    duration_minutes = _duration_minutes(started_at, finished_at)
    per_task: dict[str, dict[str, list[float] | int | str]] = defaultdict(lambda: {
        "task_id": "",
        "counts": 0,
        "success": 0,
        "fail": 0,
        "skip": 0,
        "estimated_io": [],
    })
    scope_counts: Counter[str] = Counter()
    success_count = 0
    fail_count = 0
    skip_count = 0
    promoted_process_checkpoint_count = 0
    process_sizes: list[float] = []
    filesystem_written_sizes: list[float] = []
    estimated_io_sizes: list[float] = []

    for interval in flow_intervals:
        attributes = interval["attributes"]
        sandbox_id = _maybe_str(attributes.get("sandbox_id"))
        checkpoint_id = _maybe_str(attributes.get("checkpoint_id"))
        job_id = _maybe_str(attributes.get("job_id"))
        task_id = _maybe_str(attributes.get("task_id")) or sandbox_id
        status = _maybe_str(attributes.get("status")).lower()
        metric_record = checkpoint_metrics.get((sandbox_id, checkpoint_id, job_id), {})
        scope = _maybe_str(attributes.get("checkpoint_scope")) or _maybe_str(metric_record.get("checkpoint_scope")) or "unknown"
        scope_counts[scope] += 1
        if status == "succeeded":
            success_count += 1
        elif status == "skipped":
            skip_count += 1
        else:
            fail_count += 1
        if bool(attributes.get("promoted_process_checkpoint", False)):
            promoted_process_checkpoint_count += 1
        process_size = _safe_float(metric_record.get("checkpoint.process.size_bytes"))
        filesystem_written = _safe_float(metric_record.get("checkpoint.filesystem.written_bytes"))
        estimated_io = _safe_float(metric_record.get("checkpoint.estimated_io_bytes"))
        if process_size is not None:
            process_sizes.append(process_size)
        if filesystem_written is not None:
            filesystem_written_sizes.append(filesystem_written)
        if estimated_io is not None:
            estimated_io_sizes.append(estimated_io)
        task = per_task[sandbox_id]
        task["task_id"] = task_id or _maybe_str(task.get("task_id"))
        task["counts"] = int(task["counts"]) + 1
        if status == "succeeded":
            task["success"] = int(task["success"]) + 1
        elif status == "skipped":
            task["skip"] = int(task["skip"]) + 1
        else:
            task["fail"] = int(task["fail"]) + 1
        if estimated_io is not None:
            task["estimated_io"].append(estimated_io)

    total_count = len(flow_intervals)
    scope_rates = {
        scope: (count / total_count) if total_count else 0.0
        for scope, count in sorted(scope_counts.items())
    }
    per_task_rows = []
    for sandbox_id, item in sorted(per_task.items()):
        estimated_values = [float(value) for value in item["estimated_io"]]
        total_task_count = int(item["counts"])
        per_task_rows.append(
            OperationTaskAnalysis(
                task_id=_maybe_str(item.get("task_id")),
                sandbox_id=sandbox_id,
                total_count=total_task_count,
                success_count=int(item["success"]),
                fail_count=int(item["fail"]),
                skip_count=int(item["skip"]),
                success_rate=(int(item["success"]) / total_task_count) if total_task_count else 0.0,
                per_minute_rate=(total_task_count / duration_minutes) if duration_minutes else 0.0,
                mean_estimated_io_bytes=(sum(estimated_values) / len(estimated_values)) if estimated_values else 0.0,
            )
        )
    return CheckpointAnalysisSummary(
        total_count=total_count,
        success_count=success_count,
        fail_count=fail_count,
        skip_count=skip_count,
        fail_rate=(fail_count / total_count) if total_count else 0.0,
        skip_rate=(skip_count / total_count) if total_count else 0.0,
        overall_rate_per_minute=(total_count / duration_minutes) if duration_minutes else 0.0,
        process_frequency_per_minute=(len(process_intervals) / duration_minutes) if duration_minutes else 0.0,
        filesystem_frequency_per_minute=(len(filesystem_intervals) / duration_minutes) if duration_minutes else 0.0,
        scope_counts=dict(scope_counts),
        scope_rates=scope_rates,
        filesystem_only_to_full_ratio=(
            (scope_counts.get("filesystem_only", 0) / scope_counts.get("full", 1))
            if scope_counts.get("full", 0)
            else 0.0
        ),
        promoted_process_checkpoint_count=promoted_process_checkpoint_count,
        total_process_size_bytes=sum(process_sizes),
        total_filesystem_written_bytes=sum(filesystem_written_sizes),
        total_estimated_io_bytes=sum(estimated_io_sizes),
        mean_process_size_bytes=(sum(process_sizes) / len(process_sizes)) if process_sizes else 0.0,
        mean_filesystem_written_bytes=(
            (sum(filesystem_written_sizes) / len(filesystem_written_sizes)) if filesystem_written_sizes else 0.0
        ),
        mean_estimated_io_bytes=(sum(estimated_io_sizes) / len(estimated_io_sizes)) if estimated_io_sizes else 0.0,
        p95_process_size_bytes=_percentile(sorted(process_sizes), 0.95),
        p95_filesystem_written_bytes=_percentile(sorted(filesystem_written_sizes), 0.95),
        per_task=sorted(per_task_rows, key=lambda item: (item.total_count, item.sandbox_id, item.task_id), reverse=True),
        load_over_time=_build_load_series(
            started_at=started_at,
            flow_intervals=flow_intervals,
            process_intervals=process_intervals,
            filesystem_intervals=filesystem_intervals,
            estimated_io_lookup=lambda interval: _lookup_interval_metric_value(
                interval=interval,
                metric_records=checkpoint_metrics,
                metric_name="checkpoint.estimated_io_bytes",
            ),
        ),
        latency_over_time={
            "checkpoint.flow.duration_ms": _build_latency_series(started_at=started_at, intervals=flow_intervals),
            "checkpoint.process.duration_ms": _build_latency_series(started_at=started_at, intervals=process_intervals),
            "checkpoint.filesystem.duration_ms": _build_latency_series(started_at=started_at, intervals=filesystem_intervals),
        },
    )


def _build_restore_analysis(
    *,
    started_at: datetime | None,
    finished_at: datetime | None,
    flow_intervals: list[dict[str, Any]],
    process_intervals: list[dict[str, Any]],
    filesystem_intervals: list[dict[str, Any]],
    restore_metrics: dict[tuple[str, str, str], dict[str, Any]],
) -> RestoreAnalysisSummary:
    duration_minutes = _duration_minutes(started_at, finished_at)
    success_count = 0
    fail_count = 0
    skip_count = 0
    mixed_source_count = 0
    estimated_io_sizes: list[float] = []
    source_gap_turns: list[float] = []
    source_gap_ms: list[float] = []
    per_task: dict[str, dict[str, list[float] | int | str]] = defaultdict(lambda: {
        "task_id": "",
        "counts": 0,
        "success": 0,
        "fail": 0,
        "skip": 0,
        "estimated_io": [],
        "gap_turns": [],
        "gap_ms": [],
    })

    for interval in flow_intervals:
        attributes = interval["attributes"]
        sandbox_id = _maybe_str(attributes.get("sandbox_id"))
        checkpoint_id = _maybe_str(attributes.get("checkpoint_id"))
        job_id = _maybe_str(attributes.get("job_id"))
        task_id = _maybe_str(attributes.get("task_id")) or sandbox_id
        status = _maybe_str(attributes.get("status")).lower()
        metric_record = restore_metrics.get((sandbox_id, checkpoint_id, job_id), {})
        mixed_sources = bool(metric_record.get("restore.mixed_sources", attributes.get("mixed_sources", False)))
        estimated_io = _safe_float(metric_record.get("restore.estimated_io_bytes"))
        gap_turns = _safe_float(metric_record.get("restore.source_gap.turns"))
        gap_ms = _safe_float(metric_record.get("restore.source_gap.ms"))
        if status == "succeeded":
            success_count += 1
        elif status == "skipped":
            skip_count += 1
        else:
            fail_count += 1
        if mixed_sources:
            mixed_source_count += 1
        if estimated_io is not None:
            estimated_io_sizes.append(estimated_io)
        if gap_turns is not None:
            source_gap_turns.append(gap_turns)
        if gap_ms is not None:
            source_gap_ms.append(gap_ms)
        task = per_task[sandbox_id]
        task["task_id"] = task_id or _maybe_str(task.get("task_id"))
        task["counts"] = int(task["counts"]) + 1
        if status == "succeeded":
            task["success"] = int(task["success"]) + 1
        elif status == "skipped":
            task["skip"] = int(task["skip"]) + 1
        else:
            task["fail"] = int(task["fail"]) + 1
        if estimated_io is not None:
            task["estimated_io"].append(estimated_io)
        if gap_turns is not None:
            task["gap_turns"].append(gap_turns)
        if gap_ms is not None:
            task["gap_ms"].append(gap_ms)

    per_task_rows = []
    for sandbox_id, item in sorted(per_task.items()):
        total_task_count = int(item["counts"])
        estimated_values = [float(value) for value in item["estimated_io"]]
        gap_turn_values = [float(value) for value in item["gap_turns"]]
        gap_ms_values = [float(value) for value in item["gap_ms"]]
        per_task_rows.append(
            OperationTaskAnalysis(
                task_id=_maybe_str(item.get("task_id")),
                sandbox_id=sandbox_id,
                total_count=total_task_count,
                success_count=int(item["success"]),
                fail_count=int(item["fail"]),
                skip_count=int(item["skip"]),
                success_rate=(int(item["success"]) / total_task_count) if total_task_count else 0.0,
                per_minute_rate=(total_task_count / duration_minutes) if duration_minutes else 0.0,
                mean_estimated_io_bytes=(sum(estimated_values) / len(estimated_values)) if estimated_values else 0.0,
                mean_source_gap_turns=(sum(gap_turn_values) / len(gap_turn_values)) if gap_turn_values else 0.0,
                mean_source_gap_ms=(sum(gap_ms_values) / len(gap_ms_values)) if gap_ms_values else 0.0,
            )
        )

    total_count = len(flow_intervals)
    return RestoreAnalysisSummary(
        total_count=total_count,
        success_count=success_count,
        fail_count=fail_count,
        skip_count=skip_count,
        fail_rate=(fail_count / total_count) if total_count else 0.0,
        skip_rate=(skip_count / total_count) if total_count else 0.0,
        overall_rate_per_minute=(total_count / duration_minutes) if duration_minutes else 0.0,
        mixed_source_count=mixed_source_count,
        mixed_source_rate=(mixed_source_count / total_count) if total_count else 0.0,
        mean_estimated_io_bytes=(sum(estimated_io_sizes) / len(estimated_io_sizes)) if estimated_io_sizes else 0.0,
        mean_source_gap_turns=(sum(source_gap_turns) / len(source_gap_turns)) if source_gap_turns else 0.0,
        p95_source_gap_turns=_percentile(sorted(source_gap_turns), 0.95),
        max_source_gap_turns=max(source_gap_turns) if source_gap_turns else 0.0,
        mean_source_gap_ms=(sum(source_gap_ms) / len(source_gap_ms)) if source_gap_ms else 0.0,
        p95_source_gap_ms=_percentile(sorted(source_gap_ms), 0.95),
        max_source_gap_ms=max(source_gap_ms) if source_gap_ms else 0.0,
        per_task=sorted(per_task_rows, key=lambda item: (item.total_count, item.sandbox_id, item.task_id), reverse=True),
        load_over_time=_build_load_series(
            started_at=started_at,
            flow_intervals=flow_intervals,
            process_intervals=process_intervals,
            filesystem_intervals=filesystem_intervals,
            estimated_io_lookup=lambda interval: _lookup_interval_metric_value(
                interval=interval,
                metric_records=restore_metrics,
                metric_name="restore.estimated_io_bytes",
            ),
        ),
        latency_over_time={
            "restore.flow.duration_ms": _build_latency_series(started_at=started_at, intervals=flow_intervals),
            "restore.process.duration_ms": _build_latency_series(started_at=started_at, intervals=process_intervals),
            "restore.filesystem.duration_ms": _build_latency_series(started_at=started_at, intervals=filesystem_intervals),
        },
    )


def _build_resource_analysis(
    *,
    started_at: datetime | None,
    host_series_raw: dict[str, list[tuple[datetime, float]]],
    sandbox_series_raw: dict[str, dict[str, list[tuple[datetime, float]]]],
) -> ResourceAnalysisSummary:
    host_series = {
        metric_name: [
            TimeSeriesPoint(
                timestamp=timestamp.isoformat(),
                offset_seconds=_offset_seconds(timestamp, started_at),
                value=value,
            )
            for timestamp, value in sorted(items)
        ]
        for metric_name, items in sorted(host_series_raw.items())
    }
    sandbox_series = {
        metric_name: {
            sandbox_id: [
                TimeSeriesPoint(
                    timestamp=timestamp.isoformat(),
                    offset_seconds=_offset_seconds(timestamp, started_at),
                    value=value,
                )
                for timestamp, value in sorted(series)
            ]
            for sandbox_id, series in sorted(by_sandbox.items())
        }
        for metric_name, by_sandbox in sorted(sandbox_series_raw.items())
    }

    host_metrics = [
        ResourceMetricSummary(
            metric_name=metric_name,
            sandbox_id="",
            sample_count=len(points),
            mean_value=(sum(point.value for point in points) / len(points)) if points else 0.0,
            max_value=max((point.value for point in points), default=0.0),
        )
        for metric_name, points in host_series.items()
    ]

    sandbox_metrics: list[ResourceMetricSummary] = []
    for metric_name, by_sandbox in sandbox_series.items():
        for sandbox_id, points in by_sandbox.items():
            sandbox_metrics.append(
                ResourceMetricSummary(
                    metric_name=metric_name,
                    sandbox_id=sandbox_id,
                    sample_count=len(points),
                    mean_value=(sum(point.value for point in points) / len(points)) if points else 0.0,
                    max_value=max((point.value for point in points), default=0.0),
                )
            )
    sandbox_metrics.sort(key=lambda item: (item.metric_name, item.sandbox_id))

    coverage = {
        "host_cpu": "resource.host.cpu.usage_percent" in host_series,
        "host_memory": "resource.host.memory.used_bytes" in host_series,
        "host_disk": any(name.startswith("resource.host.disk.") for name in host_series),
        "host_network": any(name.startswith("resource.host.network.") for name in host_series),
        "sandbox_cpu": "resource.sandbox.cpu.usage_percent" in sandbox_series,
        "sandbox_memory": (
            "resource.sandbox.memory.current_bytes" in sandbox_series
            or "resource.sandbox.memory.peak_bytes" in sandbox_series
        ),
        "sandbox_filesystem": any(name.startswith("resource.sandbox.filesystem.") for name in sandbox_series),
        "sandbox_disk": any(name.startswith("resource.sandbox.disk.") for name in sandbox_series),
        "sandbox_network": any(name.startswith("resource.sandbox.network.") for name in sandbox_series),
    }
    return ResourceAnalysisSummary(
        host_metrics=host_metrics,
        sandbox_metrics=sandbox_metrics,
        host_series=host_series,
        sandbox_series=sandbox_series,
        coverage=coverage,
    )


def _collect_phase_overview(
    path: Path,
    *,
    run_id: str,
) -> tuple[list[PhaseSummary], tuple[datetime, datetime] | None]:
    phase_states: dict[str, dict[str, Any]] = defaultdict(dict)
    for payload in _iter_jsonl(path):
        attributes = payload.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        if _maybe_str(attributes.get("run_id")) != run_id:
            continue
        name = _maybe_str(payload.get("name"))
        phase = _benchmark_phase_name(name, attributes)
        if phase is None:
            continue
        state = phase_states[phase]
        state["sandbox_count"] = _safe_int(attributes.get("sandbox_count")) or state.get("sandbox_count")
        configured_workers = _safe_int(attributes.get("configured_max_workers"))
        if configured_workers is not None:
            state["configured_max_workers"] = configured_workers
        timestamp = _record_timestamp(payload)
        kind = _maybe_str(payload.get("kind"))
        if kind == "event" and name.endswith(".start"):
            state["started_at"] = timestamp
        elif kind == "event" and name.endswith(".finish"):
            state["finished_at"] = timestamp
            status = _maybe_str(attributes.get("status"))
            if status:
                state["status"] = status
        elif kind == "metric" and name.endswith(".configured_max_workers"):
            metric_workers = _safe_int(payload.get("value"))
            if metric_workers is not None:
                state["configured_max_workers"] = metric_workers
        elif kind == "metric" and name.endswith(".duration_ms"):
            duration_ms = _safe_float(payload.get("value"))
            if duration_ms is not None:
                state["duration_ms"] = duration_ms

    phase_overview: list[PhaseSummary] = []
    for phase, state in phase_states.items():
        started_at = state.get("started_at")
        finished_at = state.get("finished_at")
        duration_ms = _safe_float(state.get("duration_ms"))
        if duration_ms is None and isinstance(started_at, datetime) and isinstance(finished_at, datetime):
            duration_ms = max(0.0, (finished_at - started_at).total_seconds() * 1000.0)
        phase_overview.append(
            PhaseSummary(
                phase=phase,
                started_at=_format_ts(started_at if isinstance(started_at, datetime) else None),
                finished_at=_format_ts(finished_at if isinstance(finished_at, datetime) else None),
                duration_ms=0.0 if duration_ms is None else duration_ms,
                configured_max_workers=_safe_int(state.get("configured_max_workers")),
                sandbox_count=_safe_int(state.get("sandbox_count")),
                status=_maybe_str(state.get("status")),
            )
        )
    phase_overview.sort(key=lambda item: (item.started_at or item.finished_at, item.phase))

    run_state = phase_states.get("run", {})
    run_started_at = run_state.get("started_at")
    run_finished_at = run_state.get("finished_at")
    detail_window: tuple[datetime, datetime] | None = None
    if isinstance(run_started_at, datetime) and isinstance(run_finished_at, datetime):
        detail_window = (run_started_at, run_finished_at)
    return phase_overview, detail_window


def analyze_telemetry_file(
    telemetry_path: Path,
    *,
    run_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    exclude_failed_tasks: bool = False,
) -> TelemetryAnalysis:
    path = telemetry_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    selected_run_id = run_id or detect_primary_run_id(path)

    failed_sandbox_ids: set[str] = set()
    if exclude_failed_tasks:
        failed_sandbox_ids = _detect_failed_sandboxes(path, selected_run_id)
    phase_overview, detail_window = _collect_phase_overview(path, run_id=selected_run_id)
    detail_scope_name = "run" if detail_window is not None else "all"

    metric_accumulators: dict[str, _MetricAccumulator] = {}
    lifecycle_accumulators: dict[str, _LifecycleAccumulator] = {}
    task_accumulators: dict[str, _TaskAccumulator] = {}
    sandbox_to_task: dict[str, str] = {}
    excluded_sandbox_to_task: dict[str, str] = {}
    event_name_counts: Counter[str] = Counter()
    metric_name_counts: Counter[str] = Counter()
    sandbox_ids: set[str] = set()
    task_ids: set[str] = set()
    request_ids: set[str] = set()
    checkpoint_ids: set[str] = set()
    job_ids: set[str] = set()
    total_records = 0
    total_events = 0
    total_metrics = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    detail_total_records = 0
    detail_total_events = 0
    detail_total_metrics = 0
    detail_started_at: datetime | None = None if detail_window is None else detail_window[0]
    detail_finished_at: datetime | None = None if detail_window is None else detail_window[1]

    interval_sources: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    checkpoint_metrics: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    restore_metrics: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(dict)
    overhead_series_raw: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    host_series_raw: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    sandbox_series_raw: dict[str, dict[str, list[tuple[datetime, float]]]] = defaultdict(lambda: defaultdict(list))
    request_metric_records: dict[tuple[str, str], _RequestMetricAccumulator] = {}

    interval_operations = {
        "checkpoint.flow",
        "checkpoint.process",
        "checkpoint.filesystem",
        "restore.flow",
        "restore.process",
        "restore.filesystem",
    }

    for row_index, payload in enumerate(_iter_jsonl(path), start=1):
        attributes = payload.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        if _maybe_str(attributes.get("run_id")) != selected_run_id:
            continue

        sandbox_id_raw = _maybe_str(attributes.get("sandbox_id"))
        task_id_raw = _maybe_str(attributes.get("task_id"))
        if sandbox_id_raw and sandbox_id_raw in failed_sandbox_ids:
            if task_id_raw:
                excluded_sandbox_to_task[sandbox_id_raw] = task_id_raw
            continue

        total_records += 1
        timestamp = _record_timestamp(payload)
        if timestamp is not None:
            started_at = timestamp if started_at is None else min(started_at, timestamp)
            finished_at = timestamp if finished_at is None else max(finished_at, timestamp)

        kind = _maybe_str(payload.get("kind"))
        if kind == "event":
            total_events += 1
        elif kind == "metric":
            total_metrics += 1

        name = _maybe_str(payload.get("name"))
        if not _is_detail_record(name=name, timestamp=timestamp, detail_window=detail_window):
            continue

        detail_total_records += 1
        if timestamp is not None:
            detail_started_at = timestamp if detail_started_at is None else min(detail_started_at, timestamp)
            detail_finished_at = timestamp if detail_finished_at is None else max(detail_finished_at, timestamp)

        sandbox_id = _maybe_str(attributes.get("sandbox_id"))
        task_id = _maybe_str(attributes.get("task_id"))
        if sandbox_id:
            sandbox_ids.add(sandbox_id)
        if task_id:
            sandbox_to_task[sandbox_id] = task_id
            task_ids.add(task_id)
        elif sandbox_id and sandbox_id in sandbox_to_task:
            task_id = sandbox_to_task[sandbox_id]
        request_id = _maybe_str(attributes.get("request_id"))
        checkpoint_id = _maybe_str(attributes.get("checkpoint_id"))
        job_id = _maybe_str(attributes.get("job_id"))
        if request_id:
            request_ids.add(request_id)
        if checkpoint_id:
            checkpoint_ids.add(checkpoint_id)
        if job_id:
            job_ids.add(job_id)

        enriched_attributes = dict(attributes)
        if task_id and not enriched_attributes.get("task_id"):
            enriched_attributes["task_id"] = task_id

        if kind == "event":
            detail_total_events += 1
            event_name_counts[name] += 1
            if name.endswith(".start"):
                base_name = name[:-6]
                lifecycle_accumulators.setdefault(base_name, _LifecycleAccumulator(name=base_name)).start_count += 1
            elif name.endswith(".finish"):
                base_name = name[:-7]
                accumulator = lifecycle_accumulators.setdefault(base_name, _LifecycleAccumulator(name=base_name))
                accumulator.finish_count += 1
                status = _maybe_str(enriched_attributes.get("status"))
                if status:
                    accumulator.finish_status_counts[status] += 1

            base_name = ""
            if name.endswith(".start"):
                base_name = name[:-6]
            elif name.endswith(".finish"):
                base_name = name[:-7]
            if base_name in interval_operations:
                key = _interval_key(enriched_attributes, f"{base_name}:{row_index}")
                state = interval_sources[base_name].setdefault(key, {})
                if name.endswith(".start"):
                    state["start"] = timestamp
                    state["start_attributes"] = dict(enriched_attributes)
                else:
                    state["finish"] = timestamp
                    state["finish_attributes"] = dict(enriched_attributes)
            continue

        if kind != "metric":
            continue

        detail_total_metrics += 1
        metric_name_counts[name] += 1
        value = _safe_float(payload.get("value"))
        if value is None:
            continue
        accumulator = metric_accumulators.setdefault(name, _MetricAccumulator(name=name))
        accumulator.add(value, payload=payload, attributes=enriched_attributes, top_k=top_k)
        if (
            timestamp is not None
            and sandbox_id
            and request_id
            and (name in _TURN_ANALYSIS_RESPONSE_METRIC_SOURCES or name == _TURN_ANALYSIS_PURE_LLM_METRIC)
        ):
            request_metric_records.setdefault(
                (sandbox_id, request_id),
                _RequestMetricAccumulator(
                    sandbox_id=sandbox_id,
                    request_id=request_id,
                    task_id=task_id,
                ),
            ).add(
                metric_name=name,
                timestamp=timestamp,
                value_ms=value,
                task_id=task_id,
                request_kind=_maybe_str(enriched_attributes.get("request_kind")),
            )
        if timestamp is not None and name in _OVERHEAD_ANALYSIS_METRICS:
            overhead_series_raw[name].append((timestamp, value))

        metric_task_id = task_id
        if sandbox_id and (
            name in _DEFAULT_BREAKDOWN_METRICS
            or name in _LLM_BREAKDOWN_METRICS
            or name in _CHECKPOINT_BREAKDOWN_METRICS
        ):
            task_accumulator = task_accumulators.setdefault(
                sandbox_id,
                _TaskAccumulator(sandbox_id=sandbox_id, task_id=metric_task_id),
            )
            task_accumulator.add(
                metric_name=name,
                value=value,
                task_id=metric_task_id,
                agent_type=_maybe_str(enriched_attributes.get("agent_type")),
                llm_service_type=_maybe_str(enriched_attributes.get("llm_service_type")),
            )

        if name in _CHECKPOINT_SIZE_METRICS:
            checkpoint_metrics[(sandbox_id, checkpoint_id, job_id)][name] = value
            checkpoint_scope = _maybe_str(enriched_attributes.get("checkpoint_scope"))
            if checkpoint_scope:
                checkpoint_metrics[(sandbox_id, checkpoint_id, job_id)]["checkpoint_scope"] = checkpoint_scope
        elif name == "checkpoint.total_ms":
            checkpoint_metrics[(sandbox_id, checkpoint_id, job_id)]["checkpoint_scope"] = _maybe_str(
                enriched_attributes.get("checkpoint_scope")
            )
        elif name in _RESTORE_EXTRA_METRICS:
            restore_metrics[(sandbox_id, checkpoint_id, job_id)][name] = value
            for key in (
                "restore.process_source_checkpoint_id",
                "restore.filesystem_source_checkpoint_id",
                "restore.process_source_created_at",
                "restore.filesystem_source_created_at",
                "restore.process_source_trace_cursor",
                "restore.filesystem_source_trace_cursor",
                "restore.mixed_sources",
            ):
                if key in enriched_attributes:
                    restore_metrics[(sandbox_id, checkpoint_id, job_id)][key] = enriched_attributes[key]
        elif name == "restore.total_ms":
            for key in (
                "restore.process_source_checkpoint_id",
                "restore.filesystem_source_checkpoint_id",
                "restore.process_source_created_at",
                "restore.filesystem_source_created_at",
                "restore.process_source_trace_cursor",
                "restore.filesystem_source_trace_cursor",
                "restore.mixed_sources",
            ):
                if key in enriched_attributes:
                    restore_metrics[(sandbox_id, checkpoint_id, job_id)][key] = enriched_attributes[key]

        if timestamp is not None and name.startswith(_RESOURCE_HOST_PREFIX):
            host_series_raw[name].append((timestamp, value))
        elif timestamp is not None and name.startswith(_RESOURCE_SANDBOX_PREFIX):
            sandbox_series_raw[name][sandbox_id].append((timestamp, value))

    operation_summaries = _select_operation_summaries(metric_accumulators)
    summary_by_name = _summary_lookup(operation_summaries)

    top_total_time_operations = sorted(
        (
            summary
            for summary in operation_summaries
            if summary.metric_name.endswith(".duration_ms") and summary.metric_name not in _EXCLUDED_FROM_TOTAL_TIME
        ),
        key=lambda item: item.total_ms,
        reverse=True,
    )[:top_k]
    top_invocation_operations = sorted(
        (summary for summary in operation_summaries if summary.metric_name.endswith(".duration_ms")),
        key=lambda item: item.count,
        reverse=True,
    )[:top_k]
    top_tail_latency_operations = sorted(
        (
            summary
            for summary in operation_summaries
            if summary.metric_name.endswith(".duration_ms") and summary.count >= 3
        ),
        key=lambda item: (item.p95_ms, item.max_ms),
        reverse=True,
    )[:top_k]

    slowest_records = sorted(
        [
            record
            for summary in operation_summaries
            if summary.metric_name.endswith(".duration_ms") and summary.metric_name not in _EXCLUDED_FROM_TOTAL_TIME
            for record in summary.top_records
        ],
        key=lambda item: item.value_ms,
        reverse=True,
    )[:top_k]

    task_summaries = sorted(
        (accumulator.build_summary() for accumulator in task_accumulators.values()),
        key=lambda item: next(
            (item.metrics.get(metric_name, 0.0) for metric_name in _TASK_SUMMARY_SORT_METRICS if metric_name in item.metrics),
            0.0,
        ),
        reverse=True,
    )
    lifecycle_summaries = sorted(
        (accumulator.build_summary() for accumulator in lifecycle_accumulators.values()),
        key=lambda item: item.operation_name,
    )
    lifecycle_gaps = [item for item in lifecycle_summaries if item.missing_finish_count > 0]

    llm_breakdown = {
        metric_name: summary_by_name[metric_name].mean_ms
        for metric_name in _LLM_BREAKDOWN_METRICS
        if metric_name in summary_by_name
    }
    checkpoint_breakdown = {
        metric_name: summary_by_name[metric_name].mean_ms
        for metric_name in _CHECKPOINT_BREAKDOWN_METRICS
        if metric_name in summary_by_name
    }
    overhead_started_at = detail_started_at or started_at
    overhead_metrics = [
        summary_by_name[metric_name]
        for metric_name in _OVERHEAD_ANALYSIS_METRICS
        if metric_name in summary_by_name
    ]
    overhead_time_series = {
        metric_name: [
            TimeSeriesPoint(
                timestamp=timestamp.isoformat(),
                offset_seconds=_offset_seconds(timestamp, overhead_started_at),
                value=value,
            )
            for timestamp, value in sorted(overhead_series_raw.get(metric_name, []))
        ]
        for metric_name in _OVERHEAD_ANALYSIS_METRICS
        if overhead_series_raw.get(metric_name)
    }
    overhead_cdf_series = {
        metric_name: _build_cdf_points([value for _, value in sorted(overhead_series_raw.get(metric_name, []))])
        for metric_name in _OVERHEAD_ANALYSIS_METRICS
        if overhead_series_raw.get(metric_name)
    }
    overhead_analysis = None
    if overhead_metrics or overhead_time_series:
        overhead_analysis = OverheadAnalysisSummary(
            metrics=overhead_metrics,
            time_series=overhead_time_series,
            cdf_series=overhead_cdf_series,
        )
    turn_analysis = _build_turn_analysis(
        started_at=detail_started_at or started_at,
        request_metric_records=request_metric_records,
    )

    checkpoint_flow_intervals = _pair_intervals(interval_sources["checkpoint.flow"])
    checkpoint_process_intervals = _pair_intervals(interval_sources["checkpoint.process"])
    checkpoint_filesystem_intervals = _pair_intervals(interval_sources["checkpoint.filesystem"])
    restore_flow_intervals = _pair_intervals(interval_sources["restore.flow"])
    restore_process_intervals = _pair_intervals(interval_sources["restore.process"])
    restore_filesystem_intervals = _pair_intervals(interval_sources["restore.filesystem"])

    checkpoint_analysis = _build_checkpoint_analysis(
        started_at=started_at,
        finished_at=finished_at,
        flow_intervals=checkpoint_flow_intervals,
        process_intervals=checkpoint_process_intervals,
        filesystem_intervals=checkpoint_filesystem_intervals,
        checkpoint_metrics=checkpoint_metrics,
    )
    restore_analysis = _build_restore_analysis(
        started_at=started_at,
        finished_at=finished_at,
        flow_intervals=restore_flow_intervals,
        process_intervals=restore_process_intervals,
        filesystem_intervals=restore_filesystem_intervals,
        restore_metrics=restore_metrics,
    )
    resource_analysis = _build_resource_analysis(
        started_at=started_at,
        host_series_raw=host_series_raw,
        sandbox_series_raw=sandbox_series_raw,
    )

    excluded_pairs = sorted(
        (sid, excluded_sandbox_to_task.get(sid, ""))
        for sid in failed_sandbox_ids
    )

    return TelemetryAnalysis(
        input_path=str(path),
        run_id=selected_run_id,
        total_records=total_records,
        total_events=total_events,
        total_metrics=total_metrics,
        started_at="" if started_at is None else started_at.isoformat(),
        finished_at="" if finished_at is None else finished_at.isoformat(),
        distinct_sandboxes=len(sandbox_ids),
        distinct_tasks=len(task_ids),
        distinct_requests=len(request_ids),
        distinct_checkpoints=len(checkpoint_ids),
        distinct_jobs=len(job_ids),
        event_name_counts=dict(event_name_counts),
        metric_name_counts=dict(metric_name_counts),
        operation_summaries=operation_summaries,
        top_total_time_operations=top_total_time_operations,
        top_invocation_operations=top_invocation_operations,
        top_tail_latency_operations=top_tail_latency_operations,
        task_summaries=task_summaries,
        lifecycle_summaries=lifecycle_summaries,
        lifecycle_gaps=lifecycle_gaps,
        slowest_records=slowest_records,
        llm_breakdown=llm_breakdown,
        checkpoint_breakdown=checkpoint_breakdown,
        phase_overview=phase_overview,
        detail_scope_name=detail_scope_name,
        detail_total_records=detail_total_records,
        detail_total_events=detail_total_events,
        detail_total_metrics=detail_total_metrics,
        detail_started_at=_format_ts(detail_started_at),
        detail_finished_at=_format_ts(detail_finished_at),
        checkpoint_analysis=checkpoint_analysis,
        overhead_analysis=overhead_analysis,
        turn_analysis=turn_analysis,
        restore_analysis=restore_analysis,
        resource_analysis=resource_analysis,
        exclude_failed_tasks=exclude_failed_tasks,
        excluded_sandbox_task_pairs=excluded_pairs,
    )

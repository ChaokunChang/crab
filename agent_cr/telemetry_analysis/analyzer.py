from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Iterable


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

_EXCLUDED_FROM_TOTAL_TIME = {
    "sandbox.command_duration_ms",
}


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
    if metric_name.startswith("scheduler.") or metric_name.startswith("executor."):
        return "scheduler_executor"
    if metric_name.startswith("sandbox.") or metric_name.startswith("image."):
        return "runtime"
    if component:
        return component
    return "other"


def _escape_csv_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


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
    task_id: str
    sandbox_ids: list[str]
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
    task_id: str
    sandbox_ids: set[str] = field(default_factory=set)
    agent_types: Counter[str] = field(default_factory=Counter)
    llm_service_types: Counter[str] = field(default_factory=Counter)
    metrics: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))

    def add(self, *, metric_name: str, value: float, sandbox_id: str, agent_type: str, llm_service_type: str) -> None:
        if sandbox_id:
            self.sandbox_ids.add(sandbox_id)
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
            task_id=self.task_id,
            sandbox_ids=sorted(self.sandbox_ids),
            agent_type=self.agent_types.most_common(1)[0][0] if self.agent_types else "",
            llm_service_type=self.llm_service_types.most_common(1)[0][0] if self.llm_service_types else "",
            metrics=averaged_metrics,
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


def _detect_failed_sandboxes(
    path: Path,
    run_id: str,
) -> set[str]:
    """First-pass scan: collect sandbox_ids where benchmark.task.success_ratio == 0."""
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
    for metric_name, accumulator in raw_metrics.items():
        if metric_name in used_sources:
            continue
        if metric_name in {
            alias
            for aliases in _PREFERRED_METRIC_SOURCES.values()
            for alias in aliases[1:]
        }:
            continue
        summaries.append(accumulator.build_summary(canonical_name=metric_name, source_metric_name=metric_name))
    summaries.sort(key=lambda item: (item.category, item.metric_name))
    return summaries


def _summary_lookup(summaries: Iterable[MetricSummary]) -> dict[str, MetricSummary]:
    return {summary.metric_name: summary for summary in summaries}


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

    # When filtering is enabled, first-pass detects sandbox_ids with failed tasks.
    failed_sandbox_ids: set[str] = set()
    if exclude_failed_tasks:
        failed_sandbox_ids = _detect_failed_sandboxes(path, selected_run_id)

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

    for payload in _iter_jsonl(path):
        attributes = payload.get("attributes")
        if not isinstance(attributes, dict):
            attributes = {}
        if _maybe_str(attributes.get("run_id")) != selected_run_id:
            continue

        # Track sandbox-to-task mapping for excluded sandboxes before skipping.
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

        kind = _maybe_str(payload.get("kind"))
        name = _maybe_str(payload.get("name"))
        if kind == "event":
            total_events += 1
            event_name_counts[name] += 1
            if name.endswith(".start"):
                base_name = name[:-6]
                lifecycle_accumulators.setdefault(base_name, _LifecycleAccumulator(name=base_name)).start_count += 1
            elif name.endswith(".finish"):
                base_name = name[:-7]
                accumulator = lifecycle_accumulators.setdefault(base_name, _LifecycleAccumulator(name=base_name))
                accumulator.finish_count += 1
                status = _maybe_str(attributes.get("status"))
                if status:
                    accumulator.finish_status_counts[status] += 1
            continue

        if kind != "metric":
            continue
        total_metrics += 1
        metric_name_counts[name] += 1
        value = _safe_float(payload.get("value"))
        if value is None:
            continue
        accumulator = metric_accumulators.setdefault(name, _MetricAccumulator(name=name))
        enriched_attributes = dict(attributes)
        if task_id and not enriched_attributes.get("task_id"):
            enriched_attributes["task_id"] = task_id
        accumulator.add(value, payload=payload, attributes=enriched_attributes, top_k=top_k)

        metric_task_id = task_id
        if metric_task_id and (
            name in _DEFAULT_BREAKDOWN_METRICS
            or name in _LLM_BREAKDOWN_METRICS
            or name in _CHECKPOINT_BREAKDOWN_METRICS
        ):
            task_accumulator = task_accumulators.setdefault(metric_task_id, _TaskAccumulator(task_id=metric_task_id))
            task_accumulator.add(
                metric_name=name,
                value=value,
                sandbox_id=sandbox_id,
                agent_type=_maybe_str(attributes.get("agent_type")),
                llm_service_type=_maybe_str(attributes.get("llm_service_type")),
            )

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
        key=lambda item: item.metrics.get("benchmark.task.duration_ms", 0.0),
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
        exclude_failed_tasks=exclude_failed_tasks,
        excluded_sandbox_task_pairs=excluded_pairs,
    )

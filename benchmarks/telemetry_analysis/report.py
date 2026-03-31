from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
from typing import Iterable

from .analyzer import (
    CDFPoint,
    CheckpointAnalysisSummary,
    MetricSummary,
    OverheadAnalysisSummary,
    PhaseSummary,
    ResourceAnalysisSummary,
    RestoreAnalysisSummary,
    TaskSummary,
    TelemetryAnalysis,
    TurnAnalysisSummary,
    analyze_telemetry_file,
)


def _format_ms(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.2f} ms"


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.0f}%"


def _format_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    unit = units[0]
    for candidate in units:
        unit = candidate
        if abs(size) < 1024.0 or candidate == units[-1]:
            break
        size /= 1024.0
    return f"{size:.2f} {unit}"


def _format_metric_label(metric_name: str) -> str:
    return metric_name.replace(".duration_ms", "").replace("_", " ")


_OVERHEAD_METRIC_LABELS = {
    "llm.gate_wait_ms": "llm.gate_wait",
    "llm.agentcr_delay_ms": "llm.agentcr_delay",
}

_TURN_METRIC_LABELS = {
    "llm_response_time": "llm_response_time",
    "pure_llm_time": "pure_llm_time",
    "action_time": "action_time",
    "turn_time": "turn_time",
}

_TASK_TIMING_METRICS = (
    "benchmark.task.duration_ms",
    "benchmark.task.run.duration_ms",
    "benchmark.task.queue_wait_ms",
)

_TASK_TIMING_LABELS = {
    "benchmark.task.duration_ms": "end_to_end",
    "benchmark.task.run.duration_ms": "pure_run",
    "benchmark.task.queue_wait_ms": "queue_wait",
}

def _format_window_seconds(window_size_seconds: float) -> str:
    rounded = round(window_size_seconds)
    if abs(window_size_seconds - rounded) < 1e-9:
        return f"{int(rounded)}s"
    return f"{window_size_seconds:g}s"


def _svg_bar_chart(
    title: str,
    rows: list[tuple[str, float]],
    *,
    width: int = 900,
    row_height: int = 32,
    log_scale: bool = False,
    formatter=_format_number,
) -> str:
    if not rows:
        return f"<section><h3>{escape(title)}</h3><p>No data.</p></section>"
    max_label = max(len(label) for label, _ in rows)
    left_pad = min(480, max(200, max_label * 11))
    max_value_text_len = max(len(formatter(value)) for _, value in rows)
    value_text_reserve = max(130, max_value_text_len * 12 + 28)
    bar_area = max(40, width - left_pad - value_text_reserve)
    height = 44 + row_height * len(rows)

    if log_scale:
        log_values = [math.log10(1.0 + max(0.0, v)) for _, v in rows]
        max_log = max(log_values) or 1.0
    else:
        max_value = max(value for _, value in rows) or 1.0
        sorted_values = sorted((v for _, v in rows), reverse=True)
        median_value = sorted_values[len(sorted_values) // 2] if sorted_values else max_value
        crop_threshold = max(median_value * 5.0, max_value * 0.15) if median_value > 0 else max_value
        use_crop = max_value > crop_threshold and crop_threshold > 0

    pieces = [
        f"<svg style='width:100%;max-width:{width}px;height:auto' viewBox='0 0 {width} {height}' preserveAspectRatio='xMinYMin meet' role='img' aria-label='{escape(title)}'>",
        "<defs><pattern id='crop-hatch' width='6' height='6' patternUnits='userSpaceOnUse' patternTransform='rotate(45)'>"
        "<rect width='2' height='6' fill='#93c5fd'/></pattern></defs>",
        "<style>"
        ".axis-label{font:19px sans-serif;fill:#334155}"
        ".bar{fill:#2563eb}"
        ".bar-cropped{fill:url(#crop-hatch);stroke:#2563eb;stroke-width:0.5}"
        ".bar-text{font:17px sans-serif;fill:#0f172a}"
        "</style>",
    ]
    y = 28
    for label, value in rows:
        if log_scale:
            scaled = 0.0 if max_log <= 0 else (math.log10(1.0 + max(0.0, value)) / max_log) * bar_area
            cropped = False
        else:
            if use_crop and value > crop_threshold:
                scaled = bar_area
                cropped = True
            else:
                reference = crop_threshold if use_crop else max_value
                scaled = 0.0 if reference <= 0 else (value / reference) * bar_area
                cropped = False

        pieces.append(f"<text class='axis-label' x='8' y='{y}'>{escape(label)}</text>")
        if cropped:
            solid_frac = crop_threshold / value if value > 0 else 1.0
            solid_w = bar_area * solid_frac
            pieces.append(
                f"<rect class='bar' x='{left_pad}' y='{y - 13}' width='{solid_w:.2f}' height='16' rx='3'></rect>"
            )
            pieces.append(
                f"<rect class='bar-cropped' x='{left_pad + solid_w:.2f}' y='{y - 13}'"
                f" width='{bar_area - solid_w:.2f}' height='16' rx='0 3 3 0'></rect>"
            )
        else:
            pieces.append(
                f"<rect class='bar' x='{left_pad}' y='{y - 13}' width='{scaled:.2f}' height='16' rx='3'></rect>"
            )
        text_x = left_pad + scaled + 8
        pieces.append(
            f"<text class='bar-text' x='{text_x:.2f}' y='{y}'>{escape(formatter(value))}</text>"
        )
        y += row_height
    pieces.append("</svg>")
    return "".join(pieces)


def _svg_line_chart(
    title: str,
    series: list[tuple[str, list[tuple[float, float]]]],
    *,
    width: int = 980,
    height: int = 360,
    formatter=_format_number,
    show_points: bool = False,
) -> str:
    non_empty = [(label, points) for label, points in series if points]
    if not non_empty:
        return f"<section><h3>{escape(title)}</h3><p>No data.</p></section>"
    left_pad = 64
    right_pad = 24
    legend_left = left_pad + 12
    legend_right = width - right_pad - 12
    legend_item_padding = 54
    legend_row_height = 26
    legend_layout: list[list[tuple[str, str]]] = []
    current_row: list[tuple[str, str]] = []
    current_width = 0.0
    colors = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]
    for index, (label, _) in enumerate(non_empty):
        color = colors[index % len(colors)]
        item_width = 52 + len(label) * 8.5
        if current_row and legend_left + current_width + item_width > legend_right:
            legend_layout.append(current_row)
            current_row = []
            current_width = 0.0
        current_row.append((label, color))
        current_width += item_width + legend_item_padding
    if current_row:
        legend_layout.append(current_row)
    legend_height = 20 + len(legend_layout) * legend_row_height
    top_pad = 24 + legend_height
    bottom_pad = 36
    chart_width = width - left_pad - right_pad
    chart_height = height - top_pad - bottom_pad
    max_x = max(point[0] for _, points in non_empty for point in points) or 1.0
    max_y = max(point[1] for _, points in non_empty for point in points)
    max_y = max(1.0, max_y)
    pieces = [
        f"<svg style='width:100%;max-width:{width}px;height:auto' viewBox='0 0 {width} {height}' preserveAspectRatio='xMinYMin meet' role='img' aria-label='{escape(title)}'>",
        "<style>"
        ".axis{stroke:#94a3b8;stroke-width:1}"
        ".grid{stroke:#e2e8f0;stroke-width:1}"
        ".series{fill:none;stroke-width:2.5}"
        ".legend-bg{fill:#ffffff;stroke:#cbd5e1;stroke-width:1}"
        ".legend-line{stroke-width:3}"
        ".legend{font:16px sans-serif;fill:#0f172a}"
        ".tick{font:15px sans-serif;fill:#475569}"
        ".point{stroke:#ffffff;stroke-width:1.5}"
        "</style>",
    ]
    pieces.append(
        f"<rect class='legend-bg' x='{left_pad}' y='12' width='{chart_width:.2f}' height='{legend_height:.2f}' rx='12'></rect>"
    )
    for row_index, row in enumerate(legend_layout):
        cursor_x = legend_left
        y = 32 + row_index * legend_row_height
        for label, color in row:
            pieces.append(
                f"<line class='legend-line' x1='{cursor_x}' y1='{y - 5}' x2='{cursor_x + 22}' y2='{y - 5}' stroke='{color}'></line>"
            )
            pieces.append(
                f"<circle cx='{cursor_x + 11}' cy='{y - 5}' r='3.5' fill='{color}' stroke='#ffffff' stroke-width='1'></circle>"
            )
            pieces.append(f"<text class='legend' x='{cursor_x + 30}' y='{y}'>{escape(label)}</text>")
            cursor_x += 52 + len(label) * 8.5 + legend_item_padding

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top_pad + chart_height - chart_height * fraction
        value = max_y * fraction
        pieces.append(f"<line class='grid' x1='{left_pad}' y1='{y:.2f}' x2='{width - right_pad}' y2='{y:.2f}'></line>")
        pieces.append(f"<text class='tick' x='4' y='{y + 4:.2f}'>{escape(formatter(value))}</text>")
    pieces.append(f"<line class='axis' x1='{left_pad}' y1='{top_pad}' x2='{left_pad}' y2='{top_pad + chart_height}'></line>")
    pieces.append(f"<line class='axis' x1='{left_pad}' y1='{top_pad + chart_height}' x2='{width - right_pad}' y2='{top_pad + chart_height}'></line>")
    for index, (label, points) in enumerate(non_empty):
        color = colors[index % len(colors)]
        path_parts = []
        for point_index, (offset_seconds, value) in enumerate(points):
            x = left_pad + (offset_seconds / max_x) * chart_width if max_x > 0 else left_pad
            y = top_pad + chart_height - (value / max_y) * chart_height if max_y > 0 else top_pad + chart_height
            command = "M" if point_index == 0 else "L"
            path_parts.append(f"{command}{x:.2f},{y:.2f}")
        pieces.append(f"<path class='series' stroke='{color}' d='{' '.join(path_parts)}'></path>")
        if show_points:
            for offset_seconds, value in points:
                x = left_pad + (offset_seconds / max_x) * chart_width if max_x > 0 else left_pad
                y = top_pad + chart_height - (value / max_y) * chart_height if max_y > 0 else top_pad + chart_height
                pieces.append(f"<circle class='point' cx='{x:.2f}' cy='{y:.2f}' r='3.5' fill='{color}'></circle>")
    pieces.append(f"<text class='tick' x='{left_pad}' y='{height - 8}'>0s</text>")
    pieces.append(f"<text class='tick' x='{width - right_pad - 36}' y='{height - 8}'>{max_x:.1f}s</text>")
    pieces.append("</svg>")
    return "".join(pieces)


def _svg_cdf_chart(
    title: str,
    points: list[CDFPoint],
    *,
    width: int = 900,
    height: int = 320,
    formatter=_format_ms,
) -> str:
    if not points:
        return f"<section><h3>{escape(title)}</h3><p>No data.</p></section>"
    left_pad = 72
    right_pad = 24
    top_pad = 20
    bottom_pad = 36
    chart_width = width - left_pad - right_pad
    chart_height = height - top_pad - bottom_pad
    max_x = max((point.value for point in points), default=1.0)
    max_x = max(1.0, max_x)
    pieces = [
        f"<svg style='width:100%;max-width:{width}px;height:auto' viewBox='0 0 {width} {height}' preserveAspectRatio='xMinYMin meet' role='img' aria-label='{escape(title)}'>",
        "<style>"
        ".axis{stroke:#94a3b8;stroke-width:1}"
        ".grid{stroke:#e2e8f0;stroke-width:1}"
        ".cdf{fill:none;stroke:#2563eb;stroke-width:2.5}"
        ".tick{font:15px sans-serif;fill:#475569}"
        "</style>",
    ]
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top_pad + chart_height - chart_height * fraction
        pieces.append(f"<line class='grid' x1='{left_pad}' y1='{y:.2f}' x2='{width - right_pad}' y2='{y:.2f}'></line>")
        pieces.append(f"<text class='tick' x='4' y='{y + 4:.2f}'>{escape(_format_percent(fraction))}</text>")
    pieces.append(f"<line class='axis' x1='{left_pad}' y1='{top_pad}' x2='{left_pad}' y2='{top_pad + chart_height}'></line>")
    pieces.append(f"<line class='axis' x1='{left_pad}' y1='{top_pad + chart_height}' x2='{width - right_pad}' y2='{top_pad + chart_height}'></line>")
    path_parts = []
    for index, point in enumerate(points):
        x = left_pad + (point.value / max_x) * chart_width if max_x > 0 else left_pad
        y = top_pad + chart_height - point.cumulative_probability * chart_height
        path_parts.append(f"{'M' if index == 0 else 'L'}{x:.2f},{y:.2f}")
    pieces.append(f"<path class='cdf' d='{' '.join(path_parts)}'></path>")
    pieces.append(f"<text class='tick' x='{left_pad}' y='{height - 8}'>0</text>")
    pieces.append(
        f"<text class='tick' x='{width - right_pad - 72}' y='{height - 8}'>{escape(formatter(max_x))}</text>"
    )
    pieces.append("</svg>")
    return "".join(pieces)


def _counter_window_series(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    windowed: list[tuple[float, float]] = []
    previous_value: float | None = None
    for offset_seconds, value in points:
        if previous_value is None:
            windowed_value = 0.0
        else:
            windowed_value = max(0.0, value - previous_value)
        windowed.append((offset_seconds, windowed_value))
        previous_value = value
    return windowed


def _aggregate_window_series(
    points: list[tuple[float, float]],
    *,
    window_size_seconds: float | None,
) -> list[tuple[float, float]]:
    if window_size_seconds is None or window_size_seconds <= 0 or len(points) <= 1:
        return list(points)
    aggregated: list[tuple[float, float]] = []
    current_bucket: int | None = None
    current_sum = 0.0
    current_count = 0
    last_offset = 0.0
    for offset_seconds, value in points:
        bucket = int(offset_seconds // window_size_seconds)
        if current_bucket is None:
            current_bucket = bucket
        if bucket != current_bucket:
            aggregated.append((last_offset, current_sum / current_count))
            current_bucket = bucket
            current_sum = 0.0
            current_count = 0
        current_sum += value
        current_count += 1
        last_offset = offset_seconds
    if current_count:
        aggregated.append((last_offset, current_sum / current_count))
    return aggregated


def _window_agg_note(window_size_seconds: float | None) -> str:
    if window_size_seconds is None or window_size_seconds <= 0:
        return ""
    return f"<p>Line charts in this section are averaged within {_format_window_seconds(window_size_seconds)} windows.</p>"


def _html_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    header_cells = "".join(f"<th>{escape(str(header))}</th>" for header in headers)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{escape(str(cell))}</td>" for cell in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return (
        "<table>"
        f"<thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _operation_rows(items: list[MetricSummary]) -> list[tuple[str, float]]:
    return [(_format_metric_label(item.metric_name), item.total_ms) for item in items]


def _operation_tail_rows(items: list[MetricSummary]) -> list[tuple[str, float]]:
    return [(_format_metric_label(item.metric_name), item.p95_ms) for item in items]


def _task_metric_rows(tasks: list[TaskSummary], metric_name: str) -> list[tuple[str, float]]:
    rows = [
        (task.sandbox_id, task.metrics.get(metric_name, 0.0))
        for task in tasks
        if metric_name in task.metrics
    ]
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows


def _format_overhead_metric_label(metric_name: str) -> str:
    return _OVERHEAD_METRIC_LABELS.get(metric_name, _format_metric_label(metric_name))


def _format_turn_metric_label(metric_name: str) -> str:
    return _TURN_METRIC_LABELS.get(metric_name, metric_name)


def _format_task_timing_metric_label(metric_name: str) -> str:
    return _TASK_TIMING_LABELS.get(metric_name, _format_metric_label(metric_name))
def _phase_rows_for_csv(items: list[PhaseSummary]) -> list[dict[str, object]]:
    return [
        {
            "phase": item.phase,
            "started_at": item.started_at,
            "finished_at": item.finished_at,
            "duration_ms": f"{item.duration_ms:.6f}",
            "configured_max_workers": "" if item.configured_max_workers is None else item.configured_max_workers,
            "sandbox_count": "" if item.sandbox_count is None else item.sandbox_count,
            "status": item.status,
        }
        for item in items
    ]


def _load_jobs_series(points, *, window_size_seconds: float | None = None) -> list[tuple[str, list[tuple[float, float]]]]:
    return [
        (
            "active jobs",
            _aggregate_window_series(
                [(point.offset_seconds, float(point.active_jobs)) for point in points],
                window_size_seconds=window_size_seconds,
            ),
        ),
        (
            "active process jobs",
            _aggregate_window_series(
                [(point.offset_seconds, float(point.active_process_jobs)) for point in points if point.active_process_jobs or point.active_jobs],
                window_size_seconds=window_size_seconds,
            ),
        ),
        (
            "active filesystem jobs",
            _aggregate_window_series(
                [(point.offset_seconds, float(point.active_filesystem_jobs)) for point in points if point.active_filesystem_jobs or point.active_jobs],
                window_size_seconds=window_size_seconds,
            ),
        ),
    ]


def _load_io_series(points, *, window_size_seconds: float | None = None) -> list[tuple[str, list[tuple[float, float]]]]:
    return [
        (
            "estimated io bytes",
            _aggregate_window_series(
                [(point.offset_seconds, point.active_estimated_io_bytes) for point in points],
                window_size_seconds=window_size_seconds,
            ),
        )
    ]


def _overhead_series(
    overhead: OverheadAnalysisSummary,
    *,
    window_size_seconds: float | None = None,
) -> list[tuple[str, list[tuple[float, float]]]]:
    return [
        (
            _format_overhead_metric_label(metric_name),
            _aggregate_window_series(
                [(point.offset_seconds, point.value) for point in points],
                window_size_seconds=window_size_seconds,
            ),
        )
        for metric_name, points in overhead.time_series.items()
        if points
    ]


def _turn_series(
    turn_analysis: TurnAnalysisSummary,
    *,
    window_size_seconds: float | None = None,
) -> list[tuple[str, list[tuple[float, float]]]]:
    return [
        (
            _format_turn_metric_label(metric_name),
            _aggregate_window_series(
                [(point.offset_seconds, point.value) for point in points],
                window_size_seconds=window_size_seconds,
            ),
        )
        for metric_name, points in turn_analysis.time_series.items()
        if points
    ]
_CHECKPOINT_LATENCY_FIGURES = (
    ("checkpoint_flow_latency.svg", "Checkpoint Latency Over Time", "checkpoint.flow.duration_ms", "checkpoint"),
    (
        "checkpoint_process_latency.svg",
        "Process Checkpoint Latency Over Time",
        "checkpoint.process.duration_ms",
        "process checkpoint",
    ),
    (
        "checkpoint_filesystem_latency.svg",
        "Filesystem Checkpoint Latency Over Time",
        "checkpoint.filesystem.duration_ms",
        "filesystem checkpoint",
    ),
)

_RESTORE_LATENCY_FIGURES = (
    ("restore_flow_latency.svg", "Restore Latency Over Time", "restore.flow.duration_ms", "restore"),
    ("restore_process_latency.svg", "Process Restore Latency Over Time", "restore.process.duration_ms", "process restore"),
    (
        "restore_filesystem_latency.svg",
        "Filesystem Restore Latency Over Time",
        "restore.filesystem.duration_ms",
        "filesystem restore",
    ),
)


def _latency_series(
    points,
    label: str,
    *,
    window_size_seconds: float | None = None,
) -> list[tuple[str, list[tuple[float, float]]]]:
    return [
        (
            label,
            _aggregate_window_series(
                [(point.offset_seconds, point.value) for point in points],
                window_size_seconds=window_size_seconds,
            ),
        )
    ]


def _resource_host_figures(resource: ResourceAnalysisSummary) -> dict[str, str]:
    figures: dict[str, str] = {}
    for metric_name, title, formatter in (
        ("resource.host.cpu.usage_percent", "Host CPU Usage Over Time", _format_number),
        ("resource.host.memory.used_bytes", "Host Memory Usage Over Time", _format_bytes),
    ):
        points = resource.host_series.get(metric_name, [])
        figures[metric_name] = _svg_line_chart(
            title,
            [(metric_name.replace("resource.host.", ""), [(point.offset_seconds, point.value) for point in points])],
            formatter=formatter,
        )
    disk_series = [
        (
            "read bytes / window",
            _counter_window_series(
                [(point.offset_seconds, point.value) for point in resource.host_series.get("resource.host.disk.read_bytes", [])]
            ),
        ),
        (
            "write bytes / window",
            _counter_window_series(
                [(point.offset_seconds, point.value) for point in resource.host_series.get("resource.host.disk.write_bytes", [])]
            ),
        ),
    ]
    figures["resource.host.disk"] = _svg_line_chart(
        "Host Disk I/O Per Sample Window",
        disk_series,
        formatter=_format_bytes,
    )
    network_series = [
        (
            "rx bytes / window",
            _counter_window_series(
                [(point.offset_seconds, point.value) for point in resource.host_series.get("resource.host.network.rx_bytes", [])]
            ),
        ),
        (
            "tx bytes / window",
            _counter_window_series(
                [(point.offset_seconds, point.value) for point in resource.host_series.get("resource.host.network.tx_bytes", [])]
            ),
        ),
    ]
    figures["resource.host.network"] = _svg_line_chart(
        "Host Network I/O Per Sample Window",
        network_series,
        formatter=_format_bytes,
    )
    return figures


def _sandbox_peak_rows(resource: ResourceAnalysisSummary, metric_name: str) -> list[tuple[str, float]]:
    rows = [
        (summary.sandbox_id, summary.max_value)
        for summary in resource.sandbox_metrics
        if summary.metric_name == metric_name
    ]
    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:12]


def build_figure_svgs(
    analysis: TelemetryAnalysis,
    *,
    log_scale: bool = False,
    window_size_seconds: float | None = None,
) -> dict[str, str]:
    figures = {
        "top_total_time.svg": _svg_bar_chart(
            "Top Operations By Cumulative Time (ms)",
            _operation_rows(analysis.top_total_time_operations[:15]),
            log_scale=log_scale,
        ),
        "top_invocation.svg": _svg_bar_chart(
            "Top Operations By Invocation Count",
            [(_format_metric_label(item.metric_name), float(item.count)) for item in analysis.top_invocation_operations[:15]],
            log_scale=log_scale,
        ),
        "top_tail_latency.svg": _svg_bar_chart(
            "Top Operations By P95 Latency (ms)",
            _operation_tail_rows(analysis.top_tail_latency_operations[:15]),
            log_scale=log_scale,
        ),
        "task_latency.svg": _svg_bar_chart(
            "Sandbox End-To-End Latency (ms)",
            _task_metric_rows(analysis.task_summaries, "benchmark.task.duration_ms")[:20],
            width=980,
            log_scale=log_scale,
        ),
        "task_run_latency.svg": _svg_bar_chart(
            "Sandbox Pure Task Run Time (ms)",
            _task_metric_rows(analysis.task_summaries, "benchmark.task.run.duration_ms")[:20],
            width=980,
            log_scale=log_scale,
        ),
        "task_queue_wait.svg": _svg_bar_chart(
            "Sandbox Task Queue Wait (ms)",
            _task_metric_rows(analysis.task_summaries, "benchmark.task.queue_wait_ms")[:20],
            width=980,
            log_scale=log_scale,
        ),
        "llm_breakdown.svg": _svg_bar_chart(
            "Average LLM Path Breakdown (ms)",
            [(_format_metric_label(name), value) for name, value in analysis.llm_breakdown.items()],
            log_scale=log_scale,
        ),
        "checkpoint_breakdown.svg": _svg_bar_chart(
            "Average Checkpoint/Restore Breakdown (ms)",
            [(_format_metric_label(name), value) for name, value in analysis.checkpoint_breakdown.items()],
            width=980,
            log_scale=log_scale,
        ),
    }
    if analysis.checkpoint_analysis is not None:
        figures["checkpoint_load_jobs.svg"] = _svg_line_chart(
            "Checkpoint Load Over Time (Jobs)",
            _load_jobs_series(analysis.checkpoint_analysis.load_over_time, window_size_seconds=window_size_seconds),
        )
        figures["checkpoint_load_io.svg"] = _svg_line_chart(
            "Checkpoint Load Over Time (Estimated I/O)",
            _load_io_series(analysis.checkpoint_analysis.load_over_time, window_size_seconds=window_size_seconds),
            formatter=_format_bytes,
        )
        for figure_name, title, metric_name, label in _CHECKPOINT_LATENCY_FIGURES:
            figures[figure_name] = _svg_line_chart(
                title,
                _latency_series(
                    analysis.checkpoint_analysis.latency_over_time.get(metric_name, []),
                    label,
                    window_size_seconds=window_size_seconds,
                ),
                formatter=_format_ms,
                show_points=True,
            )
    if analysis.overhead_analysis is not None:
        figures["overhead_latency.svg"] = _svg_line_chart(
            "LLM Overhead Over Time",
            _overhead_series(analysis.overhead_analysis, window_size_seconds=window_size_seconds),
            formatter=_format_ms,
        )
    if analysis.turn_analysis is not None:
        figures["turn_timing.svg"] = _svg_line_chart(
            "Turn Timing Over Time",
            _turn_series(analysis.turn_analysis, window_size_seconds=window_size_seconds),
            formatter=_format_ms,
        )
        for metric_name, points in analysis.turn_analysis.cdf_series.items():
            figures[f"turn_{metric_name}_cdf.svg"] = _svg_cdf_chart(
                f"{_format_turn_metric_label(metric_name)} CDF",
                points,
                formatter=_format_ms,
            )
    if analysis.restore_analysis is not None:
        figures["restore_load_jobs.svg"] = _svg_line_chart(
            "Restore Load Over Time (Jobs)",
            _load_jobs_series(analysis.restore_analysis.load_over_time, window_size_seconds=window_size_seconds),
        )
        figures["restore_load_io.svg"] = _svg_line_chart(
            "Restore Load Over Time (Estimated I/O)",
            _load_io_series(analysis.restore_analysis.load_over_time, window_size_seconds=window_size_seconds),
            formatter=_format_bytes,
        )
        for figure_name, title, metric_name, label in _RESTORE_LATENCY_FIGURES:
            figures[figure_name] = _svg_line_chart(
                title,
                _latency_series(
                    analysis.restore_analysis.latency_over_time.get(metric_name, []),
                    label,
                    window_size_seconds=window_size_seconds,
                ),
                formatter=_format_ms,
                show_points=True,
            )
    if analysis.resource_analysis is not None:
        figures.update(
            {
                "resource_host_cpu.svg": _resource_host_figures(analysis.resource_analysis).get(
                    "resource.host.cpu.usage_percent",
                    "<p>No data.</p>",
                ),
                "resource_host_memory.svg": _resource_host_figures(analysis.resource_analysis).get(
                    "resource.host.memory.used_bytes",
                    "<p>No data.</p>",
                ),
                "resource_host_disk.svg": _resource_host_figures(analysis.resource_analysis).get(
                    "resource.host.disk",
                    "<p>No data.</p>",
                ),
                "resource_host_network.svg": _resource_host_figures(analysis.resource_analysis).get(
                    "resource.host.network",
                    "<p>No data.</p>",
                ),
                "resource_sandbox_cpu_peaks.svg": _svg_bar_chart(
                    "Sandbox CPU Peak Usage",
                    _sandbox_peak_rows(analysis.resource_analysis, "resource.sandbox.cpu.usage_percent"),
                    formatter=_format_number,
                ),
                "resource_sandbox_memory_peaks.svg": _svg_bar_chart(
                    "Sandbox Memory Peak Usage",
                    _sandbox_peak_rows(analysis.resource_analysis, "resource.sandbox.memory.peak_bytes"),
                    formatter=_format_bytes,
                ),
            }
        )
    return figures


def render_report_html(
    analysis: TelemetryAnalysis,
    *,
    log_scale: bool = False,
    window_size_seconds: float | None = None,
) -> str:
    figures = build_figure_svgs(analysis, log_scale=log_scale, window_size_seconds=window_size_seconds)
    summary_rows = [
        ("Input", analysis.input_path),
        ("Run ID", analysis.run_id),
        ("Started", analysis.started_at),
        ("Finished", analysis.finished_at),
        ("Records (All Phases)", analysis.total_records),
        ("Events (All Phases)", analysis.total_events),
        ("Metrics (All Phases)", analysis.total_metrics),
        ("Detail Scope", analysis.detail_scope_name),
        ("Detail Started", analysis.detail_started_at or analysis.started_at),
        ("Detail Finished", analysis.detail_finished_at or analysis.finished_at),
        ("Records (Detail Scope)", analysis.detail_total_records),
        ("Events (Detail Scope)", analysis.detail_total_events),
        ("Metrics (Detail Scope)", analysis.detail_total_metrics),
        ("Sandboxes", analysis.distinct_sandboxes),
        ("Tasks", analysis.distinct_tasks),
        ("Requests", analysis.distinct_requests),
        ("Checkpoint IDs", analysis.distinct_checkpoints),
        ("Job IDs", analysis.distinct_jobs),
    ]
    if analysis.exclude_failed_tasks:
        excluded_count = len(analysis.excluded_sandbox_task_pairs)
        excluded_detail = ", ".join(f"{sid}:{tid}" for sid, tid in analysis.excluded_sandbox_task_pairs) if analysis.excluded_sandbox_task_pairs else "none"
        summary_rows.append(("Excluded Failed Sandboxes", f"{excluded_count} ({excluded_detail})"))
    summary_table = _html_table(["Field", "Value"], summary_rows)
    phase_overview_table = _html_table(
        ["Phase", "Started", "Finished", "Duration", "Max Workers", "Sandboxes", "Status"],
        [
            (
                item.phase,
                item.started_at,
                item.finished_at,
                _format_ms(item.duration_ms),
                "" if item.configured_max_workers is None else item.configured_max_workers,
                "" if item.sandbox_count is None else item.sandbox_count,
                item.status,
            )
            for item in analysis.phase_overview
        ] if analysis.phase_overview else [("None", "", "", "", "", "", "")],
    )

    operation_table = _html_table(
        [
            "Metric",
            "Count",
            "Total (ms)",
            "Mean (ms)",
            "Min (ms)",
            "P25 (ms)",
            "P50 (ms)",
            "P95 (ms)",
            "P99 (ms)",
            "Max (ms)",
            "Failures",
            "Component",
        ],
        [
            (
                item.metric_name,
                item.count,
                f"{item.total_ms:.2f}",
                f"{item.mean_ms:.2f}",
                f"{item.min_ms:.2f}",
                f"{item.p25_ms:.2f}",
                f"{item.p50_ms:.2f}",
                f"{item.p95_ms:.2f}",
                f"{item.p99_ms:.2f}",
                f"{item.max_ms:.2f}",
                item.failure_count,
                item.component,
            )
            for item in analysis.top_total_time_operations[:20]
        ],
    )

    summary_by_name = {item.metric_name: item for item in analysis.operation_summaries}
    task_timing_items = [
        summary_by_name[metric_name]
        for metric_name in _TASK_TIMING_METRICS
        if metric_name in summary_by_name
    ]
    task_timing_table = _html_table(
        ["Metric", "Count", "Mean (ms)", "Median (ms)", "P95 (ms)", "Max (ms)"],
        [
            (
                _format_task_timing_metric_label(item.metric_name),
                item.count,
                f"{item.mean_ms:.2f}",
                f"{item.p50_ms:.2f}",
                f"{item.p95_ms:.2f}",
                f"{item.max_ms:.2f}",
            )
            for item in task_timing_items
        ] if task_timing_items else [("None", 0, "0.00", "0.00", "0.00", "0.00")],
    )

    task_metric_names = sorted({metric_name for task in analysis.task_summaries for metric_name in task.metrics})
    task_table = _html_table(
        ["Sandbox", "Task", "Agent", "LLM Service"] + task_metric_names,
        [
            [task.sandbox_id, task.task_id, task.agent_type, task.llm_service_type]
            + [f"{task.metrics.get(metric_name, 0.0):.2f}" for metric_name in task_metric_names]
            for task in analysis.task_summaries
        ],
    )

    slow_table = _html_table(
        ["Metric", "Value (ms)", "Task", "Sandbox", "Request", "Checkpoint", "Job", "Status", "Timestamp"],
        [
            (
                item.metric_name,
                f"{item.value_ms:.2f}",
                item.task_id,
                item.sandbox_id,
                item.request_id,
                item.checkpoint_id,
                item.job_id,
                item.status,
                item.timestamp,
            )
            for item in analysis.slowest_records[:25]
        ],
    )

    lifecycle_table = _html_table(
        ["Operation", "Starts", "Finishes", "Missing Finish", "Statuses"],
        [
            (
                item.operation_name,
                item.start_count,
                item.finish_count,
                item.missing_finish_count,
                json.dumps(item.finish_status_counts, sort_keys=True),
            )
            for item in analysis.lifecycle_gaps[:25]
        ] if analysis.lifecycle_gaps else [("None", 0, 0, 0, "{}")],
    )

    checkpoint_section = "<section><h2>Checkpoint Analysis</h2><p>No checkpoint data.</p></section>"
    if analysis.checkpoint_analysis is not None:
        checkpoint = analysis.checkpoint_analysis
        checkpoint_note = _window_agg_note(window_size_seconds)
        checkpoint_summary = _html_table(
            ["Field", "Value"],
            [
                ("Total Checkpoints", checkpoint.total_count),
                ("Succeeded", checkpoint.success_count),
                ("Failed", checkpoint.fail_count),
                ("Skipped", checkpoint.skip_count),
                ("Fail Rate", f"{checkpoint.fail_rate:.2%}"),
                ("Skip Rate", f"{checkpoint.skip_rate:.2%}"),
                ("Rate Per Minute", f"{checkpoint.overall_rate_per_minute:.2f}"),
                ("Process Frequency / Min", f"{checkpoint.process_frequency_per_minute:.2f}"),
                ("Filesystem Frequency / Min", f"{checkpoint.filesystem_frequency_per_minute:.2f}"),
                ("Scope Counts", json.dumps(checkpoint.scope_counts, sort_keys=True)),
                ("Scope Rates", json.dumps({k: round(v, 4) for k, v in checkpoint.scope_rates.items()}, sort_keys=True)),
                ("Filesystem-Only / Full", f"{checkpoint.filesystem_only_to_full_ratio:.2f}"),
                ("Promoted Process Count", checkpoint.promoted_process_checkpoint_count),
                ("Total Process Size", _format_bytes(checkpoint.total_process_size_bytes)),
                ("Total Filesystem Written", _format_bytes(checkpoint.total_filesystem_written_bytes)),
                ("Total Estimated I/O", _format_bytes(checkpoint.total_estimated_io_bytes)),
            ],
        )
        checkpoint_task_table = _html_table(
            ["Sandbox", "Task", "Count", "Succeeded", "Skipped", "Failed", "Success Rate", "Rate/Min", "Mean Estimated I/O"],
            [
                (
                    item.sandbox_id,
                    item.task_id,
                    item.total_count,
                    item.success_count,
                    item.skip_count,
                    item.fail_count,
                    f"{item.success_rate:.2%}",
                    f"{item.per_minute_rate:.2f}",
                    _format_bytes(item.mean_estimated_io_bytes),
                )
                for item in checkpoint.per_task
            ],
        )
        checkpoint_section = (
            "<section><h2>Checkpoint Analysis</h2>"
            f"{checkpoint_note}"
            f"{checkpoint_summary}"
            "<div class='grid'>"
            f"<section><h3>Checkpoint Load Over Time</h3>{figures.get('checkpoint_load_jobs.svg', '<p>No data.</p>')}</section>"
            f"<section><h3>Checkpoint I/O Load Over Time</h3>{figures.get('checkpoint_load_io.svg', '<p>No data.</p>')}</section>"
            "</div>"
            "<div class='grid'>"
            f"<section><h3>Checkpoint Latency Over Time</h3>{figures.get('checkpoint_flow_latency.svg', '<p>No data.</p>')}</section>"
            f"<section><h3>Process Checkpoint Latency Over Time</h3>{figures.get('checkpoint_process_latency.svg', '<p>No data.</p>')}</section>"
            f"<section><h3>Filesystem Checkpoint Latency Over Time</h3>{figures.get('checkpoint_filesystem_latency.svg', '<p>No data.</p>')}</section>"
            "</div>"
            "<section><h3>Checkpoint Per Sandbox</h3>"
            f"{checkpoint_task_table}"
            "</section>"
            "</section>"
        )

    overhead_section = "<section><h2>Overhead Analysis</h2><p>No overhead data.</p></section>"
    if analysis.overhead_analysis is not None and analysis.overhead_analysis.metrics:
        overhead = analysis.overhead_analysis
        overhead_note = _window_agg_note(window_size_seconds)
        overhead_summary = _html_table(
            ["Metric", "Mean (ms)", "Median (ms)", "Min (ms)", "Max (ms)", "P95 (ms)", "P99 (ms)"],
            [
                (
                    _format_overhead_metric_label(item.metric_name),
                    f"{item.mean_ms:.2f}",
                    f"{item.p50_ms:.2f}",
                    f"{item.min_ms:.2f}",
                    f"{item.max_ms:.2f}",
                    f"{item.p95_ms:.2f}",
                    f"{item.p99_ms:.2f}",
                )
                for item in overhead.metrics
            ],
        )
        overhead_section = (
            "<section><h2>Overhead Analysis</h2>"
            f"{overhead_note}"
            f"{overhead_summary}"
            "<section><h3>Window-Aggregated Overhead Over Time</h3>"
            f"{figures.get('overhead_latency.svg', '<p>No data.</p>')}"
            "</section>"
            "</section>"
        )

    turn_section = "<section><h2>Turn Analysis</h2><p>No turn timing data.</p></section>"
    if analysis.turn_analysis is not None and analysis.turn_analysis.metrics:
        turn = analysis.turn_analysis
        turn_note = _window_agg_note(window_size_seconds)
        turn_summary = _html_table(
            ["Metric", "Count", "Mean (ms)", "Median (ms)", "P95 (ms)", "P99 (ms)", "Min (ms)", "Max (ms)"],
            [
                (
                    _format_turn_metric_label(item.metric_name),
                    item.count,
                    f"{item.mean_ms:.2f}",
                    f"{item.p50_ms:.2f}",
                    f"{item.p95_ms:.2f}",
                    f"{item.p99_ms:.2f}",
                    f"{item.min_ms:.2f}",
                    f"{item.max_ms:.2f}",
                )
                for item in turn.metrics
            ],
        )
        cdf_blocks = "".join(
            f"<section><h3>{escape(_format_turn_metric_label(item.metric_name))} CDF</h3>"
            f"{figures.get(f'turn_{item.metric_name}_cdf.svg', '<p>No data.</p>')}"
            "</section>"
            for item in turn.metrics
        )
        turn_section = (
            "<section><h2>Turn Analysis</h2>"
            f"{turn_note}"
            f"{turn_summary}"
            "<section><h3>Window-Aggregated Turn Timing Over Time</h3>"
            f"{figures.get('turn_timing.svg', '<p>No data.</p>')}"
            "</section>"
            "<div class='grid'>"
            f"{cdf_blocks}"
            "</div>"
            "</section>"
        )
    restore_section = "<section><h2>Restore Analysis</h2><p>No restore data.</p></section>"
    if analysis.restore_analysis is not None:
        restore = analysis.restore_analysis
        restore_note = _window_agg_note(window_size_seconds)
        restore_summary = _html_table(
            ["Field", "Value"],
            [
                ("Total Restores", restore.total_count),
                ("Succeeded", restore.success_count),
                ("Failed", restore.fail_count),
                ("Skipped", restore.skip_count),
                ("Fail Rate", f"{restore.fail_rate:.2%}"),
                ("Skip Rate", f"{restore.skip_rate:.2%}"),
                ("Rate Per Minute", f"{restore.overall_rate_per_minute:.2f}"),
                ("Mixed Source Count", restore.mixed_source_count),
                ("Mixed Source Rate", f"{restore.mixed_source_rate:.2%}"),
                ("Mean Estimated I/O", _format_bytes(restore.mean_estimated_io_bytes)),
                ("Mean Source Gap (turns)", f"{restore.mean_source_gap_turns:.2f}"),
                ("P95 Source Gap (turns)", f"{restore.p95_source_gap_turns:.2f}"),
                ("Max Source Gap (turns)", f"{restore.max_source_gap_turns:.2f}"),
                ("Mean Source Gap (ms)", _format_ms(restore.mean_source_gap_ms)),
                ("P95 Source Gap (ms)", _format_ms(restore.p95_source_gap_ms)),
                ("Max Source Gap (ms)", _format_ms(restore.max_source_gap_ms)),
            ],
        )
        restore_task_table = _html_table(
            ["Sandbox", "Task", "Count", "Succeeded", "Skipped", "Failed", "Success Rate", "Rate/Min", "Mean Gap Turns", "Mean Gap Ms"],
            [
                (
                    item.sandbox_id,
                    item.task_id,
                    item.total_count,
                    item.success_count,
                    item.skip_count,
                    item.fail_count,
                    f"{item.success_rate:.2%}",
                    f"{item.per_minute_rate:.2f}",
                    f"{item.mean_source_gap_turns:.2f}",
                    _format_ms(item.mean_source_gap_ms),
                )
                for item in restore.per_task
            ],
        )
        restore_section = (
            "<section><h2>Restore Analysis</h2>"
            f"{restore_note}"
            f"{restore_summary}"
            "<div class='grid'>"
            f"<section><h3>Restore Load Over Time</h3>{figures.get('restore_load_jobs.svg', '<p>No data.</p>')}</section>"
            f"<section><h3>Restore I/O Load Over Time</h3>{figures.get('restore_load_io.svg', '<p>No data.</p>')}</section>"
            "</div>"
            "<div class='grid'>"
            f"<section><h3>Restore Latency Over Time</h3>{figures.get('restore_flow_latency.svg', '<p>No data.</p>')}</section>"
            f"<section><h3>Process Restore Latency Over Time</h3>{figures.get('restore_process_latency.svg', '<p>No data.</p>')}</section>"
            f"<section><h3>Filesystem Restore Latency Over Time</h3>{figures.get('restore_filesystem_latency.svg', '<p>No data.</p>')}</section>"
            "</div>"
            "<section><h3>Restore Per Sandbox</h3>"
            f"{restore_task_table}"
            "</section>"
            "</section>"
        )

    resource_section = "<section><h2>Resource Usage</h2><p>No resource data.</p></section>"
    if analysis.resource_analysis is not None:
        resource = analysis.resource_analysis
        sandbox_cpu_block = figures.get("resource_sandbox_cpu_peaks.svg", "<p>No data.</p>")
        if not resource.coverage.get("sandbox_cpu", False):
            sandbox_cpu_block = "<p>No sandbox CPU metrics were captured in this telemetry input.</p>"
        sandbox_memory_block = figures.get("resource_sandbox_memory_peaks.svg", "<p>No data.</p>")
        if not resource.coverage.get("sandbox_memory", False):
            sandbox_memory_block = "<p>No sandbox memory metrics were captured in this telemetry input.</p>"
        coverage_table = _html_table(
            ["Metric Family", "Available"],
            [(name, "yes" if available else "no") for name, available in sorted(resource.coverage.items())],
        )
        host_metric_table = _html_table(
            ["Metric", "Samples", "Mean", "Max"],
            [
                (
                    item.metric_name,
                    item.sample_count,
                    _format_number(item.mean_value) if "cpu.usage_percent" in item.metric_name else _format_bytes(item.mean_value),
                    _format_number(item.max_value) if "cpu.usage_percent" in item.metric_name else _format_bytes(item.max_value),
                )
                for item in resource.host_metrics
            ],
        )
        sandbox_metric_table = (
            _html_table(
                ["Metric", "Sandbox", "Samples", "Mean", "Max"],
                [
                    (
                        item.metric_name,
                        item.sandbox_id,
                        item.sample_count,
                        _format_number(item.mean_value) if "usage_percent" in item.metric_name else _format_bytes(item.mean_value),
                        _format_number(item.max_value) if "usage_percent" in item.metric_name else _format_bytes(item.max_value),
                    )
                    for item in resource.sandbox_metrics[:50]
                ],
            )
            if resource.sandbox_metrics
            else "<p>No sandbox resource metrics were captured in this telemetry input.</p>"
        )
        resource_section = (
            "<section><h2>Resource Usage</h2>"
            "<div class='grid'>"
            f"<section><h3>Coverage</h3>{coverage_table}</section>"
            f"<section><h3>Host Summary</h3>{host_metric_table}</section>"
            "</div>"
            "<div class='grid'>"
            f"<section><h3>Host CPU</h3>{figures.get('resource_host_cpu.svg', '<p>No data.</p>')}</section>"
            f"<section><h3>Host Memory</h3>{figures.get('resource_host_memory.svg', '<p>No data.</p>')}</section>"
            "</div>"
            "<div class='grid'>"
            f"<section><h3>Host Disk I/O</h3>{figures.get('resource_host_disk.svg', '<p>No data.</p>')}</section>"
            f"<section><h3>Host Network I/O</h3>{figures.get('resource_host_network.svg', '<p>No data.</p>')}</section>"
            "</div>"
            "<div class='grid'>"
            f"<section><h3>Sandbox CPU Peaks</h3>{sandbox_cpu_block}</section>"
            f"<section><h3>Sandbox Memory Peaks</h3>{sandbox_memory_block}</section>"
            "</div>"
            "<section><h3>Sandbox Breakdown</h3>"
            f"{sandbox_metric_table}"
            "</section>"
            "</section>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Telemetry Report</title>
  <style>
    body {{
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #0f172a;
      background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
      margin: 0;
      padding: 0;
    }}
    main {{
      max-width: 1440px;
      margin: 0 auto;
      padding: 32px;
    }}
    h1, h2, h3 {{
      margin: 0 0 12px;
    }}
    p {{
      line-height: 1.5;
    }}
    section {{
      background: rgba(255, 255, 255, 0.88);
      border: 1px solid #cbd5e1;
      border-radius: 16px;
      padding: 20px;
      margin: 0 0 20px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
      overflow-x: auto;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 15px;
    }}
    th, td {{
      border-bottom: 1px solid #e2e8f0;
      text-align: left;
      padding: 8px 10px;
      vertical-align: top;
    }}
    th {{
      background: #eff6ff;
      position: sticky;
      top: 0;
    }}
    code {{
      background: #e2e8f0;
      padding: 1px 5px;
      border-radius: 6px;
    }}
    @media (max-width: 1100px) {{
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <section>
      <h1>Agent-CR Telemetry Report</h1>
      <p>This report is generated from the JSONL telemetry stream using a streaming analyzer. It is intended for performance analysis, bottleneck diagnosis, long-tail latency inspection, checkpoint/restore characterization, and resource-usage analysis.</p>
      {summary_table}
    </section>
    <section>
      <h2>Phase Overview</h2>
      <p>Setup and verification are summarized only at the phase level here. Detailed sections below are filtered to {escape(analysis.detail_scope_name)}-phase telemetry only.</p>
      {phase_overview_table}
    </section>
    <div class="grid">
      <section><h2>Top Operations By Cumulative Time</h2>{figures['top_total_time.svg']}</section>
      <section><h2>Top Operations By Invocation Count</h2>{figures['top_invocation.svg']}</section>
    </div>
    <div class="grid">
      <section><h2>Top Operations By P95 Latency</h2>{figures['top_tail_latency.svg']}</section>
      <section><h2>Sandbox End-To-End Latency</h2>{figures['task_latency.svg']}</section>
    </div>
    <section>
      <h2>Task Timing Analysis</h2>
      {task_timing_table}
    </section>
    <div class="grid">
      <section><h2>Sandbox Pure Task Run Time</h2>{figures['task_run_latency.svg']}</section>
      <section><h2>Sandbox Task Queue Wait</h2>{figures['task_queue_wait.svg']}</section>
    </div>
    <div class="grid">
      <section><h2>Average LLM Path Breakdown</h2>{figures['llm_breakdown.svg']}</section>
      <section><h2>Average Checkpoint/Restore Breakdown</h2>{figures['checkpoint_breakdown.svg']}</section>
    </div>
    {checkpoint_section}
    {overhead_section}
    {turn_section}
    {restore_section}
    {resource_section}
    <section>
      <h2>Top Runtime Hotspots</h2>
      {operation_table}
    </section>
    <section>
      <h2>Per-Sandbox Summary</h2>
      {task_table}
    </section>
    <section>
      <h2>Slowest Recorded Operations</h2>
      {slow_table}
    </section>
    <section>
      <h2>Lifecycle Gaps</h2>
      <p>These are operations where <code>*.start</code> and <code>*.finish</code> counts do not match.</p>
      {lifecycle_table}
    </section>
  </main>
</body>
</html>"""


def _write_csv(path: Path, headers: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _operation_rows_for_csv(items: list[MetricSummary]) -> list[dict[str, object]]:
    return [
        {
            "metric_name": item.metric_name,
            "source_metric_name": item.source_metric_name,
            "category": item.category,
            "component": item.component,
            "count": item.count,
            "total_ms": f"{item.total_ms:.6f}",
            "mean_ms": f"{item.mean_ms:.6f}",
            "min_ms": f"{item.min_ms:.6f}",
            "p25_ms": f"{item.p25_ms:.6f}",
            "p50_ms": f"{item.p50_ms:.6f}",
            "p90_ms": f"{item.p90_ms:.6f}",
            "p95_ms": f"{item.p95_ms:.6f}",
            "p99_ms": f"{item.p99_ms:.6f}",
            "max_ms": f"{item.max_ms:.6f}",
            "success_count": item.success_count,
            "failure_count": item.failure_count,
            "unique_sandboxes": item.unique_sandboxes,
            "unique_tasks": item.unique_tasks,
        }
        for item in items
    ]


def _task_rows_for_csv(items: list[TaskSummary]) -> tuple[list[str], list[dict[str, object]]]:
    metric_names = sorted({metric_name for item in items for metric_name in item.metrics})
    headers = ["sandbox_id", "task_id", "agent_type", "llm_service_type"] + metric_names
    rows = []
    for item in items:
        row = {
            "sandbox_id": item.sandbox_id,
            "task_id": item.task_id,
            "agent_type": item.agent_type,
            "llm_service_type": item.llm_service_type,
        }
        for metric_name in metric_names:
            row[metric_name] = f"{item.metrics.get(metric_name, 0.0):.6f}"
        rows.append(row)
    return headers, rows


def _checkpoint_rows_for_csv(checkpoint: CheckpointAnalysisSummary | None) -> tuple[list[str], list[dict[str, object]]]:
    headers = ["metric", "value"]
    if checkpoint is None:
        return headers, []
    rows = [
        {"metric": "total_count", "value": checkpoint.total_count},
        {"metric": "success_count", "value": checkpoint.success_count},
        {"metric": "fail_count", "value": checkpoint.fail_count},
        {"metric": "skip_count", "value": checkpoint.skip_count},
        {"metric": "fail_rate", "value": f"{checkpoint.fail_rate:.6f}"},
        {"metric": "skip_rate", "value": f"{checkpoint.skip_rate:.6f}"},
        {"metric": "overall_rate_per_minute", "value": f"{checkpoint.overall_rate_per_minute:.6f}"},
        {"metric": "process_frequency_per_minute", "value": f"{checkpoint.process_frequency_per_minute:.6f}"},
        {"metric": "filesystem_frequency_per_minute", "value": f"{checkpoint.filesystem_frequency_per_minute:.6f}"},
        {"metric": "filesystem_only_to_full_ratio", "value": f"{checkpoint.filesystem_only_to_full_ratio:.6f}"},
        {"metric": "promoted_process_checkpoint_count", "value": checkpoint.promoted_process_checkpoint_count},
        {"metric": "total_process_size_bytes", "value": f"{checkpoint.total_process_size_bytes:.6f}"},
        {"metric": "total_filesystem_written_bytes", "value": f"{checkpoint.total_filesystem_written_bytes:.6f}"},
        {"metric": "total_estimated_io_bytes", "value": f"{checkpoint.total_estimated_io_bytes:.6f}"},
    ]
    for scope, count in sorted(checkpoint.scope_counts.items()):
        rows.append({"metric": f"scope_count.{scope}", "value": count})
    for scope, rate in sorted(checkpoint.scope_rates.items()):
        rows.append({"metric": f"scope_rate.{scope}", "value": f"{rate:.6f}"})
    return headers, rows


def _restore_rows_for_csv(restore: RestoreAnalysisSummary | None) -> tuple[list[str], list[dict[str, object]]]:
    headers = ["metric", "value"]
    if restore is None:
        return headers, []
    rows = [
        {"metric": "total_count", "value": restore.total_count},
        {"metric": "success_count", "value": restore.success_count},
        {"metric": "fail_count", "value": restore.fail_count},
        {"metric": "skip_count", "value": restore.skip_count},
        {"metric": "fail_rate", "value": f"{restore.fail_rate:.6f}"},
        {"metric": "skip_rate", "value": f"{restore.skip_rate:.6f}"},
        {"metric": "overall_rate_per_minute", "value": f"{restore.overall_rate_per_minute:.6f}"},
        {"metric": "mixed_source_count", "value": restore.mixed_source_count},
        {"metric": "mixed_source_rate", "value": f"{restore.mixed_source_rate:.6f}"},
        {"metric": "mean_estimated_io_bytes", "value": f"{restore.mean_estimated_io_bytes:.6f}"},
        {"metric": "mean_source_gap_turns", "value": f"{restore.mean_source_gap_turns:.6f}"},
        {"metric": "p95_source_gap_turns", "value": f"{restore.p95_source_gap_turns:.6f}"},
        {"metric": "max_source_gap_turns", "value": f"{restore.max_source_gap_turns:.6f}"},
        {"metric": "mean_source_gap_ms", "value": f"{restore.mean_source_gap_ms:.6f}"},
        {"metric": "p95_source_gap_ms", "value": f"{restore.p95_source_gap_ms:.6f}"},
        {"metric": "max_source_gap_ms", "value": f"{restore.max_source_gap_ms:.6f}"},
    ]
    return headers, rows


def _overhead_rows_for_csv(overhead: OverheadAnalysisSummary | None) -> tuple[list[str], list[dict[str, object]]]:
    headers = ["metric_name", "source_metric_name", "count", "mean_ms", "p50_ms", "min_ms", "max_ms", "p95_ms", "p99_ms"]
    if overhead is None:
        return headers, []
    rows = [
        {
            "metric_name": item.metric_name,
            "source_metric_name": item.source_metric_name,
            "count": item.count,
            "mean_ms": f"{item.mean_ms:.6f}",
            "p50_ms": f"{item.p50_ms:.6f}",
            "min_ms": f"{item.min_ms:.6f}",
            "max_ms": f"{item.max_ms:.6f}",
            "p95_ms": f"{item.p95_ms:.6f}",
            "p99_ms": f"{item.p99_ms:.6f}",
        }
        for item in overhead.metrics
    ]
    return headers, rows


def _turn_rows_for_csv(turn: TurnAnalysisSummary | None) -> tuple[list[str], list[dict[str, object]]]:
    headers = ["metric_name", "count", "mean_ms", "p50_ms", "min_ms", "max_ms", "p95_ms", "p99_ms"]
    if turn is None:
        return headers, []
    rows = [
        {
            "metric_name": item.metric_name,
            "count": item.count,
            "mean_ms": f"{item.mean_ms:.6f}",
            "p50_ms": f"{item.p50_ms:.6f}",
            "min_ms": f"{item.min_ms:.6f}",
            "max_ms": f"{item.max_ms:.6f}",
            "p95_ms": f"{item.p95_ms:.6f}",
            "p99_ms": f"{item.p99_ms:.6f}",
        }
        for item in turn.metrics
    ]
    return headers, rows
def _resource_rows_for_csv(resource: ResourceAnalysisSummary | None) -> tuple[list[str], list[dict[str, object]]]:
    headers = ["metric_name", "sandbox_id", "sample_count", "mean_value", "max_value"]
    if resource is None:
        return headers, []
    rows = [
        {
            "metric_name": item.metric_name,
            "sandbox_id": item.sandbox_id,
            "sample_count": item.sample_count,
            "mean_value": f"{item.mean_value:.6f}",
            "max_value": f"{item.max_value:.6f}",
        }
        for item in [*resource.host_metrics, *resource.sandbox_metrics]
    ]
    return headers, rows


def generate_report_bundle(
    telemetry_path: Path,
    *,
    output_dir: Path,
    run_id: str | None = None,
    top_k: int = 25,
    exclude_failed_tasks: bool = False,
    log_scale: bool = False,
    export_svg: bool = True,
    window_size_seconds: float | None = None,
) -> TelemetryAnalysis:
    analysis = analyze_telemetry_file(
        telemetry_path,
        run_id=run_id,
        top_k=top_k,
        exclude_failed_tasks=exclude_failed_tasks,
    )
    target_dir = output_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    figures = build_figure_svgs(analysis, log_scale=log_scale, window_size_seconds=window_size_seconds)
    (target_dir / "summary.json").write_text(json.dumps(analysis.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (target_dir / "report.html").write_text(
        render_report_html(analysis, log_scale=log_scale, window_size_seconds=window_size_seconds),
        encoding="utf-8",
    )

    if export_svg:
        for name, svg in figures.items():
            (target_dir / name).write_text(svg, encoding="utf-8")

    _write_csv(
        target_dir / "phase_overview.csv",
        [
            "phase",
            "started_at",
            "finished_at",
            "duration_ms",
            "configured_max_workers",
            "sandbox_count",
            "status",
        ],
        _phase_rows_for_csv(analysis.phase_overview),
    )

    _write_csv(
        target_dir / "operation_summary.csv",
        [
            "metric_name",
            "source_metric_name",
            "category",
            "component",
            "count",
            "total_ms",
            "mean_ms",
            "min_ms",
            "p25_ms",
            "p50_ms",
            "p90_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
            "success_count",
            "failure_count",
            "unique_sandboxes",
            "unique_tasks",
        ],
        _operation_rows_for_csv(analysis.operation_summaries),
    )

    task_headers, task_rows = _task_rows_for_csv(analysis.task_summaries)
    _write_csv(target_dir / "task_summary.csv", task_headers, task_rows)

    _write_csv(
        target_dir / "slow_operations.csv",
        [
            "metric_name",
            "source_metric_name",
            "timestamp",
            "value_ms",
            "component",
            "category",
            "sandbox_id",
            "task_id",
            "request_id",
            "checkpoint_id",
            "job_id",
            "status",
            "operation",
        ],
        [
            {
                "metric_name": item.metric_name,
                "source_metric_name": item.source_metric_name,
                "timestamp": item.timestamp,
                "value_ms": f"{item.value_ms:.6f}",
                "component": item.component,
                "category": item.category,
                "sandbox_id": item.sandbox_id,
                "task_id": item.task_id,
                "request_id": item.request_id,
                "checkpoint_id": item.checkpoint_id,
                "job_id": item.job_id,
                "status": item.status,
                "operation": item.operation,
            }
            for item in analysis.slowest_records
        ],
    )

    _write_csv(
        target_dir / "lifecycle_gaps.csv",
        ["operation_name", "start_count", "finish_count", "missing_finish_count", "finish_status_counts"],
        [
            {
                "operation_name": item.operation_name,
                "start_count": item.start_count,
                "finish_count": item.finish_count,
                "missing_finish_count": item.missing_finish_count,
                "finish_status_counts": json.dumps(item.finish_status_counts, sort_keys=True),
            }
            for item in analysis.lifecycle_gaps
        ],
    )

    headers, rows = _checkpoint_rows_for_csv(analysis.checkpoint_analysis)
    _write_csv(target_dir / "checkpoint_analysis.csv", headers, rows)
    _write_csv(
        target_dir / "checkpoint_per_task.csv",
        ["sandbox_id", "task_id", "total_count", "success_count", "skip_count", "fail_count", "success_rate", "per_minute_rate", "mean_estimated_io_bytes"],
        [
            {
                "sandbox_id": item.sandbox_id,
                "task_id": item.task_id,
                "total_count": item.total_count,
                "success_count": item.success_count,
                "skip_count": item.skip_count,
                "fail_count": item.fail_count,
                "success_rate": f"{item.success_rate:.6f}",
                "per_minute_rate": f"{item.per_minute_rate:.6f}",
                "mean_estimated_io_bytes": f"{item.mean_estimated_io_bytes:.6f}",
            }
            for item in ([] if analysis.checkpoint_analysis is None else analysis.checkpoint_analysis.per_task)
        ],
    )
    _write_csv(
        target_dir / "checkpoint_load.csv",
        ["timestamp", "offset_seconds", "active_jobs", "active_process_jobs", "active_filesystem_jobs", "active_estimated_io_bytes"],
        [
            {
                "timestamp": item.timestamp,
                "offset_seconds": f"{item.offset_seconds:.6f}",
                "active_jobs": item.active_jobs,
                "active_process_jobs": item.active_process_jobs,
                "active_filesystem_jobs": item.active_filesystem_jobs,
                "active_estimated_io_bytes": f"{item.active_estimated_io_bytes:.6f}",
            }
            for item in ([] if analysis.checkpoint_analysis is None else analysis.checkpoint_analysis.load_over_time)
        ],
    )

    headers, rows = _overhead_rows_for_csv(analysis.overhead_analysis)
    _write_csv(target_dir / "overhead_analysis.csv", headers, rows)
    _write_csv(
        target_dir / "overhead_timeseries.csv",
        ["metric_name", "timestamp", "offset_seconds", "value_ms"],
        [
            {
                "metric_name": metric_name,
                "timestamp": point.timestamp,
                "offset_seconds": f"{point.offset_seconds:.6f}",
                "value_ms": f"{point.value:.6f}",
            }
            for metric_name, points in ([] if analysis.overhead_analysis is None else analysis.overhead_analysis.time_series.items())
            for point in points
        ],
    )

    headers, rows = _turn_rows_for_csv(analysis.turn_analysis)
    _write_csv(target_dir / "turn_analysis.csv", headers, rows)
    _write_csv(
        target_dir / "turn_timeseries.csv",
        ["metric_name", "timestamp", "offset_seconds", "value_ms"],
        [
            {
                "metric_name": metric_name,
                "timestamp": point.timestamp,
                "offset_seconds": f"{point.offset_seconds:.6f}",
                "value_ms": f"{point.value:.6f}",
            }
            for metric_name, points in ([] if analysis.turn_analysis is None else analysis.turn_analysis.time_series.items())
            for point in points
        ],
    )
    _write_csv(
        target_dir / "turn_cdf.csv",
        ["metric_name", "value_ms", "cumulative_probability"],
        [
            {
                "metric_name": metric_name,
                "value_ms": f"{point.value:.6f}",
                "cumulative_probability": f"{point.cumulative_probability:.6f}",
            }
            for metric_name, points in ([] if analysis.turn_analysis is None else analysis.turn_analysis.cdf_series.items())
            for point in points
        ],
    )
    headers, rows = _restore_rows_for_csv(analysis.restore_analysis)
    _write_csv(target_dir / "restore_analysis.csv", headers, rows)
    _write_csv(
        target_dir / "restore_per_task.csv",
        ["sandbox_id", "task_id", "total_count", "success_count", "skip_count", "fail_count", "success_rate", "per_minute_rate", "mean_source_gap_turns", "mean_source_gap_ms", "mean_estimated_io_bytes"],
        [
            {
                "sandbox_id": item.sandbox_id,
                "task_id": item.task_id,
                "total_count": item.total_count,
                "success_count": item.success_count,
                "skip_count": item.skip_count,
                "fail_count": item.fail_count,
                "success_rate": f"{item.success_rate:.6f}",
                "per_minute_rate": f"{item.per_minute_rate:.6f}",
                "mean_source_gap_turns": f"{item.mean_source_gap_turns:.6f}",
                "mean_source_gap_ms": f"{item.mean_source_gap_ms:.6f}",
                "mean_estimated_io_bytes": f"{item.mean_estimated_io_bytes:.6f}",
            }
            for item in ([] if analysis.restore_analysis is None else analysis.restore_analysis.per_task)
        ],
    )
    _write_csv(
        target_dir / "restore_load.csv",
        ["timestamp", "offset_seconds", "active_jobs", "active_process_jobs", "active_filesystem_jobs", "active_estimated_io_bytes"],
        [
            {
                "timestamp": item.timestamp,
                "offset_seconds": f"{item.offset_seconds:.6f}",
                "active_jobs": item.active_jobs,
                "active_process_jobs": item.active_process_jobs,
                "active_filesystem_jobs": item.active_filesystem_jobs,
                "active_estimated_io_bytes": f"{item.active_estimated_io_bytes:.6f}",
            }
            for item in ([] if analysis.restore_analysis is None else analysis.restore_analysis.load_over_time)
        ],
    )

    headers, rows = _resource_rows_for_csv(analysis.resource_analysis)
    _write_csv(target_dir / "resource_summary.csv", headers, rows)
    resource_sample_rows: list[dict[str, object]] = []
    if analysis.resource_analysis is not None:
        for metric_name, points in analysis.resource_analysis.host_series.items():
            for point in points:
                resource_sample_rows.append(
                    {
                        "metric_name": metric_name,
                        "sandbox_id": "",
                        "timestamp": point.timestamp,
                        "offset_seconds": f"{point.offset_seconds:.6f}",
                        "value": f"{point.value:.6f}",
                    }
                )
        for metric_name, by_sandbox in analysis.resource_analysis.sandbox_series.items():
            for sandbox_id, points in by_sandbox.items():
                for point in points:
                    resource_sample_rows.append(
                        {
                            "metric_name": metric_name,
                            "sandbox_id": sandbox_id,
                            "timestamp": point.timestamp,
                            "offset_seconds": f"{point.offset_seconds:.6f}",
                            "value": f"{point.value:.6f}",
                        }
                    )
    _write_csv(
        target_dir / "resource_samples.csv",
        ["metric_name", "sandbox_id", "timestamp", "offset_seconds", "value"],
        resource_sample_rows,
    )

    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and visualize Agent-CR telemetry JSONL")
    parser.add_argument("--input", type=Path, required=True, help="Telemetry JSONL path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for report artifacts")
    parser.add_argument("--run-id", default=None, help="Optional run_id filter; defaults to dominant run in file")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K operations/outliers to keep")
    parser.add_argument(
        "--exclude-failed-tasks",
        action="store_true",
        default=False,
        help="Exclude sandboxes whose benchmark.task.success_ratio is 0 from analysis",
    )
    parser.add_argument(
        "--log-scale-charts",
        action="store_true",
        default=False,
        help="Use log10 scale for bar chart widths instead of linear",
    )
    parser.add_argument(
        "--no-export-svg",
        action="store_true",
        default=False,
        help="Skip standalone SVG figure export",
    )
    parser.add_argument(
        "--figure-window-seconds",
        type=float,
        default=0.0,
        help="Average checkpoint/restore line charts within fixed-size time windows; 0 disables aggregation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = generate_report_bundle(
        args.input,
        output_dir=args.output_dir,
        run_id=args.run_id,
        top_k=max(5, int(args.top_k)),
        exclude_failed_tasks=args.exclude_failed_tasks,
        log_scale=args.log_scale_charts,
        export_svg=not args.no_export_svg,
        window_size_seconds=(float(args.figure_window_seconds) if float(args.figure_window_seconds) > 0 else None),
    )
    print(f"run_id: {analysis.run_id}")
    print(f"report: {(args.output_dir.expanduser().resolve() / 'report.html')}")
    print(f"summary_json: {(args.output_dir.expanduser().resolve() / 'summary.json')}")
    print(f"operation_csv: {(args.output_dir.expanduser().resolve() / 'operation_summary.csv')}")
    print(f"task_csv: {(args.output_dir.expanduser().resolve() / 'task_summary.csv')}")
    print(f"slow_ops_csv: {(args.output_dir.expanduser().resolve() / 'slow_operations.csv')}")


if __name__ == "__main__":
    main()

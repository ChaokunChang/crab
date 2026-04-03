from __future__ import annotations

import html
import json
from pathlib import Path

from .models import RunDiagnosis, SandboxDiagnosis, TimelineItem, ToolCallSummary, to_jsonable

DEFAULT_MAX_VISUALIZED_TOOL_ARG_CHARS = 240
DEFAULT_MAX_TOOL_COMPARISON_ROWS = 40
DEFAULT_MAX_VISUALIZED_TOOL_ROWS = 40
DEFAULT_MAX_TIMELINE_EVENTS = 36


def _line_join(lines: list[str]) -> str:
    return "\n".join(lines).rstrip()


def _tool_suffix(tool: ToolCallSummary) -> str:
    parts: list[str] = []
    if tool.duration_ms is not None:
        parts.append(f"{tool.duration_ms:.0f}ms")
    if tool.exit_code is not None:
        parts.append(f"exit={tool.exit_code}")
    elif tool.result_summary:
        parts.append(tool.result_summary)
    elif tool.status:
        parts.append(f"status={tool.status}")
    if tool.is_dummy:
        parts.append("dummy")
    return f" ({', '.join(parts)})" if parts else ""


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _compact_text(value: str, *, limit: int = DEFAULT_MAX_VISUALIZED_TOOL_ARG_CHARS) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)].rstrip()}..."


def _event_badge_class(label: str) -> str:
    normalized = label.lower()
    if "failed" in normalized:
        return "event-failed"
    if "fault injection" in normalized:
        return "event-fault"
    if "restore" in normalized:
        return "event-restore"
    if "checkpoint" in normalized:
        return "event-checkpoint"
    return "event-neutral"


def _format_observed_key_events(
    events: tuple[TimelineItem, ...] | list[TimelineItem],
    *,
    limit: int = 4,
) -> str:
    if not events:
        return ""
    parts: list[str] = []
    for event in list(events)[:limit]:
        timestamp = event.timestamp or ""
        time_part = ""
        if timestamp:
            time_part = timestamp[11:23] if len(timestamp) >= 19 else timestamp
        rendered = f"{time_part} {event.label}".strip()
        parts.append(rendered)
    if len(events) > limit:
        parts.append(f"+{len(events) - limit} more")
    return "; ".join(parts)


def _render_observed_key_events_html(
    events: tuple[TimelineItem, ...] | list[TimelineItem],
    *,
    limit: int = 4,
) -> str:
    if not events:
        return "<td></td>"
    chunks: list[str] = []
    for event in list(events)[:limit]:
        timestamp = event.timestamp or ""
        time_part = timestamp[11:23] if len(timestamp) >= 19 else timestamp
        pieces = []
        if time_part:
            pieces.append(f"<span class='event-time'>{_escape(time_part)}</span>")
        pieces.append(_escape(event.label))
        chunks.append(f"<span class='event-badge {_event_badge_class(event.label)}'>{' '.join(pieces)}</span>")
    if len(events) > limit:
        chunks.append(f"<span class='event-badge event-neutral'>+{len(events) - limit} more</span>")
    return f"<td class='event-cell'>{''.join(chunks)}</td>"


def _collapse_timeline(
    items: list[TimelineItem],
    *,
    limit: int = DEFAULT_MAX_TIMELINE_EVENTS,
    detail_limit: int = 120,
) -> list[tuple[TimelineItem, int]]:
    if not items:
        return []
    groups: list[tuple[TimelineItem, int]] = []
    current = items[0]
    count = 1
    for item in items[1:]:
        if item.label == current.label:
            count += 1
            current = TimelineItem(
                source=current.source,
                timestamp=item.timestamp or current.timestamp,
                label=current.label,
                detail=(
                    f"{_compact_text(groups[-1][0].detail, limit=detail_limit) if groups else _compact_text(current.detail, limit=detail_limit)}"
                    f" -> {_compact_text(item.detail, limit=detail_limit)}"
                ),
                sandbox_id=current.sandbox_id,
                task_id=current.task_id,
                checkpoint_id=item.checkpoint_id or current.checkpoint_id,
                request_id=item.request_id or current.request_id,
                evidence_ref=item.evidence_ref or current.evidence_ref,
            )
            continue
        groups.append((current, count))
        current = item
        count = 1
    groups.append((current, count))
    if len(groups) <= limit:
        return groups
    head = groups[: max(1, limit // 2)]
    tail = groups[-max(1, limit - len(head)) :]
    return head + tail


def _tool_result_badge(tool: ToolCallSummary) -> str:
    if tool.exit_code not in (None, 0):
        return f"nonzero exit {tool.exit_code}"
    if tool.has_error_indicators:
        return tool.result_summary or "error indicators"
    return tool.result_summary or tool.status or "ok"


def _duration_drift_level(trace_duration_ms: object, observed_duration_ms: object) -> str | None:
    if not isinstance(trace_duration_ms, (int, float)) or not isinstance(observed_duration_ms, (int, float)):
        return None
    if trace_duration_ms < 1000.0 or observed_duration_ms < 1000.0:
        return None
    faster = min(trace_duration_ms, observed_duration_ms)
    slower = max(trace_duration_ms, observed_duration_ms)
    if faster <= 0:
        return None
    ratio = slower / faster
    if ratio >= 100.0:
        return "drift-100"
    if ratio >= 10.0:
        return "drift-10"
    if ratio >= 5.0:
        return "drift-5"
    return None


def _duration_drift_label(trace_duration_ms: object, observed_duration_ms: object) -> str | None:
    if not isinstance(trace_duration_ms, (int, float)) or not isinstance(observed_duration_ms, (int, float)):
        return None
    if trace_duration_ms < 1000.0 or observed_duration_ms < 1000.0:
        return None
    faster = min(trace_duration_ms, observed_duration_ms)
    slower = max(trace_duration_ms, observed_duration_ms)
    if faster <= 0:
        return None
    ratio = slower / faster
    if ratio < 5.0:
        return None
    return f"{ratio:.1f}x"


def _compare_tool_rows(
    diagnosis: SandboxDiagnosis,
    *,
    limit: int = DEFAULT_MAX_TOOL_COMPARISON_ROWS,
    arg_limit: int = DEFAULT_MAX_VISUALIZED_TOOL_ARG_CHARS,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    trace_tools = diagnosis.trace_tool_calls
    observed_tools = diagnosis.tool_calls
    total = max(len(trace_tools), len(observed_tools))
    for index in range(total):
        trace_tool = trace_tools[index] if index < len(trace_tools) else None
        observed_tool = observed_tools[index] if index < len(observed_tools) else None
        mismatch = False
        if trace_tool is None or observed_tool is None:
            mismatch = True
        elif (trace_tool.raw_arguments or {}) != (observed_tool.raw_arguments or {}) or trace_tool.tool_name != observed_tool.tool_name:
            mismatch = True
        expensive = bool(observed_tool and observed_tool.duration_ms is not None and observed_tool.duration_ms >= 30000.0)
        problematic = bool(observed_tool and (observed_tool.exit_code not in (None, 0) or observed_tool.has_error_indicators))
        if mismatch or expensive or problematic or index < 3 or index >= max(0, total - 3):
            rows.append(
                {
                    "index": index,
                    "trace_tool": None if trace_tool is None else trace_tool.tool_name,
                    "trace_args": None if trace_tool is None else _compact_text(trace_tool.arguments_preview, limit=arg_limit),
                    "trace_duration_ms": None if trace_tool is None else trace_tool.duration_ms,
                    "trace_result": None if trace_tool is None else _tool_result_badge(trace_tool),
                    "observed_tool": None if observed_tool is None else observed_tool.tool_name,
                    "observed_args": None if observed_tool is None else _compact_text(observed_tool.arguments_preview, limit=arg_limit),
                    "observed_duration_ms": None if observed_tool is None else observed_tool.duration_ms,
                    "result": None if observed_tool is None else _tool_result_badge(observed_tool),
                    "observed_key_events": () if observed_tool is None else observed_tool.observed_key_events,
                    "dummy": bool(observed_tool and observed_tool.is_dummy),
                    "mismatch": mismatch,
                    "duration_drift_level": _duration_drift_level(
                        None if trace_tool is None else trace_tool.duration_ms,
                        None if observed_tool is None else observed_tool.duration_ms,
                    ),
                    "duration_drift_label": _duration_drift_label(
                        None if trace_tool is None else trace_tool.duration_ms,
                        None if observed_tool is None else observed_tool.duration_ms,
                    ),
                }
            )
    if len(rows) <= limit:
        return rows
    return rows[:limit]


def _format_summary_map(summary: dict[str, object], *, keys: list[str]) -> str | None:
    parts: list[str] = []
    for key in keys:
        value = summary.get(key)
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}={value}")
    if not parts:
        return None
    return ", ".join(parts)


def _observed_events_for_index(diagnosis: SandboxDiagnosis, index: int) -> tuple[TimelineItem, ...]:
    if index < 0 or index >= len(diagnosis.tool_calls):
        return ()
    return diagnosis.tool_calls[index].observed_key_events


def _format_sandbox_section(
    diagnosis: SandboxDiagnosis,
    *,
    max_visualized_tool_arg_chars: int = DEFAULT_MAX_VISUALIZED_TOOL_ARG_CHARS,
    max_tool_comparison_rows: int = DEFAULT_MAX_TOOL_COMPARISON_ROWS,
    max_visualized_tool_rows: int = DEFAULT_MAX_VISUALIZED_TOOL_ROWS,
    max_timeline_events: int = DEFAULT_MAX_TIMELINE_EVENTS,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"## Sandbox `{diagnosis.sandbox_id}`")
    lines.append(f"- task: `{diagnosis.task_id}`")
    lines.append(f"- status: `{diagnosis.status}`")
    if diagnosis.findings:
        lines.append("- findings:")
        for finding in diagnosis.findings:
            lines.append(f"  - [{finding.severity}] {finding.title}: {finding.summary}")
    if diagnosis.tool_alignment_summary:
        lines.append(f"- tool alignment: {diagnosis.tool_alignment_summary}")
    if diagnosis.timeline:
        lines.append("- timeline:")
        for item, count in _collapse_timeline(diagnosis.timeline, limit=max_timeline_events, detail_limit=max_visualized_tool_arg_chars):
            timestamp = f"{item.timestamp} " if item.timestamp else ""
            prefix = f"{count}x " if count > 1 else ""
            lines.append(f"  - {timestamp}{prefix}{item.label}: {_compact_text(item.detail, limit=max(160, max_visualized_tool_arg_chars))}")
    compare_rows = _compare_tool_rows(
        diagnosis,
        limit=max_tool_comparison_rows,
        arg_limit=max_visualized_tool_arg_chars,
    )
    if compare_rows:
        lines.append("- tool comparison:")
        for row in compare_rows:
            trace_duration = ""
            if isinstance(row["trace_duration_ms"], (int, float)):
                trace_duration = f"{row['trace_duration_ms']:.0f}ms"
            observed_duration = ""
            if isinstance(row["observed_duration_ms"], (int, float)):
                observed_duration = f"{row['observed_duration_ms']:.0f}ms"
            drift_suffix = ""
            if isinstance(row["duration_drift_label"], str) and row["duration_drift_label"]:
                drift_suffix = f" drift={row['duration_drift_label']}"
            events_suffix = ""
            if row["observed_key_events"]:
                events_suffix = f" events=[{_format_observed_key_events(row['observed_key_events'])}]"
            lines.append(
                "  - "
                f"#{row['index']}: trace={row['trace_tool']} "
                f"({_compact_text(str(row['trace_args'] or ''), limit=max_visualized_tool_arg_chars)}, "
                f"{trace_duration}, {row['trace_result'] or ''})"
                f" -> observed={row['observed_tool']} "
                f"({_compact_text(str(row['observed_args'] or ''), limit=max_visualized_tool_arg_chars)}, "
                f"{observed_duration}, "
                f"{row['result'] or ''})"
                f"{events_suffix}"
                f"{' mismatch' if row['mismatch'] else ''}"
                f"{' dummy' if row['dummy'] else ''}"
                f"{drift_suffix}"
            )
        if len(diagnosis.trace_tool_calls) > max_tool_comparison_rows or len(diagnosis.tool_calls) > max_tool_comparison_rows:
            lines.append(
                f"  - showing {len(compare_rows)} comparison rows out of {max(len(diagnosis.trace_tool_calls), len(diagnosis.tool_calls))} tool positions"
            )
    trace_summary = _format_summary_map(
        diagnosis.replay_trace_summary,
        keys=[
            "trace_path",
            "exchange_count",
            "tool_call_count",
            "avg_trace_tool_turn_ms",
            "avg_trace_delay_ms",
        ],
    )
    if trace_summary:
        lines.append(f"- replay trace: {trace_summary}")
    session_summary = _format_summary_map(
        diagnosis.session_summary,
        keys=[
            "session_file",
            "tool_call_count",
            "tool_result_count",
            "dummy_tool_call_count",
            "replay_marker_count",
            "max_tool_duration_ms",
        ],
    )
    if session_summary:
        lines.append(f"- session: {session_summary}")
    if diagnosis.trace_tool_calls:
        lines.append("- trace tool calls:")
        for tool in diagnosis.trace_tool_calls[:max_visualized_tool_rows]:
            lines.append(
                f"  - #{tool.call_index}: {tool.tool_name}: "
                f"{_compact_text(tool.arguments_preview, limit=max_visualized_tool_arg_chars)}"
                f"{_tool_suffix(tool)}"
            )
    if diagnosis.tool_calls:
        lines.append("- observed tool calls:")
        for tool in diagnosis.tool_calls[:max_visualized_tool_rows]:
            lines.append(
                f"  - #{tool.call_index}: {tool.tool_name}: "
                f"{_compact_text(tool.arguments_preview, limit=max_visualized_tool_arg_chars)}"
                f"{_tool_suffix(tool)}"
                f"{f' events=[{_format_observed_key_events(tool.observed_key_events)}]' if tool.observed_key_events else ''}"
            )
    if diagnosis.replay_trace_summary and not trace_summary:
        lines.append(f"- replay trace: {diagnosis.replay_trace_summary}")
    if diagnosis.session_summary and not session_summary:
        lines.append(f"- session: {diagnosis.session_summary}")
    if diagnosis.log_excerpt:
        lines.append("- key log lines:")
        for line in diagnosis.log_excerpt[:16]:
            lines.append(f"  - {line}")
    return lines


def render_run_diagnosis_markdown(
    report: RunDiagnosis,
    *,
    max_visualized_tool_arg_chars: int = DEFAULT_MAX_VISUALIZED_TOOL_ARG_CHARS,
    max_tool_comparison_rows: int = DEFAULT_MAX_TOOL_COMPARISON_ROWS,
    max_visualized_tool_rows: int = DEFAULT_MAX_VISUALIZED_TOOL_ROWS,
    max_timeline_events: int = DEFAULT_MAX_TIMELINE_EVENTS,
) -> str:
    lines: list[str] = []
    ctx = report.context
    lines.append("# Benchmark Diagnosis")
    lines.append(f"- config: `{ctx.config_path}`")
    if ctx.log_path is not None:
        lines.append(f"- log: `{ctx.log_path}`")
    if ctx.csv_path is not None:
        lines.append(f"- csv: `{ctx.csv_path}`")
    if ctx.telemetry_path is not None:
        lines.append(f"- telemetry: `{ctx.telemetry_path}`")
    if ctx.actual_benchmark_root is not None:
        lines.append(f"- benchmark root: `{ctx.actual_benchmark_root}`")
    lines.append(f"- dataset tasks: {len(report.dataset_tasks)}")
    lines.append(f"- csv rows: {len(report.csv_rows)}")
    lines.append(f"- failed sandboxes: {len(report.failed_sandboxes)}")
    lines.append(f"- missing tasks: {len(report.missing_tasks)}")
    if report.run_findings:
        lines.append("## Run Findings")
        for finding in report.run_findings:
            lines.append(f"- [{finding.severity}] {finding.title}: {finding.summary}")
    if report.missing_tasks:
        lines.append("## Missing Tasks")
        for missing in report.missing_tasks[:20]:
            lines.append(
                f"- dataset_index={missing.dataset_index} task_id=`{missing.task_id}` reason={missing.reason} expected={missing.occurrences_expected} observed={missing.occurrences_observed}"
            )
    if report.failed_sandboxes:
        lines.append("## Failed Sandboxes")
        for sandbox_id in report.failed_sandboxes:
            diagnosis = report.sandboxes.get(sandbox_id)
            if diagnosis is None:
                continue
            lines.extend(
                _format_sandbox_section(
                    diagnosis,
                    max_visualized_tool_arg_chars=max_visualized_tool_arg_chars,
                    max_tool_comparison_rows=max_tool_comparison_rows,
                    max_visualized_tool_rows=max_visualized_tool_rows,
                    max_timeline_events=max_timeline_events,
                )
            )
    return _line_join(lines)


def render_run_diagnosis_text(
    report: RunDiagnosis,
    *,
    max_visualized_tool_arg_chars: int = DEFAULT_MAX_VISUALIZED_TOOL_ARG_CHARS,
    max_tool_comparison_rows: int = DEFAULT_MAX_TOOL_COMPARISON_ROWS,
    max_visualized_tool_rows: int = DEFAULT_MAX_VISUALIZED_TOOL_ROWS,
    max_timeline_events: int = DEFAULT_MAX_TIMELINE_EVENTS,
) -> str:
    markdown = render_run_diagnosis_markdown(
        report,
        max_visualized_tool_arg_chars=max_visualized_tool_arg_chars,
        max_tool_comparison_rows=max_tool_comparison_rows,
        max_visualized_tool_rows=max_visualized_tool_rows,
        max_timeline_events=max_timeline_events,
    )
    return markdown.replace("# ", "").replace("## ", "\n")


def render_run_diagnosis_html(
    report: RunDiagnosis,
    *,
    max_visualized_tool_arg_chars: int = DEFAULT_MAX_VISUALIZED_TOOL_ARG_CHARS,
    max_tool_comparison_rows: int = DEFAULT_MAX_TOOL_COMPARISON_ROWS,
    max_visualized_tool_rows: int = DEFAULT_MAX_VISUALIZED_TOOL_ROWS,
    max_timeline_events: int = DEFAULT_MAX_TIMELINE_EVENTS,
) -> str:
    ctx = report.context
    sections: list[str] = []
    sections.append(
        "<section class='summary'>"
        "<h1>Benchmark Diagnosis</h1>"
        f"<p><strong>Config:</strong> {_escape(ctx.config_path)}<br>"
        f"<strong>Log:</strong> {_escape(ctx.log_path)}<br>"
        f"<strong>CSV:</strong> {_escape(ctx.csv_path)}<br>"
        f"<strong>Telemetry:</strong> {_escape(ctx.telemetry_path)}<br>"
        f"<strong>Benchmark Root:</strong> {_escape(ctx.actual_benchmark_root)}</p>"
        f"<p><strong>Dataset Tasks:</strong> {len(report.dataset_tasks)} | "
        f"<strong>CSV Rows:</strong> {len(report.csv_rows)} | "
        f"<strong>Failed Sandboxes:</strong> {len(report.failed_sandboxes)} | "
        f"<strong>Missing Tasks:</strong> {len(report.missing_tasks)}</p>"
        "</section>"
    )
    if report.run_findings:
        items = "".join(
            f"<li><strong>{_escape(finding.title)}</strong>: {_escape(finding.summary)}</li>"
            for finding in report.run_findings
        )
        sections.append(f"<section><h2>Run Findings</h2><ul>{items}</ul></section>")
    for sandbox_id in report.failed_sandboxes:
        diagnosis = report.sandboxes.get(sandbox_id)
        if diagnosis is None:
            continue
        findings = "".join(
            f"<li class='finding finding-{_escape(finding.severity)}'>"
            f"<span class='severity severity-{_escape(finding.severity)}'>{_escape(finding.severity)}</span> "
            f"<strong>{_escape(finding.title)}</strong>: {_escape(finding.summary)}</li>"
            for finding in diagnosis.findings
        )
        collapsed_timeline = _collapse_timeline(
            diagnosis.timeline,
            limit=max_timeline_events,
            detail_limit=max_visualized_tool_arg_chars,
        )
        timeline_rows = "".join(
            "<tr>"
            f"<td>{_escape(item.timestamp)}</td>"
            f"<td>{_escape(f'{count}x' if count > 1 else '')}</td>"
            f"<td>{_escape(item.label)}</td>"
            f"<td>{_escape(_compact_text(item.detail, limit=max(180, max_visualized_tool_arg_chars)))}</td>"
            "</tr>"
            for item, count in collapsed_timeline
        )
        compare_rows_data = _compare_tool_rows(
            diagnosis,
            limit=max_tool_comparison_rows,
            arg_limit=max_visualized_tool_arg_chars,
        )
        def _duration_cell(value: object, level: object, label: object) -> str:
            classes = "duration-cell"
            if isinstance(level, str) and level:
                classes = f"{classes} {level}"
            suffix = ""
            if isinstance(label, str) and label:
                suffix = f" <span class='drift-badge'>{_escape(label)}</span>"
            rendered = _escape("%.0f" % value if isinstance(value, (int, float)) else "")
            return f"<td class='{classes}'>{rendered}{suffix}</td>"

        def _result_cell(value: object) -> str:
            text = "" if value is None else str(value)
            classes = "result-cell result-good" if text in {"", "ok"} else "result-cell result-bad"
            return f"<td class='{classes}'>{_escape(text)}</td>"

        def _flag_cell(*, active: bool, active_class: str) -> str:
            classes = "flag-cell"
            if active:
                classes = f"{classes} {active_class}"
            return f"<td class='{classes}'>{_escape('yes' if active else '')}</td>"

        def _compare_row_classes(row: dict[str, object]) -> str:
            classes = ["compare-row"]
            level = row.get("duration_drift_level")
            if isinstance(level, str) and level:
                classes.append(level)
            if row.get("mismatch"):
                classes.append("row-mismatch")
            if row.get("dummy"):
                classes.append("row-dummy")
            return " ".join(classes)

        compare_rows = "".join(
            f"<tr class='{_escape(_compare_row_classes(row))}'>"
            f"<td>{_escape(row['index'])}</td>"
            f"<td>{_escape(row['trace_tool'])}</td>"
            f"<td>{_escape(row['trace_args'])}</td>"
            f"{_duration_cell(row['trace_duration_ms'], row['duration_drift_level'], row['duration_drift_label'])}"
            f"{_result_cell(row['trace_result'])}"
            f"<td>{_escape(row['observed_tool'])}</td>"
            f"<td>{_escape(row['observed_args'])}</td>"
            f"{_duration_cell(row['observed_duration_ms'], row['duration_drift_level'], row['duration_drift_label'])}"
            f"{_result_cell(row['result'])}"
            f"{_render_observed_key_events_html(row['observed_key_events'])}"
            f"{_flag_cell(active=bool(row['dummy']), active_class='flag-dummy')}"
            f"{_flag_cell(active=bool(row['mismatch']), active_class='flag-mismatch')}"
            "</tr>"
            for row in compare_rows_data
        )
        alignment = "".join(
            f"<li><strong>{_escape(key)}</strong>: {_escape(value)}</li>"
            for key, value in diagnosis.tool_alignment_summary.items()
        )
        session_summary = "".join(
            f"<li><strong>{_escape(key)}</strong>: {_escape(value)}</li>"
            for key, value in diagnosis.session_summary.items()
            if value not in (None, "", [], {})
        )
        replay_summary = "".join(
            f"<li><strong>{_escape(key)}</strong>: {_escape(value)}</li>"
            for key, value in diagnosis.replay_trace_summary.items()
            if value not in (None, "", [], {})
        )
        trace_tool_rows_parts: list[str] = []
        for tool in diagnosis.trace_tool_calls[:max_visualized_tool_rows]:
            trace_tool_rows_parts.append(
                f"<tr class='detail-row{' row-error' if tool.exit_code not in (None, 0) or tool.has_error_indicators else ' row-ok'}'>"
                f"<td>{_escape(tool.call_index)}</td>"
                f"<td>{_escape(tool.timestamp)}</td>"
                f"<td>{_escape(tool.tool_name)}</td>"
                f"<td>{_escape(_compact_text(tool.arguments_preview, limit=max_visualized_tool_arg_chars))}</td>"
                f"<td class='duration-cell'>{_escape('%.0f' % tool.duration_ms if isinstance(tool.duration_ms, (int, float)) else '')}</td>"
                f"<td class='result-cell{' result-bad' if tool.exit_code not in (None, 0) or tool.has_error_indicators else ' result-good'}'>{_escape(_tool_result_badge(tool))}</td>"
                "</tr>"
            )
        trace_tool_rows = "".join(trace_tool_rows_parts)
        observed_tool_rows_parts: list[str] = []
        for tool in diagnosis.tool_calls[:max_visualized_tool_rows]:
            observed_tool_rows_parts.append(
                f"<tr class='detail-row{' row-error' if tool.exit_code not in (None, 0) or tool.has_error_indicators else ' row-ok'}'>"
                f"<td>{_escape(tool.call_index)}</td>"
                f"<td>{_escape(tool.timestamp)}</td>"
                f"<td>{_escape(tool.tool_name)}</td>"
                f"<td>{_escape(_compact_text(tool.arguments_preview, limit=max_visualized_tool_arg_chars))}</td>"
                f"<td class='duration-cell'>{_escape('%.0f' % tool.duration_ms if isinstance(tool.duration_ms, (int, float)) else '')}</td>"
                f"<td class='result-cell{' result-bad' if tool.exit_code not in (None, 0) or tool.has_error_indicators else ' result-good'}'>{_escape(_tool_result_badge(tool))}</td>"
                f"{_render_observed_key_events_html(tool.observed_key_events)}"
                "</tr>"
            )
        observed_tool_rows = "".join(observed_tool_rows_parts)
        sections.append(
            "<section class='sandbox'>"
            f"<h2>{_escape(diagnosis.sandbox_id)} <span>{_escape(diagnosis.task_id)}</span></h2>"
            f"<p><strong>Status:</strong> {_escape(diagnosis.status)}</p>"
            f"<div class='grid'><div><h3>Findings</h3><ul>{findings}</ul></div>"
            f"<div><h3>Tool Alignment</h3><ul>{alignment}</ul></div></div>"
            f"<div class='grid'><div><h3>Replay Trace Summary</h3><ul>{replay_summary}</ul></div>"
            f"<div><h3>Session Summary</h3><ul>{session_summary}</ul></div></div>"
            "<h3>Timeline</h3>"
            "<table><thead><tr><th>Time</th><th>Count</th><th>Event</th><th>Detail</th></tr></thead>"
            f"<tbody>{timeline_rows}</tbody></table>"
            f"<p class='table-note'>Showing {len(collapsed_timeline)} timeline rows.</p>"
            "<h3>Tool Comparison</h3>"
            "<table><thead><tr><th>#</th><th>Trace Tool</th><th>Trace Args</th><th>Trace ms</th><th>Trace Result</th><th>Observed Tool</th><th>Observed Args</th><th>Observed ms</th><th>Observed Result</th><th>Observed Key Event</th><th>Dummy</th><th>Mismatch</th></tr></thead>"
            f"<tbody>{compare_rows}</tbody></table>"
            f"<p class='table-note'>Showing {len(compare_rows_data)} high-signal comparison rows out of {max(len(diagnosis.trace_tool_calls), len(diagnosis.tool_calls))} tool positions.</p>"
            "<details><summary>Trace Tool Calls</summary>"
            "<table><thead><tr><th>#</th><th>Time</th><th>Tool</th><th>Args</th><th>Trace ms</th><th>Trace Result</th></tr></thead><tbody>"
            + trace_tool_rows
            + "</tbody></table></details>"
            "<details><summary>Observed Tool Calls</summary>"
            "<table><thead><tr><th>#</th><th>Time</th><th>Tool</th><th>Args</th><th>Observed ms</th><th>Result</th><th>Observed Key Event</th></tr></thead><tbody>"
            + observed_tool_rows
            + "</tbody></table></details>"
            "</section>"
        )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Benchmark Diagnosis</title>"
        "<style>"
        "body{font-family:ui-sans-serif,system-ui,sans-serif;margin:24px;background:linear-gradient(180deg,#f4f8ff 0%,#f8fafc 45%,#eef4f7 100%);color:#18212b;}"
        "section{background:#fff;border:1px solid #d8dee6;border-radius:16px;padding:20px;margin:0 0 20px 0;box-shadow:0 10px 24px rgba(15,23,42,.06);}"
        ".summary{background:linear-gradient(135deg,#0f766e 0%,#155e75 55%,#1d4ed8 100%);color:#f8fafc;border:none;}"
        ".summary strong{color:#e0f2fe;}"
        "h1,h2,h3{margin:0 0 12px 0;} h2 span{font-weight:400;color:#556070;font-size:.9em;}"
        ".sandbox h2{padding-bottom:8px;border-bottom:2px solid #dbeafe;}"
        ".grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:start;}"
        "table{width:100%;border-collapse:separate;border-spacing:0;font-size:13px;overflow:hidden;border-radius:12px;}"
        "th,td{border-right:1px solid #d8dee6;border-bottom:1px solid #d8dee6;padding:7px 9px;vertical-align:top;text-align:left;}"
        "th:first-child,td:first-child{border-left:1px solid #d8dee6;}"
        "thead th{background:linear-gradient(180deg,#dbeafe 0%,#e0ecff 100%);color:#16324f;position:sticky;top:0;}"
        "tbody tr:nth-child(odd) td{background:#fcfdff;}"
        "tbody tr:nth-child(even) td{background:#f8fbff;}"
        "ul{margin:0;padding-left:20px;} summary{cursor:pointer;font-weight:600;margin:8px 0;color:#1d4ed8;}"
        ".table-note{color:#556070;font-size:12px;margin:8px 0 0 0;}"
        ".finding{margin-bottom:8px;}"
        ".severity{display:inline-block;border-radius:999px;padding:2px 8px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;}"
        ".severity-high{background:#fee2e2;color:#991b1b;}"
        ".severity-medium{background:#fef3c7;color:#92400e;}"
        ".severity-low{background:#dcfce7;color:#166534;}"
        ".duration-cell{font-variant-numeric:tabular-nums;}"
        ".result-cell{font-weight:600;}"
        ".result-good{color:#166534;background:#ecfdf3 !important;}"
        ".result-bad{color:#9f1239;background:#fff1f2 !important;}"
        ".event-cell{min-width:220px;background:#f8fafc !important;}"
        ".event-badge{display:inline-flex;align-items:center;gap:6px;margin:2px 6px 2px 0;padding:3px 8px;border-radius:999px;font-size:11px;font-weight:700;white-space:nowrap;}"
        ".event-time{font-variant-numeric:tabular-nums;font-weight:600;opacity:.9;}"
        ".event-checkpoint{background:#dbeafe;color:#1d4ed8;}"
        ".event-restore{background:#dcfce7;color:#166534;}"
        ".event-fault{background:#ffedd5;color:#c2410c;}"
        ".event-failed{background:#fee2e2;color:#b91c1c;}"
        ".event-neutral{background:#e5e7eb;color:#374151;}"
        ".flag-cell{font-weight:700;text-transform:uppercase;font-size:11px;letter-spacing:.03em;}"
        ".flag-dummy{background:#ede9fe !important;color:#6d28d9;}"
        ".flag-mismatch{background:#fee2e2 !important;color:#b91c1c;}"
        ".drift-badge{display:inline-block;margin-left:6px;padding:1px 6px;border-radius:999px;background:#0f172a;color:#fff;font-size:10px;font-weight:700;}"
        ".drift-5{background:#fff7ed !important;color:#9a3412;}"
        ".drift-10{background:#fef3c7 !important;color:#92400e;}"
        ".drift-100{background:#fee2e2 !important;color:#991b1b;}"
        ".compare-row.drift-5 td{box-shadow:inset 0 0 0 9999px rgba(251,146,60,.08);}"
        ".compare-row.drift-10 td{box-shadow:inset 0 0 0 9999px rgba(245,158,11,.14);}"
        ".compare-row.drift-100 td{box-shadow:inset 0 0 0 9999px rgba(239,68,68,.12);}"
        ".compare-row.row-mismatch td{border-top:2px solid #ef4444;}"
        ".compare-row.row-dummy td{border-top:2px solid #8b5cf6;}"
        ".detail-row.row-error td{box-shadow:inset 0 0 0 9999px rgba(255,241,242,.65);}"
        ".detail-row.row-ok td{box-shadow:inset 0 0 0 9999px rgba(240,253,244,.45);}"
        "@media (max-width: 1100px){.grid{grid-template-columns:1fr;}}"
        "</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )


def write_run_diagnosis_outputs(
    report: RunDiagnosis,
    output_dir: Path,
    *,
    max_visualized_tool_arg_chars: int = DEFAULT_MAX_VISUALIZED_TOOL_ARG_CHARS,
    max_tool_comparison_rows: int = DEFAULT_MAX_TOOL_COMPARISON_ROWS,
    max_visualized_tool_rows: int = DEFAULT_MAX_VISUALIZED_TOOL_ROWS,
    max_timeline_events: int = DEFAULT_MAX_TIMELINE_EVENTS,
) -> dict[str, Path]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "text": output_dir / "diagnosis.txt",
        "markdown": output_dir / "diagnosis.md",
        "html": output_dir / "diagnosis.html",
        "json": output_dir / "diagnosis.json",
    }
    paths["text"].write_text(
        render_run_diagnosis_text(
            report,
            max_visualized_tool_arg_chars=max_visualized_tool_arg_chars,
            max_tool_comparison_rows=max_tool_comparison_rows,
            max_visualized_tool_rows=max_visualized_tool_rows,
            max_timeline_events=max_timeline_events,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(
        render_run_diagnosis_markdown(
            report,
            max_visualized_tool_arg_chars=max_visualized_tool_arg_chars,
            max_tool_comparison_rows=max_tool_comparison_rows,
            max_visualized_tool_rows=max_visualized_tool_rows,
            max_timeline_events=max_timeline_events,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["html"].write_text(
        render_run_diagnosis_html(
            report,
            max_visualized_tool_arg_chars=max_visualized_tool_arg_chars,
            max_tool_comparison_rows=max_tool_comparison_rows,
            max_visualized_tool_rows=max_visualized_tool_rows,
            max_timeline_events=max_timeline_events,
        )
        + "\n",
        encoding="utf-8",
    )
    paths["json"].write_text(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for sandbox_id in report.failed_sandboxes:
        diagnosis = report.sandboxes.get(sandbox_id)
        if diagnosis is None or diagnosis.status != "failed" or not diagnosis.log_lines:
            continue
        (output_dir / f"{sandbox_id}.log").write_text("\n".join(diagnosis.log_lines) + "\n", encoding="utf-8")
    return paths

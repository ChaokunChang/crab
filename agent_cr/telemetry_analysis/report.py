from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from html import escape
import json
from pathlib import Path
from typing import Iterable

from .analyzer import MetricSummary, TaskSummary, TelemetryAnalysis, analyze_telemetry_file


def _format_ms(value: float) -> str:
    if value >= 1000.0:
        return f"{value / 1000.0:.2f} s"
    return f"{value:.2f} ms"


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def _format_metric_label(metric_name: str) -> str:
    return metric_name.replace(".duration_ms", "").replace("_", " ")


def _svg_bar_chart(title: str, rows: list[tuple[str, float]], *, width: int = 900, row_height: int = 24) -> str:
    if not rows:
        return f"<section><h3>{escape(title)}</h3><p>No data.</p></section>"
    max_label = max(len(label) for label, _ in rows)
    left_pad = min(340, max(140, max_label * 7))
    bar_width = width - left_pad - 100
    max_value = max(value for _, value in rows) or 1.0
    height = 40 + row_height * len(rows)
    pieces = [
        f"<section><h3>{escape(title)}</h3>",
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' role='img' aria-label='{escape(title)}'>",
        "<style>.axis-label{font:12px sans-serif;fill:#334155}.bar{fill:#2563eb}.bar-text{font:11px sans-serif;fill:#0f172a}</style>",
    ]
    y = 24
    for label, value in rows:
        scaled = 0.0 if max_value <= 0 else (value / max_value) * bar_width
        pieces.append(f"<text class='axis-label' x='8' y='{y}'>{escape(label)}</text>")
        pieces.append(f"<rect class='bar' x='{left_pad}' y='{y - 11}' width='{scaled:.2f}' height='14' rx='3'></rect>")
        pieces.append(
            f"<text class='bar-text' x='{left_pad + scaled + 8:.2f}' y='{y}'>{escape(_format_number(value))}</text>"
        )
        y += row_height
    pieces.append("</svg></section>")
    return "".join(pieces)


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


def _task_latency_rows(tasks: list[TaskSummary]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    for task in tasks:
        rows.append((task.task_id, task.metrics.get("benchmark.task.duration_ms", 0.0)))
    return rows


def render_report_html(analysis: TelemetryAnalysis) -> str:
    top_total = _svg_bar_chart(
        "Top Operations By Cumulative Time (ms)",
        _operation_rows(analysis.top_total_time_operations[:15]),
    )
    top_count = _svg_bar_chart(
        "Top Operations By Invocation Count",
        [(_format_metric_label(item.metric_name), float(item.count)) for item in analysis.top_invocation_operations[:15]],
    )
    top_tail = _svg_bar_chart(
        "Top Operations By P95 Latency (ms)",
        _operation_tail_rows(analysis.top_tail_latency_operations[:15]),
    )
    task_chart = _svg_bar_chart(
        "Task End-To-End Latency (ms)",
        _task_latency_rows(analysis.task_summaries[:20]),
        width=980,
    )

    llm_breakdown_rows = [(name, value) for name, value in analysis.llm_breakdown.items()]
    checkpoint_breakdown_rows = [(name, value) for name, value in analysis.checkpoint_breakdown.items()]

    llm_breakdown_chart = _svg_bar_chart(
        "Average LLM Path Breakdown (ms)",
        [(_format_metric_label(name), value) for name, value in llm_breakdown_rows],
    )
    checkpoint_breakdown_chart = _svg_bar_chart(
        "Average Checkpoint/Restore Breakdown (ms)",
        [(_format_metric_label(name), value) for name, value in checkpoint_breakdown_rows],
        width=980,
    )

    summary_table = _html_table(
        ["Field", "Value"],
        [
            ("Input", analysis.input_path),
            ("Run ID", analysis.run_id),
            ("Started", analysis.started_at),
            ("Finished", analysis.finished_at),
            ("Records", analysis.total_records),
            ("Events", analysis.total_events),
            ("Metrics", analysis.total_metrics),
            ("Sandboxes", analysis.distinct_sandboxes),
            ("Tasks", analysis.distinct_tasks),
            ("Requests", analysis.distinct_requests),
            ("Checkpoint IDs", analysis.distinct_checkpoints),
            ("Job IDs", analysis.distinct_jobs),
        ],
    )

    operation_table = _html_table(
        [
            "Metric",
            "Count",
            "Total (ms)",
            "Mean (ms)",
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
                f"{item.p95_ms:.2f}",
                f"{item.p99_ms:.2f}",
                f"{item.max_ms:.2f}",
                item.failure_count,
                item.component,
            )
            for item in analysis.top_total_time_operations[:20]
        ],
    )

    task_metric_names = sorted(
        {
            metric_name
            for task in analysis.task_summaries
            for metric_name in task.metrics
        }
    )
    task_headers = ["Task", "Agent", "LLM Service"] + task_metric_names
    task_rows = []
    for task in analysis.task_summaries:
        task_rows.append(
            [task.task_id, task.agent_type, task.llm_service_type]
            + [f"{task.metrics.get(metric_name, 0.0):.2f}" for metric_name in task_metric_names]
        )
    task_table = _html_table(task_headers, task_rows)

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
        ]
        if analysis.lifecycle_gaps
        else [("None", 0, 0, 0, "{}")],
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
      max-width: 1400px;
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
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
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
      <p>This report is generated from the JSONL telemetry stream using a streaming analyzer. It is intended for performance analysis, bottleneck diagnosis, long-tail latency inspection, and research-paper figures/tables.</p>
      {summary_table}
    </section>
    <div class="grid">
      {top_total}
      {top_count}
    </div>
    <div class="grid">
      {top_tail}
      {task_chart}
    </div>
    <div class="grid">
      {llm_breakdown_chart}
      {checkpoint_breakdown_chart}
    </div>
    <section>
      <h2>Top Runtime Hotspots</h2>
      {operation_table}
    </section>
    <section>
      <h2>Per-Task Summary</h2>
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
    metric_names = sorted(
        {
            metric_name
            for item in items
            for metric_name in item.metrics
        }
    )
    headers = ["task_id", "sandbox_ids", "agent_type", "llm_service_type"] + metric_names
    rows = []
    for item in items:
        row = {
            "task_id": item.task_id,
            "sandbox_ids": ",".join(item.sandbox_ids),
            "agent_type": item.agent_type,
            "llm_service_type": item.llm_service_type,
        }
        for metric_name in metric_names:
            row[metric_name] = f"{item.metrics.get(metric_name, 0.0):.6f}"
        rows.append(row)
    return headers, rows


def generate_report_bundle(
    telemetry_path: Path,
    *,
    output_dir: Path,
    run_id: str | None = None,
    top_k: int = 25,
) -> TelemetryAnalysis:
    analysis = analyze_telemetry_file(telemetry_path, run_id=run_id, top_k=top_k)
    target_dir = output_dir.expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    (target_dir / "summary.json").write_text(json.dumps(analysis.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    (target_dir / "report.html").write_text(render_report_html(analysis), encoding="utf-8")

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

    return analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze and visualize Agent-CR telemetry JSONL")
    parser.add_argument("--input", type=Path, required=True, help="Telemetry JSONL path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for report artifacts")
    parser.add_argument("--run-id", default=None, help="Optional run_id filter; defaults to dominant run in file")
    parser.add_argument("--top-k", type=int, default=25, help="Top-K operations/outliers to keep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis = generate_report_bundle(
        args.input,
        output_dir=args.output_dir,
        run_id=args.run_id,
        top_k=max(5, int(args.top_k)),
    )
    print(f"run_id: {analysis.run_id}")
    print(f"report: {(args.output_dir.expanduser().resolve() / 'report.html')}")
    print(f"summary_json: {(args.output_dir.expanduser().resolve() / 'summary.json')}")
    print(f"operation_csv: {(args.output_dir.expanduser().resolve() / 'operation_summary.csv')}")
    print(f"task_csv: {(args.output_dir.expanduser().resolve() / 'task_summary.csv')}")
    print(f"slow_ops_csv: {(args.output_dir.expanduser().resolve() / 'slow_operations.csv')}")


if __name__ == "__main__":
    main()

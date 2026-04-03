from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import (
    diagnose_benchmark_config,
    render_run_diagnosis_html,
    render_run_diagnosis_markdown,
    render_run_diagnosis_text,
    to_jsonable,
    write_run_diagnosis_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose benchmark failures from a benchmark config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sandbox-id", type=str, default=None)
    parser.add_argument("--task-id", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--max-log-lines", type=int, default=24)
    parser.add_argument("--max-tool-arg-chars", type=int, default=240)
    parser.add_argument("--max-visualized-tool-arg-chars", type=int, default=240)
    parser.add_argument("--max-tool-comparison-rows", type=int, default=40)
    parser.add_argument("--max-visualized-tool-rows", type=int, default=40)
    parser.add_argument("--max-timeline-events", type=int, default=36)
    parser.add_argument("--include-passed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = diagnose_benchmark_config(
        args.config,
        sandbox_id=args.sandbox_id,
        task_id=args.task_id,
        max_log_lines=max(1, int(args.max_log_lines)),
        max_tool_arg_chars=max(32, int(args.max_tool_arg_chars)),
        include_passed=bool(args.include_passed),
    )
    render_kwargs = {
        "max_visualized_tool_arg_chars": max(16, int(args.max_visualized_tool_arg_chars)),
        "max_tool_comparison_rows": max(1, int(args.max_tool_comparison_rows)),
        "max_visualized_tool_rows": max(1, int(args.max_visualized_tool_rows)),
        "max_timeline_events": max(1, int(args.max_timeline_events)),
    }
    text_report = render_run_diagnosis_text(report, **render_kwargs)
    print(text_report)
    if args.output_dir is not None:
        write_run_diagnosis_outputs(report, args.output_dir, **render_kwargs)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(to_jsonable(report), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(
            render_run_diagnosis_markdown(report, **render_kwargs) + "\n",
            encoding="utf-8",
        )
    if args.output_html is not None:
        args.output_html.parent.mkdir(parents=True, exist_ok=True)
        args.output_html.write_text(
            render_run_diagnosis_html(report, **render_kwargs) + "\n",
            encoding="utf-8",
        )

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .artifacts import load_artifacts
from .csv_report import parse_csv_report
from .dataset import load_dataset_index
from .heuristics import build_findings
from .iflow_report import (
    attach_observed_key_events,
    compare_trace_and_session_tool_calls,
    extract_trace_tool_calls,
    summarize_iflow_session,
    summarize_replay_trace,
)
from .log_report import parse_log, select_log_excerpt
from .models import Finding, RunDiagnosis, SandboxDiagnosis, to_jsonable
from .report import (
    render_run_diagnosis_html,
    render_run_diagnosis_markdown,
    render_run_diagnosis_text,
    write_run_diagnosis_outputs,
)
from .telemetry_report import parse_telemetry


def diagnose_benchmark_config(
    config_path: Path,
    *,
    sandbox_id: str | None = None,
    task_id: str | None = None,
    max_log_lines: int = 24,
    max_tool_arg_chars: int = 240,
    include_passed: bool = False,
) -> RunDiagnosis:
    loaded = load_artifacts(config_path)
    dataset_index = load_dataset_index(loaded.context.task_dataset_path)
    csv_report = parse_csv_report(loaded.context.csv_path, dataset_index.tasks)
    parsed_log = parse_log(loaded.context.log_path, command_preview_chars=max_tool_arg_chars)
    parsed_telemetry = parse_telemetry(
        loaded.context.telemetry_path,
        command_preview_chars=max_tool_arg_chars,
    )
    rows_to_consider = list(csv_report.rows)
    selected_sandbox_ids: set[str] = set()
    if sandbox_id is not None:
        selected_sandbox_ids.add(sandbox_id)
    elif task_id is not None:
        selected_sandbox_ids.update(row.sandbox_id for row in rows_to_consider if row.task_id == task_id)
    else:
        for row in rows_to_consider:
            if include_passed or row.classification != "passed":
                selected_sandbox_ids.add(row.sandbox_id)
    sandboxes: dict[str, SandboxDiagnosis] = {}
    rows_by_sandbox: dict[str, list] = defaultdict(list)
    for row in rows_to_consider:
        rows_by_sandbox[row.sandbox_id].append(row)
    for selected_id in sorted(selected_sandbox_ids):
        sandbox_rows = rows_by_sandbox.get(selected_id, [])
        task_ids = {row.task_id for row in sandbox_rows}
        dataset_tasks = []
        for current_task_id in sorted(task_ids):
            dataset_tasks.extend(dataset_index.by_task_id.get(current_task_id, []))
        resolved_task_id = next(iter(task_ids), selected_id)
        primary_task = dataset_tasks[0] if dataset_tasks else None
        replay_trace_summary = summarize_replay_trace(primary_task)
        trace_tool_calls = extract_trace_tool_calls(
            primary_task,
            max_tool_arg_chars=max_tool_arg_chars,
        )
        session_summary, session_tool_calls = summarize_iflow_session(
            benchmark_root=loaded.context.actual_benchmark_root,
            sandbox_id=selected_id,
            max_tool_arg_chars=max_tool_arg_chars,
        )
        session_tool_calls = attach_observed_key_events(
            session_tool_calls,
            key_events=parsed_log.key_events_by_sandbox.get(selected_id, []),
            session_end_timestamp=session_summary.get("last_entry_timestamp")
            if isinstance(session_summary.get("last_entry_timestamp"), str)
            else None,
        )
        alignment_summary = compare_trace_and_session_tool_calls(trace_tool_calls, session_tool_calls)
        timeline = []
        timeline.extend(parsed_log.timeline_by_sandbox.get(selected_id, []))
        timeline.extend(parsed_telemetry.timeline_by_sandbox.get(selected_id, []))
        timeline.sort(key=lambda item: (item.timestamp or "", item.source, item.label))
        log_lines = parsed_log.lines_by_sandbox.get(selected_id, [])
        diagnosis = SandboxDiagnosis(
            sandbox_id=selected_id,
            task_id=resolved_task_id,
            status=sandbox_rows[0].classification if sandbox_rows else "missing",
            dataset_tasks=dataset_tasks,
            csv_rows=sandbox_rows,
            findings=[],
            timeline=timeline,
            telemetry_records=parsed_telemetry.records_by_sandbox.get(selected_id, []),
            log_lines=log_lines,
            log_excerpt=select_log_excerpt(log_lines, limit=max_log_lines),
            tool_calls=session_tool_calls,
            trace_tool_calls=trace_tool_calls,
            tool_alignment_summary=alignment_summary,
            replay_trace_summary=replay_trace_summary,
            session_summary=session_summary,
            raw_evidence=[],
            notes=[
                (
                    f"telemetry_runtime_command_count={len(parsed_telemetry.tool_calls_by_sandbox.get(selected_id, []))}"
                    if parsed_telemetry.tool_calls_by_sandbox.get(selected_id)
                    else ""
                )
            ],
        )
        diagnosis.findings = build_findings(diagnosis)
        sandboxes[selected_id] = diagnosis
    missing_to_include = []
    if sandbox_id is None:
        if task_id is not None:
            missing_to_include = [item for item in csv_report.missing_tasks if item.task_id == task_id]
        else:
            missing_to_include = list(csv_report.missing_tasks)
    for missing in missing_to_include:
        pseudo_id = f"missing-task-{missing.dataset_index}"
        if pseudo_id in sandboxes:
            continue
        dataset_task = dataset_index.tasks[missing.dataset_index]
        diagnosis = SandboxDiagnosis(
            sandbox_id=pseudo_id,
            task_id=missing.task_id,
            status="missing",
            dataset_tasks=[dataset_task],
            csv_rows=[],
            findings=[],
            timeline=[],
            telemetry_records=parsed_telemetry.records_by_task.get(missing.task_id, []),
            log_lines=parsed_log.lines_by_task.get(missing.task_id, []),
            log_excerpt=select_log_excerpt(parsed_log.lines_by_task.get(missing.task_id, []), limit=max_log_lines),
            tool_calls=[],
            trace_tool_calls=extract_trace_tool_calls(dataset_task, max_tool_arg_chars=max_tool_arg_chars),
            tool_alignment_summary={},
            replay_trace_summary=summarize_replay_trace(dataset_task),
            session_summary={"applicable": False, "reason": "no sandbox CSV row emitted"},
            raw_evidence=[],
            notes=[missing.reason],
        )
        diagnosis.findings = build_findings(diagnosis)
        sandboxes[pseudo_id] = diagnosis
    failed_sandboxes = sorted(
        sandbox_id
        for sandbox_id, diagnosis in sandboxes.items()
        if include_passed or diagnosis.status != "passed"
    )
    run_findings: list[Finding] = []
    if loaded.context.actual_benchmark_root is None:
        run_findings.append(
            Finding(
                severity="medium",
                title="Benchmark root could not be inferred from the log",
                summary="The log did not contain a unique `runc --root .../runtime-state` path, so iFlow sandbox lookup may be incomplete.",
                category="trace",
                evidence_refs=(),
            )
        )
    if csv_report.missing_tasks:
        run_findings.append(
            Finding(
                severity="medium",
                title="Dataset coverage is incomplete",
                summary=f"{len(csv_report.missing_tasks)} dataset task rows did not produce CSV output rows.",
                category="task",
                evidence_refs=(),
            )
        )
    return RunDiagnosis(
        context=loaded.context,
        dataset_tasks=dataset_index.tasks,
        csv_rows=csv_report.rows,
        missing_tasks=csv_report.missing_tasks,
        failed_sandboxes=failed_sandboxes,
        sandboxes=sandboxes,
        run_findings=run_findings,
    )


__all__ = [
    "RunDiagnosis",
    "SandboxDiagnosis",
    "diagnose_benchmark_config",
    "render_run_diagnosis_html",
    "render_run_diagnosis_markdown",
    "render_run_diagnosis_text",
    "to_jsonable",
    "write_run_diagnosis_outputs",
]

from __future__ import annotations

from .models import Finding, SandboxDiagnosis


def _contains(lines: list[str], *needles: str) -> bool:
    lowered = "\n".join(lines).lower()
    return any(needle.lower() in lowered for needle in needles)


def _final_messages(session_summary: dict[str, object]) -> list[str]:
    raw = session_summary.get("final_messages", [])
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _task_timeout_ms(diagnosis: SandboxDiagnosis) -> float | None:
    for task in diagnosis.dataset_tasks:
        task_config = task.raw.get("task_config")
        if not isinstance(task_config, dict):
            continue
        options = task_config.get("options")
        if not isinstance(options, dict):
            continue
        raw = options.get("max_agent_timeout_sec")
        try:
            return float(raw) * 1000.0
        except (TypeError, ValueError):
            continue
    return None


def build_findings(diagnosis: SandboxDiagnosis) -> list[Finding]:
    findings: list[Finding] = []
    log_lines = diagnosis.log_lines or diagnosis.log_excerpt
    verification_failed = any(
        row.verification_status == "failed" or row.classification == "failed"
        for row in diagnosis.csv_rows
    )
    if _contains(log_lines, "version '", "unable to locate package", "apt-get install", "run-tests stderr"):
        findings.append(
            Finding(
                severity="high",
                title="Verifier setup/package failure",
                summary="Verification logs include environment setup or package-install failures, so the verifier may be failing before it checks task correctness.",
                category="verifier",
                evidence_refs=tuple(
                    f"log:{line.split(':', 1)[0]}"
                    for line in log_lines
                    if "version '" in line.lower() or "unable to locate package" in line.lower()
                ),
            )
        )
    alignment = diagnosis.tool_alignment_summary
    if alignment:
        if int(alignment.get("dummy_tool_call_count", 0)) > 0:
            findings.append(
                Finding(
                    severity="medium",
                    title="Replay produced dummy tool calls",
                    summary="Observed tool calls include replay-generated dummy calls, which usually means the sandbox replayed duplicate tool requests after a restore.",
                    category="recovery",
                    evidence_refs=(),
                )
            )
        if int(alignment.get("session_only_count", 0)) > 0 or int(alignment.get("trace_only_count", 0)) > 0:
            findings.append(
                Finding(
                    severity="medium",
                    title="Trace/session tool sequence is misaligned",
                    summary="The observed iFlow session tool-call sequence does not fully align with the original replay trace, which can indicate replay drift, duplicate execution, or missing actions.",
                    category="trace",
                    evidence_refs=(),
                )
            )
    if verification_failed and diagnosis.session_summary.get("available") and diagnosis.replay_trace_summary.get("available"):
        findings.append(
            Finding(
                severity="medium",
                title="Replay completed but verification failed",
                summary="Replay/session artifacts exist for this sandbox, so the failure is more likely in task behavior or recovery behavior than in basic trace availability.",
                category="task",
                evidence_refs=(),
            )
        )
    if _contains(log_lines, "no such file or directory", "not found", "missing") or any(
        (tool.raw_result_preview or "").lower().find("no such file or directory") >= 0
        for tool in diagnosis.tool_calls
    ):
        findings.append(
            Finding(
                severity="medium",
                title="Expected artifact was missing",
                summary="Logs or tool results mention missing files or binaries, so the sandbox may have reported success without producing the expected output state.",
                category="task",
                evidence_refs=(),
            )
        )
    timeline_labels = [item.label for item in diagnosis.timeline]
    if "fault injected" in timeline_labels and ("checkpoint" in timeline_labels or "restore" in timeline_labels):
        findings.append(
            Finding(
                severity="medium",
                title="Fault and recovery events occurred in this sandbox",
                summary="The timeline includes fault injection together with checkpoint or restore activity, so the failure should be read in the context of recovery behavior.",
                category="recovery",
                evidence_refs=tuple(
                    item.evidence_ref
                    for item in diagnosis.timeline
                    if item.label in {"fault injected", "checkpoint", "restore"} and item.evidence_ref
                ),
            )
        )
    timeout_ms = _task_timeout_ms(diagnosis)
    longest_tool_ms = max((tool.duration_ms or 0.0) for tool in diagnosis.tool_calls) if diagnosis.tool_calls else 0.0
    if longest_tool_ms > 0 and (longest_tool_ms >= 30000.0 or (timeout_ms is not None and longest_tool_ms >= timeout_ms * 0.2)):
        findings.append(
            Finding(
                severity="medium",
                title="One or more tool calls were expensive",
                summary="Observed tool execution time is large enough to contribute meaningfully to timeout pressure or long-tail recovery behavior.",
                category="task",
                evidence_refs=(),
            )
        )
    final_messages = _final_messages(diagnosis.session_summary)
    success_like_final = any(
        phrase in message.lower()
        for message in final_messages
        for phrase in ("success", "completed", "done", "built", "finished")
    )
    failing_tool = any(tool.exit_code not in (None, 0) for tool in diagnosis.tool_calls)
    if success_like_final and (verification_failed or failing_tool):
        findings.append(
            Finding(
                severity="medium",
                title="Final assistant summary conflicts with observed evidence",
                summary="The final assistant/session message sounds successful, but verification or tool results still show failure.",
                category="task",
                evidence_refs=(),
            )
        )
    if diagnosis.replay_trace_summary.get("applicable") and not diagnosis.replay_trace_summary.get("available", False):
        findings.append(
            Finding(
                severity="medium",
                title="Replay trace unavailable",
                summary=str(diagnosis.replay_trace_summary.get("reason", "Replay trace could not be loaded.")),
                category="trace",
                evidence_refs=(),
            )
        )
    if diagnosis.session_summary.get("applicable") and not diagnosis.session_summary.get("available", False):
        findings.append(
            Finding(
                severity="medium",
                title="iFlow session unavailable",
                summary=str(diagnosis.session_summary.get("reason", "Session trajectory could not be loaded.")),
                category="trace",
                evidence_refs=(),
            )
        )
    order = {"verifier": 0, "task": 1, "recovery": 2, "trace": 3}
    severity_order = {"high": 0, "medium": 1, "low": 2}
    findings.sort(key=lambda item: (order.get(item.category, 99), severity_order.get(item.severity, 99), item.title))
    return findings

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import TimelineItem


_TIMESTAMP_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)")
_SANDBOX_RE = re.compile(r"\bsandbox(?:_id)?(?:=| )([A-Za-z0-9_.:-]+)")
_TASK_RE = re.compile(r"\btask_id=([A-Za-z0-9_.:-]+)")
_COMMAND_RE = re.compile(r"\bcommand=(.+)$")
_JOB_ID_RE = re.compile(r"\bjob[-_ ]id[= ]([A-Za-z0-9_.:-]+)|\bjob[= ]([A-Za-z0-9_.:-]+)")
_CHECKPOINT_PROCESS_RE = re.compile(r"\bcheckpoint_process=(True|False)")
_CHECKPOINT_FILESYSTEM_RE = re.compile(r"\bcheckpoint_filesystem=(True|False)")


@dataclass(frozen=True)
class ParsedLog:
    lines_by_sandbox: dict[str, list[str]]
    lines_by_task: dict[str, list[str]]
    timeline_by_sandbox: dict[str, list[TimelineItem]]
    key_events_by_sandbox: dict[str, list[TimelineItem]]
    inferred_roots: tuple[str, ...]
    all_lines: list[str]


def _extract_timestamp(line: str) -> str | None:
    match = _TIMESTAMP_RE.match(line)
    if not match:
        return None
    return match.group(1)


def _line_label(lowered: str) -> str | None:
    if "injecting fault" in lowered or "notify fault" in lowered or "notifying fault" in lowered:
        return "fault injected"
    if "checkpoint" in lowered and ("completed" in lowered or "complete" in lowered or "start" in lowered):
        return "checkpoint"
    if "restore" in lowered and ("completed" in lowered or "restor" in lowered or "start" in lowered):
        return "restore"
    if "benchmark.task.verify" in lowered or "run-tests" in lowered:
        return "verification"
    if "replay_is_complete" in lowered or "iflow" in lowered and "complete" in lowered:
        return "iflow"
    return None


def _key_event_item(
    raw_line: str,
    *,
    timestamp: str | None,
    sandbox_id: str,
    task_id: str | None,
    line_number: int,
    checkpoint_scope: str | None,
) -> TimelineItem | None:
    lowered = raw_line.lower()
    if "finished checkpoint job " in lowered and f"for sandbox {sandbox_id.lower()}" in lowered:
        scope_prefix = f"Checkpoint {checkpoint_scope} " if checkpoint_scope else "Checkpoint "
        if "status=succeeded" in lowered:
            return TimelineItem(
                source="log",
                timestamp=timestamp,
                label=f"{scope_prefix}Succeed",
                detail=raw_line,
                sandbox_id=sandbox_id,
                task_id=task_id,
                evidence_ref=f"log:{line_number}",
            )
        if "status=failed" in lowered:
            return TimelineItem(
                source="log",
                timestamp=timestamp,
                label=f"{scope_prefix}Failed",
                detail=raw_line,
                sandbox_id=sandbox_id,
                task_id=task_id,
                evidence_ref=f"log:{line_number}",
            )
    if "benchmark notifying fault sandbox=" in lowered:
        return TimelineItem(
            source="log",
            timestamp=timestamp,
            label="Fault Injection Succeed",
            detail=raw_line,
            sandbox_id=sandbox_id,
            task_id=task_id,
            evidence_ref=f"log:{line_number}",
        )
    if "timed out waiting for fault injection window, skipping fault injection" in lowered:
        return TimelineItem(
            source="log",
            timestamp=timestamp,
            label="Fault Injection Failed",
            detail=raw_line,
            sandbox_id=sandbox_id,
            task_id=task_id,
            evidence_ref=f"log:{line_number}",
        )
    if "recovery restore succeeded sandbox=" in lowered:
        return TimelineItem(
            source="log",
            timestamp=timestamp,
            label="Restore Succeed",
            detail=raw_line,
            sandbox_id=sandbox_id,
            task_id=task_id,
            evidence_ref=f"log:{line_number}",
        )
    if "recovery restore failed sandbox=" in lowered or "recovery restore failed; invoking relaunch handler sandbox=" in lowered:
        return TimelineItem(
            source="log",
            timestamp=timestamp,
            label="Restore Failed",
            detail=raw_line,
            sandbox_id=sandbox_id,
            task_id=task_id,
            evidence_ref=f"log:{line_number}",
        )
    return None


def _extract_job_id(line: str) -> str | None:
    match = _JOB_ID_RE.search(line)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _extract_checkpoint_scope(line: str) -> str | None:
    process_match = _CHECKPOINT_PROCESS_RE.search(line)
    filesystem_match = _CHECKPOINT_FILESYSTEM_RE.search(line)
    if process_match is None or filesystem_match is None:
        return None
    process_enabled = process_match.group(1) == "True"
    filesystem_enabled = filesystem_match.group(1) == "True"
    if process_enabled and filesystem_enabled:
        return "P+F"
    if process_enabled:
        return "P"
    if filesystem_enabled:
        return "F"
    return None


def _crop_command(line: str, *, limit: int) -> str:
    match = _COMMAND_RE.search(line)
    if not match:
        return line
    command = match.group(1).strip()
    if len(command) <= limit:
        return line
    cropped = f"{command[: max(0, limit - 3)].rstrip()}..."
    return f"{line[: match.start(1)]}{cropped}"


def parse_log(log_path: Path | None, *, command_preview_chars: int = 240) -> ParsedLog:
    if log_path is None or not log_path.exists():
        return ParsedLog(
            lines_by_sandbox={},
            lines_by_task={},
            timeline_by_sandbox={},
            key_events_by_sandbox={},
            inferred_roots=(),
            all_lines=[],
        )
    from .artifacts import infer_actual_benchmark_root

    _, inferred_roots = infer_actual_benchmark_root(log_path)
    lines_by_sandbox: dict[str, list[str]] = defaultdict(list)
    lines_by_task: dict[str, list[str]] = defaultdict(list)
    timeline_by_sandbox: dict[str, list[TimelineItem]] = defaultdict(list)
    key_events_by_sandbox: dict[str, list[TimelineItem]] = defaultdict(list)
    all_lines: list[str] = []
    pending_checkpoint_scope_by_sandbox: dict[str, str] = {}
    checkpoint_scope_by_job_id: dict[str, str] = {}
    continuation_sandbox_ids: list[str] = []
    continuation_task_ids: list[str] = []
    for line_number, raw_line in enumerate(
        log_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        cropped_line = _crop_command(raw_line, limit=command_preview_chars)
        all_lines.append(cropped_line)
        sandbox_ids = sorted(set(_SANDBOX_RE.findall(raw_line)))
        task_ids = sorted(set(_TASK_RE.findall(raw_line)))
        timestamp = _extract_timestamp(raw_line)
        if not sandbox_ids and continuation_sandbox_ids and timestamp is None:
            sandbox_ids = list(continuation_sandbox_ids)
        if not task_ids and continuation_task_ids and timestamp is None:
            task_ids = list(continuation_task_ids)
        lowered = raw_line.lower()
        label = _line_label(lowered)
        checkpoint_scope = _extract_checkpoint_scope(raw_line)
        if checkpoint_scope is not None:
            for sandbox_id in sandbox_ids:
                pending_checkpoint_scope_by_sandbox[sandbox_id] = checkpoint_scope
        job_id = _extract_job_id(raw_line)
        if "queued checkpoint job " in lowered and job_id is not None:
            for sandbox_id in sandbox_ids:
                scope = pending_checkpoint_scope_by_sandbox.get(sandbox_id)
                if scope is not None:
                    checkpoint_scope_by_job_id[job_id] = scope
        if "run-tests stdout sandbox=" in lowered or "run-tests stderr sandbox=" in lowered:
            continuation_sandbox_ids = list(sandbox_ids)
            continuation_task_ids = list(task_ids)
        elif timestamp is not None and not sandbox_ids:
            continuation_sandbox_ids = []
            continuation_task_ids = []
        for sandbox_id in sandbox_ids:
            lines_by_sandbox[sandbox_id].append(f"{line_number}: {cropped_line}")
            if label is not None:
                timeline_by_sandbox[sandbox_id].append(
                    TimelineItem(
                        source="log",
                        timestamp=timestamp,
                        label=label,
                        detail=cropped_line,
                        sandbox_id=sandbox_id,
                        task_id=task_ids[0] if task_ids else None,
                        evidence_ref=f"log:{line_number}",
                    )
                )
            key_event = _key_event_item(
                raw_line,
                timestamp=timestamp,
                sandbox_id=sandbox_id,
                task_id=task_ids[0] if task_ids else None,
                line_number=line_number,
                checkpoint_scope=(
                    checkpoint_scope_by_job_id.get(job_id)
                    if job_id is not None and checkpoint_scope_by_job_id.get(job_id) is not None
                    else pending_checkpoint_scope_by_sandbox.get(sandbox_id)
                ),
            )
            if key_event is not None:
                key_events_by_sandbox[sandbox_id].append(key_event)
        for task_id in task_ids:
            lines_by_task[task_id].append(f"{line_number}: {cropped_line}")
    return ParsedLog(
        lines_by_sandbox=dict(lines_by_sandbox),
        lines_by_task=dict(lines_by_task),
        timeline_by_sandbox={key: value for key, value in timeline_by_sandbox.items()},
        key_events_by_sandbox={key: value for key, value in key_events_by_sandbox.items()},
        inferred_roots=inferred_roots,
        all_lines=all_lines,
    )


def select_log_excerpt(lines: list[str], *, limit: int = 24) -> list[str]:
    if len(lines) <= limit:
        return list(lines)
    failure_keywords = (
        "failed",
        "error",
        "traceback",
        "checkpoint",
        "restore",
        "fault",
        "verify",
        "run-tests",
        "stderr",
    )
    indexed: list[str] = []
    indexed.extend(lines[: min(6, len(lines))])
    for line in lines:
        lowered = line.lower()
        if any(keyword in lowered for keyword in failure_keywords):
            indexed.append(line)
    indexed.extend(lines[max(0, len(lines) - 6):])
    deduped: list[str] = []
    seen: set[str] = set()
    for line in indexed:
        if line in seen:
            continue
        seen.add(line)
        deduped.append(line)
        if len(deduped) >= limit:
            break
    return deduped

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from integrations.llm_services.iflow_trace_replay.service import parse_replay_trace

from .models import DatasetTaskInfo, TimelineItem, ToolCallSummary


_DUMMY_SHELL_COMMAND = 'sh -lc "echo hello world >> /dev/null"'
_EXIT_CODE_RE = re.compile(r"Exit Code:\s+([^\n]+)")
_STDERR_RE = re.compile(r"Stderr:\s*(.*?)(?:\n(?:Error:|Exit Code:)|\Z)", re.DOTALL)
_ERROR_LINE_RE = re.compile(r"Error:\s*(.*?)(?:\n(?:Exit Code:)|\Z)", re.DOTALL)
_ERROR_INDICATOR_PATTERNS = (
    "error",
    "failed",
    "traceback",
    "exception",
    "timed out",
    "timeout",
    "not found",
    "no such file",
)
_NOISE_LINE_PATTERNS = (
    "bash: warning: setlocale: lc_all: cannot change locale (zh_cn.utf-8)",
    "perl: warning: setting locale failed.",
    "perl: warning: please check that your locale settings:",
    'perl: warning: falling back to the standard locale ("c").',
)
_REPLAY_MARKER_KEYS = frozenset({"trace_cursor", "consumed_response_count", "action_replay"})


def _crop_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(0, limit - 3)].rstrip()}..."


def _parse_exit_code(result_text: str) -> int | None:
    match = _EXIT_CODE_RE.search(result_text)
    if not match:
        return None
    raw_value = match.group(1).strip()
    if raw_value in {"(none)", ""}:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def _extract_section(pattern: re.Pattern[str], result_text: str) -> str:
    match = pattern.search(result_text)
    if not match:
        return ""
    return match.group(1).strip()


def _strip_noise_lines(text: str) -> str:
    if not text:
        return text
    kept: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line
        if "bash: warning: setlocale: LC_ALL: cannot change locale (zh_CN.UTF-8)" in line:
            line = line.replace(
                "bash: warning: setlocale: LC_ALL: cannot change locale (zh_CN.UTF-8)",
                "",
            ).strip()
            if line in {"", "Stderr:", "stderr:"}:
                continue
        lowered = line.strip().lower()
        if lowered in _NOISE_LINE_PATTERNS:
            continue
        if lowered.startswith("\tlanguage =") or lowered.startswith("\tlc_all =") or lowered.startswith("\tlang ="):
            continue
        if lowered == "are supported and installed on your system.":
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _result_summary(*, exit_code: int | None, stderr_text: str, error_text: str, output: str) -> tuple[bool, str | None]:
    lowered = output.lower()
    sanitized = lowered.replace("error: (none)", "").replace("stderr: (none)", "")
    indicator = any(token in sanitized for token in _ERROR_INDICATOR_PATTERNS)
    if exit_code not in (None, 0):
        return (True, f"nonzero exit ({exit_code})")
    if error_text and error_text.lower() not in {"(none)", "none"}:
        return (True, f"error: {_crop_text(error_text, limit=72)}")
    if stderr_text and stderr_text.lower() not in {"(none)", ""}:
        return (True, f"stderr: {_crop_text(stderr_text, limit=72)}")
    if indicator:
        return (True, "output contains error indicators")
    return (False, "ok" if exit_code == 0 else None)


def _parse_iso_timestamp(raw_value: object) -> datetime | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    text = raw_value.strip()
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _parse_log_timestamp(raw_value: object) -> datetime | None:
    if not isinstance(raw_value, str) or not raw_value.strip():
        return None
    try:
        naive = datetime.strptime(raw_value.strip(), "%Y-%m-%d %H:%M:%S,%f")
    except ValueError:
        return None
    local_tz = datetime.now().astimezone().tzinfo or timezone.utc
    return naive.replace(tzinfo=local_tz)


def _parse_any_timestamp(raw_value: object) -> datetime | None:
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if text and "T" not in text and "," in text:
            return _parse_log_timestamp(text) or _parse_iso_timestamp(text)
    return _parse_iso_timestamp(raw_value) or _parse_log_timestamp(raw_value)


def _collect_replay_marker_paths(value: object, *, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            child_prefix = f"{prefix}.{key_text}"
            if key_text in _REPLAY_MARKER_KEYS:
                paths.append(child_prefix)
            paths.extend(_collect_replay_marker_paths(item, prefix=child_prefix))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_collect_replay_marker_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _trace_tool_turn_duration_ms(*, response_timestamp: float | None, next_request_timestamp: float | None) -> float | None:
    if response_timestamp is None or next_request_timestamp is None:
        return None
    return max(0.0, (next_request_timestamp - response_timestamp) * 1000.0)


def _extract_trace_tool_results(request_payload: dict[str, Any], *, max_tool_arg_chars: int) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        return ([], {})
    ordered: list[dict[str, Any]] = []
    by_tool_call_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if not isinstance(message, dict) or str(message.get("role", "")) != "tool":
            continue
        tool_call_id = str(message.get("tool_call_id", "")).strip() or None
        raw_content = message.get("content")
        output = raw_content if isinstance(raw_content, str) else json.dumps(raw_content, ensure_ascii=False, sort_keys=True)
        clean_output = _strip_noise_lines(output)
        exit_code = _parse_exit_code(output)
        stderr_text = _strip_noise_lines(_extract_section(_STDERR_RE, clean_output))
        error_text = _strip_noise_lines(_extract_section(_ERROR_LINE_RE, clean_output))
        has_error_indicators, result_summary = _result_summary(
            exit_code=exit_code,
            stderr_text=stderr_text,
            error_text=error_text,
            output=clean_output,
        )
        info = {
            "tool_call_id": tool_call_id,
            "exit_code": exit_code,
            "has_error_indicators": has_error_indicators,
            "result_summary": result_summary,
            "raw_result_preview": _crop_text(clean_output, limit=max_tool_arg_chars),
        }
        ordered.append(info)
        if tool_call_id is not None:
            by_tool_call_id[tool_call_id] = info
    return (ordered, by_tool_call_id)


def _tool_signature(tool_name: str, raw_arguments: dict[str, Any] | None) -> str:
    args = raw_arguments or {}
    command = args.get("command")
    if isinstance(command, str):
        return f"{tool_name}::{command.strip()}"
    return f"{tool_name}::{json.dumps(args, sort_keys=True, ensure_ascii=False)}"


def _is_dummy_tool_call(tool_name: str, raw_arguments: dict[str, Any] | None) -> bool:
    args = raw_arguments or {}
    if tool_name == "run_shell_command":
        command = args.get("command")
        return isinstance(command, str) and command.strip() == _DUMMY_SHELL_COMMAND
    return False


def summarize_replay_trace(task: DatasetTaskInfo | None) -> dict[str, Any]:
    if task is None:
        return {"applicable": False, "reason": "no dataset task"}
    if task.llm_service_type != "iflow_trace_replay" and task.trace_path is None:
        return {"applicable": False, "reason": "task is not an iflow replay task"}
    if task.trace_path is None:
        return {"applicable": True, "available": False, "reason": "dataset row has no trace_path"}
    if not task.trace_path.exists():
        return {
            "applicable": True,
            "available": False,
            "trace_path": str(task.trace_path),
            "reason": "trace file missing",
        }
    parsed = parse_replay_trace(task.trace_path)
    delays = [exchange.trace_delay_ms for exchange in parsed.exchanges if exchange.trace_delay_ms is not None]
    tool_turn_durations: list[float] = []
    tool_call_count = 0
    for exchange_index, exchange in enumerate(parsed.exchanges):
        message = ((exchange.response.get("choices") or [{}])[0].get("message") or {})
        tool_calls = message.get("tool_calls") if isinstance(message, dict) else None
        if isinstance(tool_calls, list):
            tool_call_count += len(tool_calls)
            next_request_timestamp = None
            if exchange_index + 1 < len(parsed.exchanges):
                next_request_timestamp = parsed.exchanges[exchange_index + 1].request_timestamp
            tool_turn_duration_ms = _trace_tool_turn_duration_ms(
                response_timestamp=exchange.response_timestamp,
                next_request_timestamp=next_request_timestamp,
            )
            if tool_turn_duration_ms is not None:
                tool_turn_durations.append(tool_turn_duration_ms)
    return {
        "applicable": True,
        "available": True,
        "trace_path": str(parsed.trace_path),
        "response_count": len(parsed.responses),
        "malformed_line_count": len(parsed.malformed_lines),
        "exchange_count": len(parsed.exchanges),
        "tool_call_count": tool_call_count,
        "first_request_timestamp": parsed.exchanges[0].request_timestamp if parsed.exchanges else None,
        "last_response_timestamp": parsed.exchanges[-1].response_timestamp if parsed.exchanges else None,
        "avg_trace_delay_ms": sum(delays) / len(delays) if delays else None,
        "avg_trace_tool_turn_ms": (sum(tool_turn_durations) / len(tool_turn_durations)) if tool_turn_durations else None,
    }


def extract_trace_tool_calls(
    task: DatasetTaskInfo | None,
    *,
    max_tool_arg_chars: int = 240,
) -> list[ToolCallSummary]:
    if task is None or task.trace_path is None or not task.trace_path.exists():
        return []
    parsed = parse_replay_trace(task.trace_path)
    tool_calls: list[ToolCallSummary] = []
    call_index = 0
    for exchange_index, exchange in enumerate(parsed.exchanges):
        message = ((exchange.response.get("choices") or [{}])[0].get("message") or {})
        if not isinstance(message, dict):
            continue
        raw_tool_calls = message.get("tool_calls")
        if not isinstance(raw_tool_calls, list):
            continue
        next_request = parsed.exchanges[exchange_index + 1].request if exchange_index + 1 < len(parsed.exchanges) else {}
        next_request_timestamp = parsed.exchanges[exchange_index + 1].request_timestamp if exchange_index + 1 < len(parsed.exchanges) else None
        trace_tool_duration_ms = _trace_tool_turn_duration_ms(
            response_timestamp=exchange.response_timestamp,
            next_request_timestamp=next_request_timestamp,
        )
        ordered_results, results_by_id = _extract_trace_tool_results(
            next_request if isinstance(next_request, dict) else {},
            max_tool_arg_chars=max_tool_arg_chars,
        )
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            tool_call_id = str(raw_tool_call.get("id", "")).strip() or None
            function = raw_tool_call.get("function")
            if not isinstance(function, dict):
                continue
            tool_name = str(function.get("name", "")).strip() or "unknown"
            raw_arguments = function.get("arguments")
            parsed_arguments: dict[str, Any] | None = None
            if isinstance(raw_arguments, str):
                try:
                    loaded = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    loaded = None
                if isinstance(loaded, dict):
                    parsed_arguments = loaded
            result_info = results_by_id.get(tool_call_id) if tool_call_id is not None else None
            if result_info is None and ordered_results:
                result_info = ordered_results.pop(0)
            arguments_preview = _crop_text(
                raw_arguments if isinstance(raw_arguments, str) else json.dumps(raw_arguments, ensure_ascii=False, sort_keys=True),
                limit=max_tool_arg_chars,
            )
            tool_calls.append(
                ToolCallSummary(
                    source="replay-trace",
                    timestamp=None if exchange.response_timestamp is None else str(exchange.response_timestamp),
                    tool_name=tool_name,
                    description=f"trace response {exchange.response_index}",
                    arguments_preview=arguments_preview,
                    call_index=call_index,
                    duration_ms=trace_tool_duration_ms,
                    exit_code=None if result_info is None else result_info["exit_code"],
                    raw_arguments=parsed_arguments,
                    is_dummy=_is_dummy_tool_call(tool_name, parsed_arguments),
                    has_error_indicators=False if result_info is None else bool(result_info["has_error_indicators"]),
                    result_summary=(
                        f"trace result unavailable (response {exchange.response_index})"
                        if result_info is None
                        else str(result_info["result_summary"] or "ok")
                    ),
                    raw_result_preview=None if result_info is None else str(result_info["raw_result_preview"]),
                )
            )
            call_index += 1
    return tool_calls


def _extract_text_segments(content: object) -> list[str]:
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                raw_text = item.get("text")
                if isinstance(raw_text, str) and raw_text.strip():
                    parts.append(raw_text)
    return parts


def _select_session_file(project_dir: Path) -> tuple[Path | None, list[Path]]:
    if not project_dir.exists():
        return (None, [])
    files = sorted(project_dir.glob("session-*.jsonl"))
    if not files:
        return (None, [])
    if len(files) == 1:
        return (files[0], files)
    newest = max(files, key=lambda path: path.stat().st_mtime_ns)
    return (newest, files)


def summarize_iflow_session(
    *,
    benchmark_root: Path | None,
    sandbox_id: str,
    max_tool_arg_chars: int = 240,
) -> tuple[dict[str, Any], list[ToolCallSummary]]:
    if benchmark_root is None:
        return ({"applicable": False, "reason": "benchmark root unavailable"}, [])
    project_dir = benchmark_root / "iflow" / sandbox_id / "iflow-state" / ".iflow" / "projects" / "-app"
    selected, all_files = _select_session_file(project_dir)
    if selected is None:
        return (
            {
                "applicable": True,
                "available": False,
                "project_dir": str(project_dir),
                "reason": "no session JSONL files found",
            },
            [],
        )
    assistant_messages = 0
    tool_calls: list[ToolCallSummary] = []
    tool_results = 0
    final_messages: list[str] = []
    pending_indices: list[int] = []
    replay_marker_paths: set[str] = set()
    call_index = 0
    last_timestamp: str | None = None
    with selected.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                continue
            replay_marker_paths.update(_collect_replay_marker_paths(payload))
            timestamp = str(payload.get("timestamp", "")) or None
            last_timestamp = timestamp or last_timestamp
            entry_type = str(payload.get("type", ""))
            message = payload.get("message", {})
            if not isinstance(message, dict):
                message = {}
            role = str(message.get("role", ""))
            content = message.get("content")
            if entry_type == "assistant" or role == "assistant":
                assistant_messages += 1
                text_parts = _extract_text_segments(content)
                if text_parts:
                    final_messages.append(text_parts[-1].strip())
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, dict):
                            continue
                        function_call = item.get("functionCall")
                        if not isinstance(function_call, dict):
                            continue
                        name = str(function_call.get("name", "")).strip() or "unknown"
                        args = function_call.get("args", {})
                        if not isinstance(args, dict):
                            args = {}
                        description = str(args.get("description", "")).strip()
                        arg_preview = _crop_text(
                            json.dumps(args, ensure_ascii=False, sort_keys=True),
                            limit=max_tool_arg_chars,
                        )
                        tool_calls.append(
                            ToolCallSummary(
                                source="iflow-session",
                                timestamp=timestamp,
                                tool_name=name,
                                description=description,
                                arguments_preview=arg_preview,
                                call_index=call_index,
                                raw_arguments=dict(args),
                                is_dummy=_is_dummy_tool_call(name, args),
                            )
                        )
                        pending_indices.append(len(tool_calls) - 1)
                        call_index += 1
            if entry_type == "user" and isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "tool_result":
                        continue
                    tool_results += 1
                    result_content = item.get("content", {})
                    if not isinstance(result_content, dict):
                        continue
                    function_response = result_content.get("functionResponse", {})
                    if not isinstance(function_response, dict):
                        continue
                    response = function_response.get("response", {})
                    if not isinstance(response, dict):
                        continue
                    output = str(response.get("output", ""))
                    clean_output = _strip_noise_lines(output)
                    exit_code = _parse_exit_code(output)
                    stderr_text = _strip_noise_lines(_extract_section(_STDERR_RE, clean_output))
                    error_text = _strip_noise_lines(_extract_section(_ERROR_LINE_RE, clean_output))
                    has_error_indicators, result_summary = _result_summary(
                        exit_code=exit_code,
                        stderr_text=stderr_text,
                        error_text=error_text,
                        output=clean_output,
                    )
                    if not pending_indices:
                        continue
                    update_index = pending_indices.pop(0)
                    summary = tool_calls[update_index]
                    duration_ms = None
                    started_at = _parse_iso_timestamp(summary.timestamp)
                    finished_at = _parse_iso_timestamp(timestamp)
                    if started_at is not None and finished_at is not None:
                        duration_ms = max(0.0, (finished_at - started_at).total_seconds() * 1000.0)
                    tool_calls[update_index] = ToolCallSummary(
                        source=summary.source,
                        timestamp=summary.timestamp,
                        tool_name=summary.tool_name,
                        description=summary.description,
                        arguments_preview=summary.arguments_preview,
                        call_index=summary.call_index,
                        status=str(payload.get("toolUseResult", {}).get("status", "")) or None,
                        exit_code=exit_code,
                        duration_ms=duration_ms,
                        is_dummy=summary.is_dummy,
                        matched_trace_index=summary.matched_trace_index,
                        has_error_indicators=has_error_indicators,
                        result_summary=result_summary,
                        raw_arguments=summary.raw_arguments,
                        raw_result_preview=_crop_text(clean_output, limit=max_tool_arg_chars),
                    )
    durations = [tool.duration_ms for tool in tool_calls if tool.duration_ms is not None]
    dummy_count = sum(1 for tool in tool_calls if tool.is_dummy)
    return (
        {
            "applicable": True,
            "available": True,
            "project_dir": str(project_dir),
            "session_file": str(selected),
            "session_file_count": len(all_files),
            "selected_newest_session": len(all_files) > 1,
            "assistant_message_count": assistant_messages,
            "tool_call_count": len(tool_calls),
            "tool_result_count": tool_results,
            "dummy_tool_call_count": dummy_count,
            "replay_marker_count": len(replay_marker_paths),
            "replay_marker_paths": sorted(replay_marker_paths)[:16],
            "max_tool_duration_ms": max(durations) if durations else None,
            "avg_tool_duration_ms": (sum(durations) / len(durations)) if durations else None,
            "last_entry_timestamp": last_timestamp,
            "final_messages": final_messages[-3:],
        },
        tool_calls,
    )


def attach_observed_key_events(
    observed_tool_calls: list[ToolCallSummary],
    *,
    key_events: list[TimelineItem],
    session_end_timestamp: str | None,
) -> list[ToolCallSummary]:
    if not observed_tool_calls or not key_events:
        return observed_tool_calls
    parsed_events: list[tuple[datetime, TimelineItem]] = []
    for event in key_events:
        event_timestamp = _parse_any_timestamp(event.timestamp)
        if event_timestamp is None:
            continue
        parsed_events.append((event_timestamp, event))
    if not parsed_events:
        return observed_tool_calls
    updated: list[ToolCallSummary] = []
    session_end = _parse_any_timestamp(session_end_timestamp)
    for index, tool in enumerate(observed_tool_calls):
        start = _parse_any_timestamp(tool.timestamp)
        next_start = None
        if index + 1 < len(observed_tool_calls):
            next_start = _parse_any_timestamp(observed_tool_calls[index + 1].timestamp)
        if next_start is None:
            next_start = session_end
        observed_key_events: list[TimelineItem] = []
        if start is not None:
            for event_timestamp, event in parsed_events:
                if event_timestamp < start:
                    continue
                if next_start is not None and event_timestamp >= next_start:
                    continue
                observed_key_events.append(event)
        updated.append(
            ToolCallSummary(
                source=tool.source,
                timestamp=tool.timestamp,
                tool_name=tool.tool_name,
                description=tool.description,
                arguments_preview=tool.arguments_preview,
                call_index=tool.call_index,
                status=tool.status,
                exit_code=tool.exit_code,
                duration_ms=tool.duration_ms,
                is_dummy=tool.is_dummy,
                matched_trace_index=tool.matched_trace_index,
                has_error_indicators=tool.has_error_indicators,
                result_summary=tool.result_summary,
                raw_arguments=tool.raw_arguments,
                raw_result_preview=tool.raw_result_preview,
                observed_key_events=tuple(observed_key_events),
            )
        )
    return updated


def compare_trace_and_session_tool_calls(
    trace_tool_calls: list[ToolCallSummary],
    session_tool_calls: list[ToolCallSummary],
) -> dict[str, Any]:
    trace_signatures = [_tool_signature(tool.tool_name, tool.raw_arguments) for tool in trace_tool_calls]
    session_signatures = [_tool_signature(tool.tool_name, tool.raw_arguments) for tool in session_tool_calls]
    non_dummy_session_signatures = [
        _tool_signature(tool.tool_name, tool.raw_arguments)
        for tool in session_tool_calls
        if not tool.is_dummy
    ]
    prefix_match_count = 0
    for trace_sig, session_sig in zip(trace_signatures, session_signatures):
        if trace_sig != session_sig:
            break
        prefix_match_count += 1
    matcher = SequenceMatcher(a=trace_signatures, b=non_dummy_session_signatures, autojunk=False)
    aligned = sum(block.size for block in matcher.get_matching_blocks())
    dummy_count = sum(1 for tool in session_tool_calls if tool.is_dummy)
    first_mismatch: dict[str, Any] | None = None
    for index in range(max(len(trace_tool_calls), len(session_tool_calls))):
        trace_tool = trace_tool_calls[index] if index < len(trace_tool_calls) else None
        session_tool = session_tool_calls[index] if index < len(session_tool_calls) else None
        if trace_tool is None or session_tool is None:
            first_mismatch = {
                "index": index,
                "trace": None if trace_tool is None else trace_tool.arguments_preview,
                "session": None if session_tool is None else session_tool.arguments_preview,
            }
            break
        if _tool_signature(trace_tool.tool_name, trace_tool.raw_arguments) != _tool_signature(session_tool.tool_name, session_tool.raw_arguments):
            first_mismatch = {
                "index": index,
                "trace": trace_tool.arguments_preview,
                "session": session_tool.arguments_preview,
            }
            break
    return {
        "trace_tool_call_count": len(trace_tool_calls),
        "session_tool_call_count": len(session_tool_calls),
        "session_non_dummy_tool_call_count": len(non_dummy_session_signatures),
        "dummy_tool_call_count": dummy_count,
        "exact_prefix_match_count": prefix_match_count,
        "aligned_non_dummy_count": aligned,
        "trace_only_count": max(0, len(trace_tool_calls) - aligned),
        "session_only_count": max(0, len(non_dummy_session_signatures) - aligned),
        "first_mismatch": first_mismatch,
    }

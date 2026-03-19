from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_RESPONSE_DELAY_MS = 250
_TOOL_RESULT_SENTINEL = "<tool-result>"
_IFLOW_SYSTEM_PROMPT_SENTINEL = "<iflow-system-prompt>"
_IFLOW_CONTEXT_SENTINEL = "<iflow-context-bootstrap>"
_IFLOW_CONTEXT_ACK_SENTINEL = "<iflow-context-ack>"
_VOLATILE_TEXT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+"
            r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
            r"\d{1,2},\s+\d{4}\b"
        ),
        "<date>",
    ),
    (
        re.compile(
            r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+"
            r"\d{1,2},\s+\d{4}\b"
        ),
        "<date>",
    ),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b"), "<timestamp>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<date>"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"), "<date>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"), "<time>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE), "<uuid>"),
    (re.compile(r"https?://[^\s\"']+"), "<url>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    (re.compile(r"/tmp/[^\s\"']+"), "/tmp/<temp>"),
)
_SYSTEM_REMINDER_PATTERN = re.compile(r"<system-reminder>.*?</system-reminder>", re.DOTALL)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReplayExchange:
    path: str
    request: dict[str, Any]
    response: dict[str, Any]
    lookup_key: tuple[str, int, int, int, int]
    counts_toward_progress: bool = True


@dataclass(frozen=True)
class ParsedReplayTrace:
    trace_path: Path
    responses: tuple[dict[str, Any], ...]
    malformed_lines: tuple[int, ...]
    exchanges: tuple[ReplayExchange, ...]


def _sandbox_id_from_request(headers: dict[str, str], payload: dict[str, Any]) -> str:
    sandbox_id = headers.get("X-Agent-Sandbox-Id", "").strip()
    if sandbox_id:
        return sandbox_id
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("sandbox_id"), str):
        return metadata["sandbox_id"]
    return "sandbox-unknown"


def _decode_json_payload(raw: object) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return dict(raw)
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    return parsed


def _is_non_replayable_stop_payload(parsed: dict[str, Any]) -> bool:
    if "reasoning" in parsed and "confidence" in parsed:
        return True
    if len(parsed) != 1:
        return False
    [(key, value)] = parsed.items()
    return isinstance(key, str) and key.startswith("corrected_") and key.endswith("string_escaping") and isinstance(
        value, str
    )


def _is_non_replayable_stop_response(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return False
    if str(first_choice.get("finish_reason", "")).lower() != "stop":
        return False
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return False
    if message.get("tool_calls"):
        return False
    content = message.get("content")
    if not isinstance(content, str):
        return False
    normalized = content.strip()
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
            normalized = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, dict) and _is_non_replayable_stop_payload(parsed)


def _normalize_text(value: str) -> str:
    normalized = _SYSTEM_REMINDER_PATTERN.sub("", value).strip()
    for pattern, replacement in _VOLATILE_TEXT_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _normalize_value(value: object) -> object:
    if isinstance(value, str):
        return _normalize_text(value)
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    return value


def _normalize_tool_calls(tool_calls: object) -> list[object]:
    if not isinstance(tool_calls, list):
        return []
    normalized: list[object] = []
    for item in tool_calls:
        if not isinstance(item, dict):
            normalized.append(_normalize_value(item))
            continue
        function = item.get("function")
        payload: dict[str, object] = {
            "id": item.get("id"),
            "type": item.get("type"),
        }
        if isinstance(function, dict):
            payload["function"] = {
                "name": function.get("name"),
                "arguments": function.get("arguments"),
            }
        normalized.append(payload)
    return normalized


def _normalize_message_content(role: str, content: object) -> object:
    normalized = _normalize_value(content)
    if role == "system" and isinstance(normalized, str):
        if normalized.startswith("You are iFlow CLI, an interactive CLI agent"):
            return _IFLOW_SYSTEM_PROMPT_SENTINEL
        return normalized
    if role == "assistant" and isinstance(normalized, str):
        if "Thanks for the context" in normalized:
            return _IFLOW_CONTEXT_ACK_SENTINEL
        return normalized
    if role == "user" and isinstance(normalized, list) and len(normalized) == 1:
        item = normalized[0]
        if isinstance(item, dict) and item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str) and text.startswith("This is the iFlow CLI. We are setting up the context"):
                return [{"type": "text", "text": _IFLOW_CONTEXT_SENTINEL}]
    return normalized


def _normalize_message(message: object) -> dict[str, object]:
    if not isinstance(message, dict):
        return {"raw": _normalize_value(message)}
    role = str(message.get("role", ""))
    normalized: dict[str, object] = {"role": role}
    if "name" in message:
        normalized["name"] = message.get("name")
    if "tool_call_id" in message:
        normalized["tool_call_id"] = message.get("tool_call_id")
    if role in {"tool", "function"}:
        normalized["content"] = _TOOL_RESULT_SENTINEL
    else:
        normalized["content"] = _normalize_message_content(role, message.get("content"))
    tool_calls = _normalize_tool_calls(message.get("tool_calls"))
    if tool_calls:
        normalized["tool_calls"] = tool_calls
    return normalized


def _lookup_stats(path: str, payload: dict[str, Any]) -> tuple[str, int, int, int, int]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        messages = []
    normalized_messages = [_normalize_message(item) for item in messages]
    normalized = {
        "path": path,
        "messages": normalized_messages,
    }
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    request_hash = hashlib.sha256(canonical).hexdigest()
    assistant_message_count = 0
    tool_message_count = 0
    assistant_tool_call_count = 0
    for message in normalized_messages:
        role = str(message.get("role", ""))
        if role == "assistant":
            assistant_message_count += 1
            assistant_tool_call_count += len(message.get("tool_calls", []))
        elif role == "tool":
            tool_message_count += 1
    return (
        request_hash,
        len(normalized_messages),
        assistant_message_count,
        tool_message_count,
        assistant_tool_call_count,
    )


def _response_fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def parse_replay_trace(trace_path: Path) -> ParsedReplayTrace:
    resolved = trace_path.expanduser().resolve()
    responses: list[dict[str, Any]] = []
    malformed_lines: list[int] = []
    exchanges: list[ReplayExchange] = []
    pending_request: tuple[str, dict[str, Any]] | None = None
    fingerprints_by_key: dict[tuple[str, int, int, int, int, int], str] = {}
    for line_number, raw_line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines.append(line_number)
            continue
        if not isinstance(entry, dict):
            malformed_lines.append(line_number)
            continue
        entry_type = entry.get("type")
        if entry_type == "request":
            payload = _decode_json_payload(entry.get("data"))
            if payload is None:
                malformed_lines.append(line_number)
                continue
            pending_request = ("/v1/chat/completions", payload)
            continue
        if entry_type != "response":
            continue
        payload = _decode_json_payload(entry.get("data"))
        if payload is None:
            malformed_lines.append(line_number)
            continue
        if pending_request is None:
            if not _is_non_replayable_stop_response(payload):
                responses.append(payload)
            continue
        path, request = pending_request
        counts_toward_progress = not _is_non_replayable_stop_response(payload)
        if counts_toward_progress:
            responses.append(payload)
        lookup_key = _lookup_stats(path, request)
        fingerprint = _response_fingerprint(payload)
        previous = fingerprints_by_key.get(lookup_key)
        if previous is not None and previous != fingerprint:
            raise ValueError(
                f"trace {resolved} has ambiguous replay responses for request key={lookup_key[0][:12]}"
            )
        fingerprints_by_key[lookup_key] = fingerprint
        exchanges.append(
            ReplayExchange(
                path=path,
                request=request,
                response=payload,
                lookup_key=lookup_key,
                counts_toward_progress=counts_toward_progress,
            )
        )
        pending_request = None
    if not responses:
        raise ValueError(f"trace {resolved} did not contain any replayable responses")
    return ParsedReplayTrace(
        trace_path=resolved,
        responses=tuple(responses),
        malformed_lines=tuple(malformed_lines),
        exchanges=tuple(exchanges),
    )


class TraceReplayLLMState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        config = dict(llm_service_config or {})
        trace_path_value = config.get("trace_path")
        if not isinstance(trace_path_value, str) or not trace_path_value:
            raise ValueError("iflow_trace_replay requires llm_service_config.trace_path")
        raw_delay_ms = config.get("response_delay_ms", _DEFAULT_RESPONSE_DELAY_MS)
        try:
            response_delay_ms = int(raw_delay_ms)
        except (TypeError, ValueError):
            response_delay_ms = _DEFAULT_RESPONSE_DELAY_MS
        parsed = parse_replay_trace(Path(trace_path_value))
        if not parsed.exchanges:
            raise ValueError(f"trace {parsed.trace_path} did not contain any replayable request/response pairs")
        self._trace = parsed
        self._lock = threading.Lock()
        self._response_delay_ms = max(0, response_delay_ms)
        self._events: list[dict[str, Any]] = []
        self._matched_response_count = 0
        self._total_progress_responses = sum(1 for exchange in parsed.exchanges if exchange.counts_toward_progress)
        self._responses_by_key = {exchange.lookup_key: exchange for exchange in parsed.exchanges}
        self._hashes_by_shape: dict[tuple[int, int, int, int], list[str]] = {}
        for exchange in parsed.exchanges:
            self._hashes_by_shape.setdefault(exchange.lookup_key[1:], []).append(exchange.lookup_key[0])

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return handle_request(path=path, headers=headers, payload=payload, state=self)

    def checkpoint_metadata(self) -> dict[str, object]:
        with self._lock:
            return {"benchmark_replay_action_count": self._matched_response_count}

    def restore_from_checkpoint_metadata(self, metadata: dict[str, object]) -> None:
        _ = metadata

    def reset(self) -> None:
        with self._lock:
            self._matched_response_count = 0
            self._events.clear()

    def next_response(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        sandbox_id = _sandbox_id_from_request(headers, payload)
        lookup_key = _lookup_stats(path, payload)
        try:
            exchange = self._responses_by_key[lookup_key]
        except KeyError as exc:
            matching_hashes = self._hashes_by_shape.get(lookup_key[1:], [])
            logger.warning(
                "Replay lookup miss path=%s hash=%s messages=%d assistant_messages=%d tool_messages=%d assistant_tool_calls=%d shape_match_count=%d",
                path,
                lookup_key[0][:12],
                lookup_key[1],
                lookup_key[2],
                lookup_key[3],
                lookup_key[4],
                len(matching_hashes),
            )
            raise ValueError(
                "no replay response found for request "
                f"hash={lookup_key[0][:12]} messages={lookup_key[1]} tool_messages={lookup_key[3]}"
            ) from exc
        with self._lock:
            if exchange.counts_toward_progress:
                self._matched_response_count += 1
            matched_response_count = self._matched_response_count
            self._events.append(
                {
                    "event": "response",
                    "sandbox_id": sandbox_id,
                    "request_hash": lookup_key[0],
                    "matched_response_count": matched_response_count,
                    "counts_toward_progress": exchange.counts_toward_progress,
                }
            )
        if self._response_delay_ms > 0:
            time.sleep(self._response_delay_ms / 1000.0)
        return matched_response_count, copy.deepcopy(exchange.response)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            matched_response_count = self._matched_response_count
            return {
                "trace_path": str(self._trace.trace_path),
                "response_delay_ms": self._response_delay_ms,
                "total_responses": self._total_progress_responses,
                "matched_response_count": matched_response_count,
                "next_response_index": matched_response_count,
                "responses_served": matched_response_count,
                "is_complete": matched_response_count >= self._total_progress_responses,
                "malformed_line_count": len(self._trace.malformed_lines),
                "malformed_lines": list(self._trace.malformed_lines),
                "events": list(self._events),
            }


def handle_request(
    *,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    state: TraceReplayLLMState,
) -> dict[str, Any]:
    if path != "/v1/chat/completions":
        raise ValueError(f"unsupported path for iflow_trace_replay: {path}")
    _, response = state.next_response(path=path, headers=headers, payload=payload)
    return response

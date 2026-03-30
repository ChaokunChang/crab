from __future__ import annotations

import copy
import hashlib
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_RESPONSE_DELAY_MS = 250
_IFLOW_SYSTEM_PROMPT_SENTINEL = "<iflow-system-prompt>"
_IFLOW_CONTEXT_SENTINEL = "<iflow-context-bootstrap>"
_IFLOW_CONTEXT_ACK_SENTINEL = "<iflow-context-ack>"
_DUMMY_TOOL_RESPONSE_MODEL = "agent-cr-iflow-trace-replay"
_DUMMY_TOOL_COMMAND = 'sh -lc "echo hello world >> /dev/null"'
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
    (re.compile(
        r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})\b"), "<timestamp>"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "<date>"),
    (re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"), "<date>"),
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b"), "<time>"),
    (re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.IGNORECASE), "<uuid>"),
    (re.compile(r"https?://[^\s\"']+"), "<url>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),
    (re.compile(r"/tmp/[^\s\"']+"), "/tmp/<temp>"),
)
_SYSTEM_REMINDER_PATTERN = re.compile(
    r"<system-reminder>.*?</system-reminder>", re.DOTALL)
_SYSTEM_SETUP_SHELL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|[;&|]\s*|&&\s*|\|\|\s*)(?:apt|apt-get)\s+(?:update|install|remove|upgrade|dist-upgrade)\b"),
    re.compile(r"(?:^|[;&|]\s*|&&\s*|\|\|\s*)(?:pip|pip3|uv\s+pip)\s+(?:install|uninstall|remove|update|upgrade)\b"),
    re.compile(r"(?:^|[;&|]\s*|&&\s*|\|\|\s*)python(?:3)?\s+-m\s+(?:pip|ensurepip)\b"),
    re.compile(r"bootstrap\.pypa\.io/get-pip\.py"),
)

logger = logging.getLogger(__name__)

_IFLOW_CONTEXT_ACK_MESSAGES = {
    "Got it. Thanks for the context!",
    "Thanks for the context. Ready when you are.",
}


_RESPONSE_DELAY_POLICIES = {"fixed", "trace_replay"}


@dataclass(frozen=True)
class ReplayExchange:
    path: str
    request: dict[str, Any]
    response: dict[str, Any]
    response_index: int
    lookup_key: str
    request_timestamp: float | None = None
    response_timestamp: float | None = None

    @property
    def trace_delay_ms(self) -> float | None:
        if self.request_timestamp is not None and self.response_timestamp is not None:
            delay = (self.response_timestamp - self.request_timestamp) * 1000.0
            return max(0.0, delay)
        return None


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


def _normalize_text(value: str) -> str:
    normalized = _SYSTEM_REMINDER_PATTERN.sub("", value).strip()
    for pattern, replacement in _VOLATILE_TEXT_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    if normalized.startswith("```"):
        lines = normalized.splitlines()
        if len(lines) >= 3 and lines[0].strip().startswith("```") and lines[-1].strip() == "```":
            normalized = "\n".join(lines[1:-1]).strip()
    return normalized


def _normalize_text_relaxed(value: str) -> str:
    normalized = _normalize_text(value)
    return _collapse_escape_only_formatting(normalized)


def _collapse_escape_only_formatting(value: str) -> str:
    return (
        value.replace("\\\\r\\\\n", "\n")
        .replace("\\\\n", "\n")
        .replace("\\\\r", "\r")
        .replace("\\\\t", "\t")
        .replace('\\\\\\"', '"')
        .replace('\\"', '"')
        .replace("\\\\'", "'")
        .replace("\\'", "'")
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
    )


def _normalize_value(value: object, *, relaxed: bool = False) -> object:
    if isinstance(value, str):
        if relaxed:
            return _normalize_text_relaxed(value)
        return _normalize_text(value)
    if isinstance(value, dict):
        return {str(key): _normalize_value(item, relaxed=relaxed) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_normalize_value(item, relaxed=relaxed) for item in value]
    return value


def _normalize_message_content(role: str, content: object, *, relaxed: bool = False) -> object:
    normalized = _normalize_value(content, relaxed=relaxed)
    if role == "system" and isinstance(normalized, str):
        if normalized.startswith("You are iFlow CLI, an interactive CLI agent"):
            return _IFLOW_SYSTEM_PROMPT_SENTINEL
        return normalized
    if role == "assistant" and isinstance(normalized, str):
        if normalized in _IFLOW_CONTEXT_ACK_MESSAGES:
            return _IFLOW_CONTEXT_ACK_SENTINEL
        return normalized
    if role == "user" and isinstance(normalized, list) and len(normalized) >= 1:
        item = normalized[-1]
        if (isinstance(item, dict) and item.get("type") == "text"):
            value = item.get("text", _IFLOW_CONTEXT_SENTINEL)
            if isinstance(value, str):
                return value
            else:
                return _IFLOW_CONTEXT_SENTINEL
    return normalized


def _normalized_lookup_messages(payload: dict[str, Any], *, relaxed: bool = False) -> tuple[object, ...]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ()
    normalized_messages: list[object] = []
    last_user_msg = -1
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        if role == "user":
            last_user_msg = i
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", ""))
        if role not in {"user", "assistant"}:
            continue
        if role == "user" and i < last_user_msg:
            continue
        normalized_messages.append(
            {
                "role": role,
                "content": _normalize_message_content(role, message.get("content"), relaxed=relaxed),
            }
        )
    return tuple(normalized_messages)


def _lookup_stats(path: str, payload: dict[str, Any], *, relaxed: bool = False) -> tuple[str, int]:
    lookup_messages = _normalized_lookup_messages(payload, relaxed=relaxed)
    normalized = {
        "path": path,
        "messages": lookup_messages,
    }
    canonical = json.dumps(normalized, sort_keys=True, separators=(
        ",", ":"), ensure_ascii=False).encode("utf-8")
    request_hash = hashlib.sha256(canonical).hexdigest()
    assistant_message_count = sum(1 for message in lookup_messages if isinstance(
        message, dict) and message.get("role") == "assistant")
    return (request_hash, assistant_message_count)


def _response_fingerprint(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _tool_specs(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return []
    specs: list[tuple[str, dict[str, Any]]] = []
    for item in tools:
        if not isinstance(item, dict):
            continue
        function = item.get("function")
        if not isinstance(function, dict):
            continue
        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue
        parameters = function.get("parameters")
        specs.append(
            (name, parameters if isinstance(parameters, dict) else {}))
    return specs


def _tool_parameters_by_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    parameters_by_name: dict[str, dict[str, Any]] = {}
    for name, parameters in _tool_specs(payload):
        parameters_by_name.setdefault(name, parameters)
    return parameters_by_name


def _should_preserve_duplicate_shell_command(command: str) -> bool:
    normalized = command.strip()
    if not normalized:
        return False
    return any(pattern.search(normalized) for pattern in _SYSTEM_SETUP_SHELL_PATTERNS)


def _dummy_value_for_schema(*, name: str, schema: dict[str, Any]) -> object:
    const = schema.get("const")
    if const is not None:
        return const
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    for key in ("oneOf", "anyOf", "allOf"):
        variants = schema.get(key)
        if isinstance(variants, list):
            for variant in variants:
                if isinstance(variant, dict):
                    return _dummy_value_for_schema(name=name, schema=variant)
    normalized_name = name.lower()
    if normalized_name in {"command", "cmd", "script"} or normalized_name.endswith("_command"):
        return _DUMMY_TOOL_COMMAND
    if normalized_name in {"path", "file_path", "filepath", "filename"} or normalized_name.endswith("_path"):
        return "/dev/null"
    if normalized_name in {"dirname", "directory", "dir"}:
        return "."
    if normalized_name in {"url", "uri"}:
        return "http://127.0.0.1/"
    if normalized_name in {"content", "text", "body", "message", "note", "line", "query"}:
        return "agent-cr replay noop"
    value_type = schema.get("type")
    if value_type == "boolean":
        return False
    if value_type == "integer":
        return 0
    if value_type == "number":
        return 0
    if value_type == "array":
        return []
    if value_type == "object":
        return {}
    return ""


def _dummy_tool_input(tool_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    normalized_name = tool_name.lower()
    if normalized_name == "run_shell_command":
        return {"command": _DUMMY_TOOL_COMMAND}
    properties = parameters.get("properties")
    required = parameters.get("required")
    if not isinstance(properties, dict):
        return {}
    required_names = [str(item) for item in required] if isinstance(
        required, list) else []
    property_names = required_names or [
        str(name) for name in properties.keys()]
    payload: dict[str, Any] = {}
    for property_name in property_names:
        schema = properties.get(property_name)
        if isinstance(schema, dict):
            payload[property_name] = _dummy_value_for_schema(
                name=property_name, schema=schema)
    return payload


def _is_tool_call_response(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return False
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return False
    tool_calls = message.get("tool_calls")
    return isinstance(tool_calls, list) and len(tool_calls) > 0


def _should_preserve_duplicate_tool_response(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return False
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return False
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list) or not tool_calls:
        return False
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") != "run_shell_command":
            continue
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            continue
        try:
            parsed_arguments = json.loads(arguments)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed_arguments, dict):
            continue
        command = parsed_arguments.get("command")
        if isinstance(command, str) and _should_preserve_duplicate_shell_command(command):
            return True
    return False


def _build_dummy_tool_response(*, request_payload: dict[str, Any], original_response: dict[str, Any]) -> dict[str, Any] | None:
    choices = original_response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return None
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None
    if not isinstance(message.get("tool_calls"), list) or not message.get("tool_calls"):
        return None
    parameters_by_name = _tool_parameters_by_name(request_payload)
    if not parameters_by_name:
        return None
    response = copy.deepcopy(original_response)
    response["model"] = _DUMMY_TOOL_RESPONSE_MODEL
    response["id"] = f"chatcmpl-iflow-replay-dummy-{uuid.uuid4().hex[:8]}"
    response["created"] = int(time.time())
    response["usage"] = {"prompt_tokens": 1,
                         "completion_tokens": 1, "total_tokens": 2}
    response_choices = response.get("choices")
    assert isinstance(response_choices, list)
    response_choice = response_choices[0]
    assert isinstance(response_choice, dict)
    response_message = response_choice.get("message")
    assert isinstance(response_message, dict)
    response_tool_calls = response_message.get("tool_calls")
    assert isinstance(response_tool_calls, list)
    for tool_call in response_tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = function.get("name")
        if not isinstance(tool_name, str) or not tool_name:
            continue
        parameters = parameters_by_name.get(tool_name, {})
        tool_input = _dummy_tool_input(tool_name, parameters)
        function["arguments"] = json.dumps(tool_input, sort_keys=True)
    return response


def parse_replay_trace(trace_path: Path) -> ParsedReplayTrace:
    resolved = trace_path.expanduser().resolve()
    responses: list[dict[str, Any]] = []
    malformed_lines: list[int] = []
    exchanges: list[ReplayExchange] = []
    pending_request: tuple[str, dict[str, Any]] | None = None
    fingerprints_by_key: dict[str, str] = {}
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
            request_ts = entry.get("timestamp")
            pending_request = ("/v1/chat/completions", payload, request_ts if isinstance(request_ts, (int, float)) else None)
            continue
        if entry_type != "response":
            continue
        payload = _decode_json_payload(entry.get("data"))
        if payload is None:
            malformed_lines.append(line_number)
            continue
        if pending_request is None:
            responses.append(payload)
            continue
        path, request, request_ts = pending_request
        response_ts = entry.get("timestamp")
        response_ts = response_ts if isinstance(response_ts, (int, float)) else None
        responses.append(payload)
        lookup_key, _ = _lookup_stats(path, request)
        fingerprint = _response_fingerprint(payload)
        previous = fingerprints_by_key.get(lookup_key)
        if previous is not None and previous != fingerprint:
            raise ValueError(
                f"trace {resolved} has ambiguous replay responses for request key={lookup_key[:12]}"
            )
        fingerprints_by_key[lookup_key] = fingerprint
        exchanges.append(
            ReplayExchange(
                path=path,
                request=request,
                response=payload,
                response_index=len(exchanges),
                lookup_key=lookup_key,
                request_timestamp=request_ts,
                response_timestamp=response_ts,
            )
        )
        pending_request = None
    if not responses:
        raise ValueError(
            f"trace {resolved} did not contain any replayable responses")
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
            raise ValueError(
                "iflow_trace_replay requires llm_service_config.trace_path")
        raw_policy = str(config.get("response_delay_policy", "fixed")).strip().lower()
        if raw_policy not in _RESPONSE_DELAY_POLICIES:
            raise ValueError(
                f"response_delay_policy must be one of {sorted(_RESPONSE_DELAY_POLICIES)}, got {raw_policy!r}")
        raw_delay_ms = config.get(
            "response_delay_ms", _DEFAULT_RESPONSE_DELAY_MS)
        try:
            response_delay_ms = int(raw_delay_ms)
        except (TypeError, ValueError):
            response_delay_ms = _DEFAULT_RESPONSE_DELAY_MS
        try:
            scaling_factor = float(config.get("response_delay_scaling_factor", 1.0))
        except (TypeError, ValueError):
            scaling_factor = 1.0
        parsed = parse_replay_trace(Path(trace_path_value))
        if not parsed.exchanges:
            raise ValueError(
                f"trace {parsed.trace_path} did not contain any replayable request/response pairs")
        self._trace = parsed
        self._lock = threading.Lock()
        self._response_delay_policy = raw_policy
        self._response_delay_ms = max(0, response_delay_ms)
        self._response_delay_scaling_factor = max(0.0, scaling_factor)
        self._events: list[dict[str, Any]] = []
        self._consumed_response_count = 0
        self._duplicate_response_count = 0
        self._total_progress_responses = len(parsed.exchanges)
        self._trace_cursor = 0
        self._served_request_keys: set[str] = set()
        self._responses_by_key = {
            exchange.lookup_key: exchange for exchange in parsed.exchanges}
        self._responses_by_relaxed_key: dict[str, ReplayExchange] = {}
        self._ambiguous_relaxed_lookup_keys: set[str] = set()
        for exchange in parsed.exchanges:
            relaxed_key, _ = _lookup_stats(exchange.path, exchange.request, relaxed=True)
            if relaxed_key in self._ambiguous_relaxed_lookup_keys:
                continue
            existing = self._responses_by_relaxed_key.get(relaxed_key)
            if existing is None:
                self._responses_by_relaxed_key[relaxed_key] = exchange
                continue
            if existing.lookup_key == exchange.lookup_key:
                continue
            self._ambiguous_relaxed_lookup_keys.add(relaxed_key)
            self._responses_by_relaxed_key.pop(relaxed_key, None)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return handle_request(path=path, headers=headers, payload=payload, state=self)

    def reset(self) -> None:
        with self._lock:
            self._consumed_response_count = 0
            self._duplicate_response_count = 0
            self._trace_cursor = 0
            self._served_request_keys.clear()
            self._events.clear()

    def restore(self, *, consumed_response_count: int) -> None:
        with self._lock:
            restored_count = max(0, min(int(consumed_response_count), self._total_progress_responses))
            self._consumed_response_count = restored_count
            self._duplicate_response_count = 0
            self._trace_cursor = restored_count
            self._served_request_keys = {
                exchange.lookup_key for exchange in self._trace.exchanges[:restored_count]
            }
            self._events.append(
                {
                    "event": "restore",
                    "consumed_response_count": restored_count,
                }
            )

    def next_response(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        sandbox_id = _sandbox_id_from_request(headers, payload)
        lookup_key, assistant_message_count = _lookup_stats(path, payload)
        try:
            exchange = self._responses_by_key[lookup_key]
        except KeyError as exc:
            relaxed_lookup_key, _ = _lookup_stats(path, payload, relaxed=True)
            exchange = self._responses_by_relaxed_key.get(relaxed_lookup_key)
            if exchange is not None:
                lookup_key = exchange.lookup_key
                logger.info(
                    "Replay lookup relaxed-match path=%s sandbox_id=%s strict_hash=%s relaxed_hash=%s assistant_messages=%d",
                    path,
                    sandbox_id,
                    _lookup_stats(path, payload)[0][:12],
                    relaxed_lookup_key[:12],
                    assistant_message_count,
                )
            else:
                if self._trace_cursor >= self._total_progress_responses:
                    logger.warning(
                        "Replay lookup miss path=%s sandbox_id=%s hash=%s assistant_messages=%d",
                        path,
                        sandbox_id,
                        lookup_key[:12],
                        assistant_message_count,
                    )
                    logger.warning(f"{sandbox_id=} Replay lookup miss: {self._consumed_response_count=}, {payload=}")
                    raise ValueError(
                        "no replay response found for request "
                        f"hash={lookup_key[:12]} assistant_messages={assistant_message_count}"
                    ) from exc
                exchange = self._trace.exchanges[self._trace_cursor]
                logger.info(
                    "Replay cursor-fallback response path=%s sandbox_id=%s requested_hash=%s assistant_messages=%d response_index=%d consumed_response_count=%d",
                    path,
                    sandbox_id,
                    lookup_key[:12],
                    assistant_message_count,
                    exchange.response_index,
                    self._consumed_response_count,
                )
        self._trace_cursor = exchange.response_index + 1
        with self._lock:
            is_duplicate = lookup_key in self._served_request_keys
            response_kind = "replay"
            if is_duplicate:
                consumed_response_count = self._consumed_response_count
                self._duplicate_response_count += 1
                if _is_tool_call_response(exchange.response):
                    if _should_preserve_duplicate_tool_response(exchange.response):
                        response = exchange.response
                        response_kind = "duplicate_original"
                    else:
                        response = _build_dummy_tool_response(
                            request_payload=payload, original_response=exchange.response)
                    if response is None:
                        logger.warning(
                            "Replay duplicate request fell back to original response because no tool schema was advertised "
                            "path=%s hash=%s",
                            path,
                            lookup_key[:12],
                        )
                        response = exchange.response
                        response_kind = "duplicate_original"
                    else:
                        response_kind = "duplicate_dummy_tool"
                else:
                    response = exchange.response
                    response_kind = "duplicate_original"
            else:
                self._served_request_keys.add(exchange.lookup_key)
                self._served_request_keys.add(lookup_key)
                self._consumed_response_count += 1
                consumed_response_count = self._consumed_response_count
                response = exchange.response
            self._events.append(
                {
                    "event": "response",
                    "sandbox_id": sandbox_id,
                    "request_hash": lookup_key,
                    "consumed_response_count": consumed_response_count,
                    "response_kind": response_kind,
                    "response_index": exchange.response_index,
                    "lookup_key_matched": True,
                }
            )
        effective_delay_ms = self._compute_delay_ms(exchange)
        if effective_delay_ms > 0:
            time.sleep(effective_delay_ms / 1000.0)
        return consumed_response_count, copy.deepcopy(response)

    def _compute_delay_ms(self, exchange: ReplayExchange) -> float:
        if self._response_delay_policy == "trace_replay":
            trace_delay = exchange.trace_delay_ms
            if trace_delay is not None:
                return trace_delay * self._response_delay_scaling_factor
            return float(self._response_delay_ms)
        return float(self._response_delay_ms)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        with self._lock:
            consumed_response_count = self._consumed_response_count
            payload = {
                "trace_path": str(self._trace.trace_path),
                "response_delay_policy": self._response_delay_policy,
                "response_delay_ms": self._response_delay_ms,
                "response_delay_scaling_factor": self._response_delay_scaling_factor,
                "total_responses": self._total_progress_responses,
                "consumed_response_count": consumed_response_count,
                "duplicate_response_count": self._duplicate_response_count,
                "trace_cursor": self._trace_cursor,
                "is_complete": consumed_response_count >= self._total_progress_responses,
                "malformed_line_count": len(self._trace.malformed_lines),
                "malformed_lines": list(self._trace.malformed_lines),
            }
            if include_events:
                payload["events"] = list(self._events)
            return payload


def handle_request(
    *,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    state: TraceReplayLLMState,
) -> dict[str, Any]:
    if path != "/v1/chat/completions":
        raise ValueError(f"unsupported path for iflow_trace_replay: {path}")
    _, response = state.next_response(
        path=path, headers=headers, payload=payload)
    return response

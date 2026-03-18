from __future__ import annotations

import copy
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_STATE_METADATA_KEY = "llm_service_state"
_STATE_SERVICE_TYPE = "iflow_trace_replay"
_DEFAULT_RESPONSE_DELAY_MS = 250
_CAPTURES_INFLIGHT_LLM = "captures_inflight_llm"


@dataclass(frozen=True)
class ParsedReplayTrace:
    trace_path: Path
    responses: tuple[dict[str, Any], ...]
    malformed_lines: tuple[int, ...]


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


def parse_replay_trace(trace_path: Path) -> ParsedReplayTrace:
    resolved = trace_path.expanduser().resolve()
    responses: list[dict[str, Any]] = []
    malformed_lines: list[int] = []
    for line_number, raw_line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            malformed_lines.append(line_number)
            continue
        if not isinstance(entry, dict) or entry.get("type") != "response":
            continue
        payload = _decode_json_payload(entry.get("data"))
        if payload is None:
            malformed_lines.append(line_number)
            continue
        if _is_non_replayable_stop_response(payload):
            continue
        responses.append(payload)
    if not responses:
        raise ValueError(f"trace {resolved} did not contain any replayable responses")
    return ParsedReplayTrace(
        trace_path=resolved,
        responses=tuple(responses),
        malformed_lines=tuple(malformed_lines),
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
        self._trace = parsed
        self._lock = threading.Lock()
        self._next_response_index = 0
        self._events: list[dict[str, Any]] = []
        self._response_delay_ms = max(0, response_delay_ms)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return handle_request(path=path, headers=headers, payload=payload, state=self)

    def checkpoint_metadata(self) -> dict[str, object]:
        with self._lock:
            return {
                _STATE_METADATA_KEY: {
                    "service_type": _STATE_SERVICE_TYPE,
                    "next_response_index": self._next_response_index,
                }
            }

    def restore_from_checkpoint_metadata(self, metadata: dict[str, object]) -> None:
        raw_state = metadata.get(_STATE_METADATA_KEY)
        if not isinstance(raw_state, dict):
            return
        if raw_state.get("service_type") != _STATE_SERVICE_TYPE:
            return
        raw_index = raw_state.get("next_response_index", 0)
        try:
            next_response_index = int(raw_index)
        except (TypeError, ValueError):
            next_response_index = 0
        # The replay state is captured after the current response is selected.
        # When the checkpoint also captured an in-flight request, the restored sandbox
        # re-issues that request, so we need to rewind one turn and replay it again.
        if bool(metadata.get(_CAPTURES_INFLIGHT_LLM, False)) and next_response_index > 0:
            next_response_index -= 1
        with self._lock:
            self._next_response_index = max(0, min(next_response_index, len(self._trace.responses)))

    def reset(self) -> None:
        with self._lock:
            self._next_response_index = 0

    def next_response(self, *, headers: dict[str, str], payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        sandbox_id = _sandbox_id_from_request(headers, payload)
        with self._lock:
            response_index = min(self._next_response_index, len(self._trace.responses) - 1)
            if self._next_response_index < len(self._trace.responses):
                self._next_response_index += 1
            self._events.append(
                {
                    "event": "response",
                    "sandbox_id": sandbox_id,
                    "response_index": response_index,
                    "next_response_index": self._next_response_index,
                }
            )
            response = copy.deepcopy(self._trace.responses[response_index])
        if self._response_delay_ms > 0:
            time.sleep(self._response_delay_ms / 1000.0)
        return response_index, response

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "trace_path": str(self._trace.trace_path),
                "response_delay_ms": self._response_delay_ms,
                "total_responses": len(self._trace.responses),
                "next_response_index": self._next_response_index,
                "responses_served": min(self._next_response_index, len(self._trace.responses)),
                "is_complete": self._next_response_index >= len(self._trace.responses),
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
    _, response = state.next_response(headers=headers, payload=payload)
    return response

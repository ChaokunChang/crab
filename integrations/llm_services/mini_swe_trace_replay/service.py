from __future__ import annotations

import copy
import json
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_RESPONSE_DELAY_MS = 0
_RESPONSE_DELAY_POLICIES = {"fixed", "trace_replay"}


@dataclass(frozen=True)
class ParsedMiniSWETrace:
    trace_path: Path
    assistant_messages: tuple[dict[str, Any], ...]
    assistant_trace_delays_ms: tuple[float | None, ...]


def parse_replay_trace(trace_path: Path) -> ParsedMiniSWETrace:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    messages = payload.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError(f"mini-swe trace {trace_path} is missing a messages list")
    assistant_messages: list[dict[str, Any]] = []
    assistant_trace_delays_ms: list[float | None] = []
    previous_timestamp: float | None = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        timestamp = _message_timestamp(message)
        if message.get("role") != "assistant":
            if timestamp is not None:
                previous_timestamp = timestamp
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            if timestamp is not None:
                previous_timestamp = timestamp
            continue
        trace_delay_ms: float | None = None
        if timestamp is not None and previous_timestamp is not None:
            trace_delay_ms = max(0.0, (timestamp - previous_timestamp) * 1000.0)
        assistant_messages.append(copy.deepcopy(message))
        assistant_trace_delays_ms.append(trace_delay_ms)
        if timestamp is not None:
            previous_timestamp = timestamp
    if not assistant_messages:
        raise ValueError(f"mini-swe trace {trace_path} did not contain any assistant action messages")
    return ParsedMiniSWETrace(
        trace_path=trace_path,
        assistant_messages=tuple(assistant_messages),
        assistant_trace_delays_ms=tuple(assistant_trace_delays_ms),
    )


def _message_timestamp(message: dict[str, Any]) -> float | None:
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return None
    timestamp = extra.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return None
    return float(timestamp)


class TraceReplayLLMState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        config = dict(llm_service_config or {})
        raw_trace_path = config.get("trace_path")
        if not isinstance(raw_trace_path, str) or not raw_trace_path.strip():
            raise ValueError("mini_swe_trace_replay requires llm_service_config.trace_path")
        self._parsed = parse_replay_trace(Path(raw_trace_path).expanduser().resolve())
        raw_policy = str(config.get("response_delay_policy", "fixed")).strip().lower()
        if raw_policy not in _RESPONSE_DELAY_POLICIES:
            raise ValueError(
                f"response_delay_policy must be one of {sorted(_RESPONSE_DELAY_POLICIES)}, got {raw_policy!r}"
            )
        raw_delay = config.get("response_delay_ms", _DEFAULT_RESPONSE_DELAY_MS)
        try:
            self._response_delay_ms = max(0, int(raw_delay))
        except (TypeError, ValueError):
            self._response_delay_ms = _DEFAULT_RESPONSE_DELAY_MS
        try:
            self._response_delay_scaling_factor = max(0.0, float(config.get("response_delay_scaling_factor", 1.0)))
        except (TypeError, ValueError):
            self._response_delay_scaling_factor = 1.0
        self._response_delay_policy = raw_policy
        self._lock = threading.Lock()
        self._trace_cursor = 0

    def reset(self) -> None:
        with self._lock:
            self._trace_cursor = 0

    def restore(self, *, consumed_response_count: int) -> None:
        # Unlike iflow trace replay, mini_swe keeps the agent process and its
        # message history alive across restore. The sandbox catches up by
        # replaying already-issued shell commands locally, without re-querying
        # the LLM. Rewinding the replay cursor here would make the next real LLM
        # request return a stale assistant response.
        _ = consumed_response_count
        return

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            trace_cursor = self._trace_cursor
        return {
            "trace_path": str(self._parsed.trace_path),
            "response_delay_policy": self._response_delay_policy,
            "response_delay_ms": self._response_delay_ms,
            "response_delay_scaling_factor": self._response_delay_scaling_factor,
            "trace_cursor": trace_cursor,
            "total_responses": len(self._parsed.assistant_messages),
            "is_complete": trace_cursor >= len(self._parsed.assistant_messages),
        }

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        _ = (path, headers, payload)
        with self._lock:
            if self._trace_cursor >= len(self._parsed.assistant_messages):
                raise RuntimeError("mini_swe_trace_replay exhausted all assistant responses")
            trace_index = self._trace_cursor
            message = copy.deepcopy(self._parsed.assistant_messages[trace_index])
            trace_delay_ms = self._parsed.assistant_trace_delays_ms[trace_index]
            self._trace_cursor += 1
        effective_delay_ms = self._effective_delay_ms(trace_delay_ms)
        if effective_delay_ms > 0:
            time.sleep(effective_delay_ms / 1000.0)
        return _completion_response(message, trace_index=trace_index)

    def _effective_delay_ms(self, trace_delay_ms: float | None) -> float:
        if self._response_delay_policy == "trace_replay":
            if trace_delay_ms is not None:
                return trace_delay_ms * self._response_delay_scaling_factor
            return float(self._response_delay_ms)
        return float(self._response_delay_ms)


def _completion_response(message: dict[str, Any], *, trace_index: int) -> dict[str, Any]:
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    response_message: dict[str, Any] = {
        "role": "assistant",
        "content": "" if not isinstance(content, str) else content,
    }
    if isinstance(reasoning_content, str) and reasoning_content:
        response_message["reasoning_content"] = reasoning_content
    return {
        "id": f"chatcmpl-mini-swe-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "agent-cr-mini-swe-trace-replay",
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        "trace_index": trace_index,
    }

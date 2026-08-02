from __future__ import annotations

import copy
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

_DEFAULT_RESPONSE_DELAY_MS = 0
_DEFAULT_MINIMAL_DELAY_MS = 0.0
_DEFAULT_MAXIMAL_DELAY_MS = 1_000_000_000.0
_RESPONSE_DELAY_POLICIES = {"fixed", "trace_replay"}

_BASH_COMMAND_TOOL_NAME = "bash_command"
_TASK_COMPLETE_TOOL_NAME = "mark_task_complete"
_PLAN_PREFIX_RE = re.compile(r"\n\s*Plan\s*:\s*", re.IGNORECASE)


def _config_with_legacy_alias(
    config: dict[str, object],
    key: str,
    legacy_key: str,
    default: object,
) -> object:
    if key in config:
        return config[key]
    if legacy_key in config:
        return config[legacy_key]
    return default


@dataclass(frozen=True)
class ParsedTerminusTrace:
    trace_path: Path
    model_name: str
    initial_user_prompt: str
    assistant_messages: tuple[dict[str, Any], ...]
    assistant_trace_delays_ms: tuple[float | None, ...]


def _parse_iso_timestamp(value: str) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _split_analysis_plan(message: str) -> tuple[str, str]:
    match = _PLAN_PREFIX_RE.search(message)
    if match is None:
        return message.strip(), ""
    analysis = message[: match.start()].strip()
    plan = message[match.end():].strip()
    if analysis.lower().startswith("analysis:"):
        analysis = analysis.split(":", 1)[1].lstrip()
    return analysis, plan


def _build_terminus_response_json(step: dict[str, Any]) -> tuple[str, bool]:
    raw_message = step.get("message")
    message = raw_message if isinstance(raw_message, str) else ""
    analysis, plan = _split_analysis_plan(message)
    commands: list[dict[str, Any]] = []
    task_complete = False
    tool_calls = step.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = tool_call.get("function_name")
            arguments = tool_call.get("arguments")
            if name == _BASH_COMMAND_TOOL_NAME and isinstance(arguments, dict):
                keystrokes = arguments.get("keystrokes", "")
                duration = arguments.get("duration", 1.0)
                if not isinstance(keystrokes, str):
                    continue
                try:
                    duration_value = float(duration)
                except (TypeError, ValueError):
                    duration_value = 1.0
                commands.append({"keystrokes": keystrokes, "duration": duration_value})
            elif name == _TASK_COMPLETE_TOOL_NAME:
                task_complete = True
    payload: dict[str, Any] = {
        "analysis": analysis,
        "plan": plan,
        "commands": commands,
    }
    if task_complete:
        payload["task_complete"] = True
    return json.dumps(payload, ensure_ascii=False), task_complete


def _trace_model_name(payload: dict[str, Any]) -> str:
    agent = payload.get("agent")
    if isinstance(agent, dict):
        name = agent.get("model_name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return "crab-terminus-trace-replay"


def parse_replay_trace(trace_path: Path) -> ParsedTerminusTrace:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"terminus trace {trace_path} must decode to an object")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"terminus trace {trace_path} is missing a non-empty steps list")
    initial_prompt: str | None = None
    assistant_messages: list[dict[str, Any]] = []
    assistant_delays_ms: list[float | None] = []
    previous_timestamp: float | None = None
    for step in steps:
        if not isinstance(step, dict):
            continue
        timestamp = _parse_iso_timestamp(step.get("timestamp", ""))
        source = str(step.get("source", "")).lower()
        if source == "user" and initial_prompt is None:
            raw_message = step.get("message")
            if isinstance(raw_message, str) and raw_message.strip():
                initial_prompt = raw_message
            if timestamp is not None:
                previous_timestamp = timestamp
            continue
        if source != "agent":
            if timestamp is not None:
                previous_timestamp = timestamp
            continue
        content, _ = _build_terminus_response_json(step)
        delay_ms: float | None = None
        if timestamp is not None and previous_timestamp is not None:
            delay_ms = max(0.0, (timestamp - previous_timestamp) * 1000.0)
        assistant_messages.append({"role": "assistant", "content": content})
        assistant_delays_ms.append(delay_ms)
        if timestamp is not None:
            previous_timestamp = timestamp
    if initial_prompt is None:
        raise ValueError(f"terminus trace {trace_path} is missing the initial user prompt step")
    if not assistant_messages:
        raise ValueError(f"terminus trace {trace_path} has no replayable assistant turns")
    return ParsedTerminusTrace(
        trace_path=trace_path,
        model_name=_trace_model_name(payload),
        initial_user_prompt=initial_prompt,
        assistant_messages=tuple(assistant_messages),
        assistant_trace_delays_ms=tuple(assistant_delays_ms),
    )


class TraceReplayLLMState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        config = dict(llm_service_config or {})
        raw_trace_path = config.get("trace_path")
        if not isinstance(raw_trace_path, str) or not raw_trace_path.strip():
            raise ValueError("terminus_trace_replay requires llm_service_config.trace_path")
        self._parsed = parse_replay_trace(Path(raw_trace_path).expanduser().resolve())
        raw_policy = str(config.get("response_delay_policy", "fixed")).strip().lower()
        if raw_policy not in _RESPONSE_DELAY_POLICIES:
            raise ValueError(
                f"response_delay_policy must be one of {sorted(_RESPONSE_DELAY_POLICIES)}, got {raw_policy!r}"
            )
        try:
            self._response_delay_ms = max(
                0, int(config.get("response_delay_ms", _DEFAULT_RESPONSE_DELAY_MS))
            )
        except (TypeError, ValueError):
            self._response_delay_ms = _DEFAULT_RESPONSE_DELAY_MS
        try:
            self._response_delay_scaling_factor = max(
                0.0, float(config.get("response_delay_scaling_factor", 1.0))
            )
        except (TypeError, ValueError):
            self._response_delay_scaling_factor = 1.0
        try:
            self._minimal_delay_ms = max(
                0.0,
                float(
                    _config_with_legacy_alias(
                        config, "minimal_delay", "minimal-delay", _DEFAULT_MINIMAL_DELAY_MS
                    )
                ),
            )
        except (TypeError, ValueError):
            self._minimal_delay_ms = _DEFAULT_MINIMAL_DELAY_MS
        try:
            self._maximal_delay_ms = max(
                0.0,
                float(
                    _config_with_legacy_alias(
                        config, "maximal_delay", "maximal-delay", _DEFAULT_MAXIMAL_DELAY_MS
                    )
                ),
            )
        except (TypeError, ValueError):
            self._maximal_delay_ms = _DEFAULT_MAXIMAL_DELAY_MS
        if self._maximal_delay_ms < self._minimal_delay_ms:
            raise ValueError(
                "maximal_delay must be greater than or equal to minimal_delay, "
                f"got {self._maximal_delay_ms!r} < {self._minimal_delay_ms!r}"
            )
        self._response_delay_policy = raw_policy
        self._lock = threading.Lock()
        self._trace_cursor = 0

    @property
    def initial_user_prompt(self) -> str:
        return self._parsed.initial_user_prompt

    def reset(self) -> None:
        with self._lock:
            self._trace_cursor = 0

    def restore(self, *, consumed_response_count: int) -> None:
        # Like mini_swe, the terminus agent keeps its full chat history alive across
        # restore. The sandbox replays already-issued shell commands locally, without
        # rewinding the LLM cursor.
        _ = consumed_response_count
        return

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        _ = include_events
        with self._lock:
            cursor = self._trace_cursor
        return {
            "trace_path": str(self._parsed.trace_path),
            "model_name": self._parsed.model_name,
            "response_delay_policy": self._response_delay_policy,
            "response_delay_ms": self._response_delay_ms,
            "response_delay_scaling_factor": self._response_delay_scaling_factor,
            "minimal_delay": self._minimal_delay_ms,
            "maximal_delay": self._maximal_delay_ms,
            "trace_cursor": cursor,
            "total_responses": len(self._parsed.assistant_messages),
            "is_complete": cursor >= len(self._parsed.assistant_messages),
            "initial_user_prompt": self._parsed.initial_user_prompt,
        }

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        _ = (path, headers, payload)
        with self._lock:
            if self._trace_cursor >= len(self._parsed.assistant_messages):
                raise RuntimeError("terminus_trace_replay exhausted all assistant responses")
            trace_index = self._trace_cursor
            message = copy.deepcopy(self._parsed.assistant_messages[trace_index])
            trace_delay_ms = self._parsed.assistant_trace_delays_ms[trace_index]
            self._trace_cursor += 1
        effective_delay_ms = self._effective_delay_ms(trace_delay_ms)
        if effective_delay_ms > 0:
            time.sleep(effective_delay_ms / 1000.0)
        return _completion_response(
            message,
            trace_index=trace_index,
            model_name=self._parsed.model_name,
        )

    def _effective_delay_ms(self, trace_delay_ms: float | None) -> float:
        if self._response_delay_policy == "trace_replay":
            if trace_delay_ms is not None:
                delay_ms = trace_delay_ms * self._response_delay_scaling_factor
            else:
                delay_ms = float(self._response_delay_ms)
        else:
            delay_ms = float(self._response_delay_ms)
        return min(self._maximal_delay_ms, max(self._minimal_delay_ms, delay_ms))


def _completion_response(
    message: dict[str, Any],
    *,
    trace_index: int,
    model_name: str,
) -> dict[str, Any]:
    content = message.get("content")
    response_message: dict[str, Any] = {
        "role": "assistant",
        "content": "" if not isinstance(content, str) else content,
    }
    return {
        "id": f"chatcmpl-terminus-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": response_message,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        "trace_index": trace_index,
    }

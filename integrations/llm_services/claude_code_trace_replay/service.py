from __future__ import annotations

import base64
import copy
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .request_classification import (
    DEFAULT_REPLAY_MODEL_NAME,
    REQUEST_KIND_COUNT_TOKENS,
    REQUEST_KIND_HELPER,
    REQUEST_KIND_MAIN_LOOP,
    REQUEST_KIND_OTHER,
    classify_replay_request,
    is_helper_model_request,
    normalize_model_family,
)

logger = logging.getLogger(__name__)


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

_DEFAULT_MODEL_NAME = DEFAULT_REPLAY_MODEL_NAME
_DEFAULT_RESPONSE_DELAY_MS = 0
_DEFAULT_MINIMAL_DELAY_MS = 0.0
_DEFAULT_MAXIMAL_DELAY_MS = 1_000_000_000.0
_NOOP_BASH_TIMEOUT_MS = 1_000
_RESPONSE_DELAY_POLICIES = {"fixed", "trace_replay"}
_PLACEHOLDER_RE = re.compile(r"^\$[0-9A-Fa-f]{2,4}$")
_BACKGROUND_TASK_ID_RE = re.compile(r"Command running in background with ID:\s*([A-Za-z0-9_-]+)")
_MAILMAN_RELAY_DOMAINS_LINE_RE = re.compile(
    r"(?m)^\s*relay_domains = hash:/var/lib/mailman3/data/postfix_domains\s*\n?"
)

# Claude Code only exposes these tools to the Anthropic Messages API.
# These tools are handled entirely client-side by Claude Code. We skip them when
# replaying API responses, but they should not make a trace non-replayable in
# strict mode by themselves.
_SKIP_TOOLS = {
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
}

_STRICT_NON_REPLAYABLE_SKIP_TOOLS = {
    "Task",
    "TaskOutput",
    "TaskStop",
}


@dataclass(frozen=True)
class ParsedClaudeCodeTrace:
    trace_path: Path
    model_name: str
    agent_steps: tuple[dict[str, Any], ...]
    agent_step_delays_ms: tuple[float | None, ...]


@dataclass(frozen=True)
class ReplayRequestContext:
    recovered_git_commit_hash: str | None = None
    background_task_aliases: dict[str, str] = field(default_factory=dict)
    filename_aliases: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedClaudeCodeSidechain:
    trace_path: Path
    agent_id: str
    initial_prompt: str
    model_name: str
    agent_steps: tuple[dict[str, Any], ...]
    agent_step_delays_ms: tuple[float | None, ...]


def load_trace_payload(trace_path: Path) -> dict[str, Any]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"claude-code trace {trace_path} must decode to an object")
    return payload


def strict_replayability_errors(trace_path: Path) -> tuple[str, ...]:
    payload = load_trace_payload(trace_path)
    errors: list[str] = []
    for step_index, step in enumerate(_iter_agent_steps(payload)):
        for tool in _extract_tool_entries(step):
            errors.extend(_tool_strict_errors(tool, step_index=step_index))
    return tuple(errors)


def is_strictly_replayable_trace(trace_path: Path) -> bool:
    return not strict_replayability_errors(trace_path)


def parse_replay_trace(trace_path: Path) -> ParsedClaudeCodeTrace:
    payload = load_trace_payload(trace_path)
    merged_turns = _merge_agent_steps(payload)
    if not merged_turns:
        raise ValueError(f"claude-code trace {trace_path} has no replayable agent turns")
    _mark_detached_background_tools(merged_turns)
    return ParsedClaudeCodeTrace(
        trace_path=trace_path,
        model_name=_trace_model_name(payload),
        agent_steps=tuple(merged_turns),
        agent_step_delays_ms=tuple(_compute_turn_delays_ms(merged_turns)),
    )


class TraceReplayLLMState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        config = dict(llm_service_config or {})
        raw_trace_path = config.get("trace_path")
        if not isinstance(raw_trace_path, str) or not raw_trace_path.strip():
            raise ValueError("claude_code_trace_replay requires llm_service_config.trace_path")
        trace_path = Path(raw_trace_path).expanduser().resolve()
        self._parsed = parse_replay_trace(trace_path)
        self._sidechains_by_agent_id, self._sidechains_by_prompt = _load_task_sidechains(trace_path)
        self._background_task_ids_by_tool_use_id = _recorded_background_task_ids_by_tool_use_id(
            self._parsed.agent_steps,
        )
        self._recorded_observations_by_tool_use_id = _recorded_observations_by_tool_use_id(
            self._parsed.agent_steps,
        )
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
            self._response_delay_scaling_factor = max(
                0.0,
                float(config.get("response_delay_scaling_factor", 1.0)),
            )
        except (TypeError, ValueError):
            self._response_delay_scaling_factor = 1.0
        try:
            self._minimal_delay_ms = max(
                0.0,
                float(
                    _config_with_legacy_alias(
                        config,
                        "minimal_delay",
                        "minimal-delay",
                        _DEFAULT_MINIMAL_DELAY_MS,
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
                        config,
                        "maximal_delay",
                        "maximal-delay",
                        _DEFAULT_MAXIMAL_DELAY_MS,
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
        self._duplicate_response_count = 0
        self._main_loop_request_count = 0
        self._helper_request_count = 0
        self._count_tokens_request_count = 0
        self._other_request_count = 0
        self._catchup_restore_cursor: int | None = None
        self._sticky_recovered_git_commit_hash: str | None = None
        self._sticky_background_task_aliases: dict[str, str] = {}
        self._sticky_filename_aliases: dict[str, str] = {}

    def reset(self) -> None:
        with self._lock:
            self._trace_cursor = 0
            self._duplicate_response_count = 0
            self._main_loop_request_count = 0
            self._helper_request_count = 0
            self._count_tokens_request_count = 0
            self._other_request_count = 0
            self._catchup_restore_cursor = None
            self._sticky_recovered_git_commit_hash = None
            self._sticky_background_task_aliases = {}
            self._sticky_filename_aliases = {}

    def restore(self, *, consumed_response_count: int) -> None:
        with self._lock:
            self._trace_cursor = max(0, int(consumed_response_count))
            self._duplicate_response_count = 0
            self._catchup_restore_cursor = self._trace_cursor

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        _ = include_events
        with self._lock:
            trace_cursor = self._trace_cursor
            duplicate_response_count = self._duplicate_response_count
            main_loop_request_count = self._main_loop_request_count
            helper_request_count = self._helper_request_count
            count_tokens_request_count = self._count_tokens_request_count
            other_request_count = self._other_request_count
        return {
            "trace_path": str(self._parsed.trace_path),
            "response_delay_policy": self._response_delay_policy,
            "response_delay_ms": self._response_delay_ms,
            "response_delay_scaling_factor": self._response_delay_scaling_factor,
            "minimal_delay": self._minimal_delay_ms,
            "maximal_delay": self._maximal_delay_ms,
            # trace_cursor tracks committed replay progress for scheduler-facing
            # Claude Code main-loop turns only. Helper and count-tokens traffic
            # are intentionally excluded because they are auxiliary requests.
            "trace_cursor": trace_cursor,
            "total_responses": len(self._parsed.agent_steps),
            "is_complete": trace_cursor >= len(self._parsed.agent_steps),
            "sidechain_count": len(self._sidechains_by_agent_id),
            "duplicate_response_count": duplicate_response_count,
            "main_loop_request_count": main_loop_request_count,
            "helper_request_count": helper_request_count,
            "count_tokens_request_count": count_tokens_request_count,
            "other_request_count": other_request_count,
        }

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        _ = headers
        is_streaming = bool(payload.get("stream", False))
        requested_model = _coerce_string(payload.get("model"))
        request_kind = classify_replay_request(
            path=path,
            requested_model=requested_model,
            replay_model=self._parsed.model_name,
        )
        if request_kind == REQUEST_KIND_COUNT_TOKENS:
            with self._lock:
                self._count_tokens_request_count += 1
            return _anthropic_count_tokens_response(payload)
        if request_kind == REQUEST_KIND_OTHER:
            with self._lock:
                self._other_request_count += 1
            logger.warning(
                "Ignoring non-replay Claude Code request path=%s model=%s trace_path=%s",
                path,
                requested_model,
                self._parsed.trace_path,
            )
            return _end_turn_response(
                model_name=requested_model or self._parsed.model_name,
                is_streaming=is_streaming,
            )
        request_context = _request_context_from_payload(
            payload,
            background_task_ids_by_tool_use_id=self._background_task_ids_by_tool_use_id,
            recorded_observations_by_tool_use_id=self._recorded_observations_by_tool_use_id,
        )
        if request_kind == REQUEST_KIND_HELPER:
            with self._lock:
                self._helper_request_count += 1
            sidechain = self._resolve_sidechain_for_request(payload)
            if sidechain is not None:
                return self._handle_sidechain_request(
                    sidechain,
                    payload=payload,
                    request_context=request_context,
                    is_streaming=is_streaming,
                )
            return _helper_model_response(model_name=requested_model, is_streaming=is_streaming)
        messages = payload.get("messages")
        has_request_history = isinstance(messages, list)
        requested_assistant_count = _assistant_message_count_from_request(payload)
        with self._lock:
            self._main_loop_request_count += 1
            if request_context.recovered_git_commit_hash is not None:
                self._sticky_recovered_git_commit_hash = request_context.recovered_git_commit_hash
            self._sticky_background_task_aliases.update(request_context.background_task_aliases)
            self._sticky_filename_aliases.update(request_context.filename_aliases)
            request_context = ReplayRequestContext(
                recovered_git_commit_hash=(
                    request_context.recovered_git_commit_hash
                    or self._sticky_recovered_git_commit_hash
                ),
                background_task_aliases={
                    **self._sticky_background_task_aliases,
                    **request_context.background_task_aliases,
                },
                filename_aliases={
                    **self._sticky_filename_aliases,
                    **request_context.filename_aliases,
                },
            )
            catchup_restore_cursor = self._catchup_restore_cursor
            if (
                catchup_restore_cursor is not None
                and has_request_history
                and requested_assistant_count < self._trace_cursor
            ):
                if requested_assistant_count >= len(self._parsed.agent_steps):
                    return _end_turn_response(model_name=self._parsed.model_name, is_streaming=is_streaming)
                step = copy.deepcopy(self._parsed.agent_steps[requested_assistant_count])
                self._duplicate_response_count += 1
                is_last = False
                duplicate_step = True
            else:
                duplicate_step = False
                if self._trace_cursor >= len(self._parsed.agent_steps):
                    return _end_turn_response(model_name=self._parsed.model_name, is_streaming=is_streaming)
                trace_index = self._trace_cursor
                step = copy.deepcopy(self._parsed.agent_steps[trace_index])
                trace_delay_ms = self._parsed.agent_step_delays_ms[trace_index]
                self._trace_cursor += 1
                self._catchup_restore_cursor = None
                is_last = self._trace_cursor >= len(self._parsed.agent_steps)
        _apply_request_context_to_step(step, request_context)
        if duplicate_step:
            logger.info(
                "Replay duplicate catch-up response sandbox request assistant_messages=%d trace_cursor=%d trace_path=%s",
                requested_assistant_count,
                self.snapshot()["trace_cursor"],
                self._parsed.trace_path,
            )
            return _anthropic_duplicate_response(
                step,
                model_name=self._parsed.model_name,
                is_streaming=is_streaming,
            )
        effective_delay_ms = self._effective_delay_ms(trace_delay_ms)
        if effective_delay_ms > 0:
            time.sleep(effective_delay_ms / 1000.0)
        return _anthropic_response(
            step,
            model_name=self._parsed.model_name,
            trace_index=trace_index,
            is_last=is_last,
            is_streaming=is_streaming,
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

    def _resolve_sidechain_for_request(self, payload: dict[str, Any]) -> ParsedClaudeCodeSidechain | None:
        prompt_candidates = [_normalize_prompt_key(prompt) for prompt in _user_text_messages_from_request(payload)]
        for prompt_key in prompt_candidates:
            sidechain = self._sidechains_by_prompt.get(prompt_key)
            if sidechain is not None:
                return sidechain
        fuzzy_matches: dict[Path, ParsedClaudeCodeSidechain] = {}
        for prompt_key in prompt_candidates:
            for stored_prompt, sidechain in self._sidechains_by_prompt.items():
                if sidechain is None:
                    continue
                if stored_prompt in prompt_key or prompt_key in stored_prompt:
                    fuzzy_matches[sidechain.trace_path] = sidechain
        if len(fuzzy_matches) == 1:
            return next(iter(fuzzy_matches.values()))
        return None

    def _handle_sidechain_request(
        self,
        sidechain: ParsedClaudeCodeSidechain,
        *,
        payload: dict[str, Any],
        request_context: ReplayRequestContext,
        is_streaming: bool,
    ) -> dict[str, Any]:
        trace_index = _assistant_message_count_from_request(payload)
        if trace_index >= len(sidechain.agent_steps):
            return _helper_model_response(model_name=sidechain.model_name, is_streaming=is_streaming)
        step = copy.deepcopy(sidechain.agent_steps[trace_index])
        _apply_request_context_to_step(step, request_context)
        trace_delay_ms = sidechain.agent_step_delays_ms[trace_index]
        effective_delay_ms = self._effective_delay_ms(trace_delay_ms)
        if effective_delay_ms > 0:
            time.sleep(effective_delay_ms / 1000.0)
        return _anthropic_response(
            step,
            model_name=sidechain.model_name,
            trace_index=trace_index,
            is_last=(trace_index + 1) >= len(sidechain.agent_steps),
            is_streaming=is_streaming,
        )


def _iter_agent_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("claude-code trace payload is missing a steps list")
    return [
        step
        for step in steps
        if isinstance(step, dict) and step.get("source") == "agent"
    ]


def _trace_model_name(payload: dict[str, Any]) -> str:
    agent = payload.get("agent")
    if isinstance(agent, dict):
        model_name = agent.get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    for step in _iter_agent_steps(payload):
        model_name = step.get("model_name")
        if isinstance(model_name, str) and model_name.strip():
            return model_name.strip()
    return _DEFAULT_MODEL_NAME


def _parse_timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw_value = value.strip()
    if raw_value.endswith("Z"):
        raw_value = raw_value[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw_value).timestamp()
    except ValueError:
        return None


def _is_placeholder_text(value: object) -> bool:
    return isinstance(value, str) and _PLACEHOLDER_RE.fullmatch(value.strip()) is not None


def _meaningful_step_text(step: dict[str, Any]) -> str | None:
    message = step.get("message")
    if not isinstance(message, str):
        return None
    stripped = message.strip()
    if not stripped or _PLACEHOLDER_RE.fullmatch(stripped):
        return None
    if isinstance(step.get("tool_calls"), list) and step["tool_calls"]:
        return None
    return message


def _parse_observation_results(step: dict[str, Any]) -> dict[str, str]:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return {}
    results = observation.get("results")
    if not isinstance(results, list):
        return {}
    by_call_id: dict[str, str] = {}
    for item in results:
        if not isinstance(item, dict):
            continue
        call_id = item.get("source_call_id")
        content = item.get("content")
        if isinstance(call_id, str) and call_id and isinstance(content, str):
            by_call_id[call_id] = content
    return by_call_id


def _parse_metadata_from_observation_text(obs: str | None) -> dict[str, Any] | None:
    if not isinstance(obs, str) or "[metadata]" not in obs:
        return None
    marker = "[metadata] "
    idx = obs.find(marker)
    if idx < 0:
        return None
    try:
        parsed = json.loads(obs[idx + len(marker):])
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _step_tool_result_metadata(step: dict[str, Any]) -> dict[str, Any] | None:
    extra = step.get("extra")
    if not isinstance(extra, dict):
        return None
    for key in ("tool_result_metadata", "metadata"):
        value = extra.get(key)
        if isinstance(value, dict):
            return value
    return None


def _tool_result_payload(tool: dict[str, Any]) -> dict[str, Any] | None:
    metadata = tool.get("tool_result_metadata")
    if not isinstance(metadata, dict):
        return None
    payload = metadata.get("tool_use_result")
    return payload if isinstance(payload, dict) else None


def _extract_tool_entries(step: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = step.get("tool_calls")
    if not isinstance(tool_calls, list):
        return []
    obs_by_call_id = _parse_observation_results(step)
    tool_result_metadata = _step_tool_result_metadata(step)
    tools: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        function_name = tool_call.get("function_name")
        if not isinstance(function_name, str) or not function_name:
            continue
        tool_call_id = tool_call.get("tool_call_id")
        arguments = tool_call.get("arguments")
        observation_text = (
            obs_by_call_id.get(tool_call_id)
            if isinstance(tool_call_id, str)
            else None
        )
        tools.append(
            {
                "tool_call_id": tool_call_id if isinstance(tool_call_id, str) else "",
                "function_name": function_name,
                "arguments": dict(arguments) if isinstance(arguments, dict) else {},
                "observation_text": observation_text,
                "observation_metadata": _parse_metadata_from_observation_text(observation_text),
                "tool_result_metadata": tool_result_metadata,
            }
        )
    return tools


def _is_effective_tool_function(function_name: str) -> bool:
    return function_name not in _SKIP_TOOLS


def _turn_has_content(turn: dict[str, Any]) -> bool:
    return bool(turn["text_segments"] or turn["tools"])


def _new_turn(*, model_name: str, timestamp_s: float | None) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "text_segments": [],
        "tools": [],
        "start_timestamp_s": timestamp_s,
        "end_timestamp_s": timestamp_s,
    }


def _merge_agent_steps(payload: dict[str, Any]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    pending_text_turn: dict[str, Any] | None = None

    for step in _iter_agent_steps(payload):
        timestamp_s = _parse_timestamp(step.get("timestamp"))
        model_name = str(step.get("model_name") or _trace_model_name(payload))
        text = _meaningful_step_text(step)
        all_tools = _extract_tool_entries(step)
        effective_tools = [tool for tool in all_tools if _is_effective_tool_function(tool["function_name"])]

        if text is None and not effective_tools:
            if pending_text_turn is not None and timestamp_s is not None:
                pending_text_turn["end_timestamp_s"] = timestamp_s
            continue

        if text is not None and not effective_tools:
            if pending_text_turn is None:
                pending_text_turn = _new_turn(model_name=model_name, timestamp_s=timestamp_s)
            pending_text_turn["model_name"] = model_name or pending_text_turn["model_name"]
            pending_text_turn["text_segments"].append(text)
            if timestamp_s is not None:
                if pending_text_turn["start_timestamp_s"] is None:
                    pending_text_turn["start_timestamp_s"] = timestamp_s
                pending_text_turn["end_timestamp_s"] = timestamp_s
            continue

        if effective_tools:
            if pending_text_turn is None:
                turn = _new_turn(model_name=model_name, timestamp_s=timestamp_s)
            else:
                turn = pending_text_turn
            turn["model_name"] = model_name or turn["model_name"]
            turn["tools"] = list(effective_tools)
            if timestamp_s is not None:
                if turn["start_timestamp_s"] is None:
                    turn["start_timestamp_s"] = timestamp_s
                turn["end_timestamp_s"] = timestamp_s
            if _turn_has_content(turn):
                merged.append(turn)
            pending_text_turn = None

    if pending_text_turn is not None and _turn_has_content(pending_text_turn):
        merged.append(pending_text_turn)

    return merged


def _compute_turn_delays_ms(turns: list[dict[str, Any]]) -> list[float | None]:
    delays: list[float | None] = []
    previous_end_s: float | None = None
    for turn in turns:
        start_s = turn.get("start_timestamp_s")
        end_s = turn.get("end_timestamp_s")
        if previous_end_s is None or start_s is None:
            delays.append(None)
        else:
            delays.append(max(0.0, (float(start_s) - previous_end_s) * 1000.0))
        if isinstance(end_s, (int, float)):
            previous_end_s = float(end_s)
        elif isinstance(start_s, (int, float)):
            previous_end_s = float(start_s)
    return delays


def _normalize_prompt_key(value: str) -> str:
    return value.replace("\r\n", "\n").strip()


def _extract_text_blocks(content: object) -> list[str]:
    if isinstance(content, str):
        stripped = content.strip()
        return [content] if stripped else []
    if not isinstance(content, list):
        return []
    texts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return texts


def _user_text_messages_from_request(payload: dict[str, Any]) -> list[str]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    prompts: list[str] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        texts = _extract_text_blocks(message.get("content"))
        if texts:
            prompts.append("\n".join(texts))
    return prompts


def _assistant_message_count_from_request(payload: dict[str, Any]) -> int:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return 0
    return sum(1 for message in messages if isinstance(message, dict) and message.get("role") == "assistant")


def _parse_sidechain_jsonl(trace_path: Path) -> ParsedClaudeCodeSidechain | None:
    agent_id = ""
    initial_prompt: str | None = None
    merged_turns: list[dict[str, Any]] = []
    pending_turn: dict[str, Any] | None = None
    pending_message_id: str | None = None

    for raw_line in trace_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if not agent_id and isinstance(entry.get("agentId"), str):
            agent_id = entry["agentId"].strip()
        message = entry.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        timestamp_s = _parse_timestamp(entry.get("timestamp"))
        if role == "user":
            if pending_turn is not None and _turn_has_content(pending_turn):
                merged_turns.append(pending_turn)
            pending_turn = None
            pending_message_id = None
            if initial_prompt is None:
                texts = _extract_text_blocks(message.get("content"))
                if texts:
                    initial_prompt = "\n".join(texts)
            continue
        if role != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        message_id = _coerce_string(message.get("id"), allow_empty=True, allow_placeholder=True) or ""
        model_name = _coerce_string(message.get("model")) or _DEFAULT_MODEL_NAME
        if pending_turn is None or pending_message_id != message_id:
            if pending_turn is not None and _turn_has_content(pending_turn):
                merged_turns.append(pending_turn)
            pending_turn = _new_turn(model_name=model_name, timestamp_s=timestamp_s)
            pending_message_id = message_id
        else:
            pending_turn["model_name"] = model_name or pending_turn["model_name"]
            if timestamp_s is not None:
                if pending_turn["start_timestamp_s"] is None:
                    pending_turn["start_timestamp_s"] = timestamp_s
                pending_turn["end_timestamp_s"] = timestamp_s
        assert pending_turn is not None
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    pending_turn["text_segments"].append(text)
            elif block.get("type") == "tool_use":
                pending_turn["tools"].append(
                    {
                        "tool_call_id": _coerce_string(block.get("id"), allow_empty=True, allow_placeholder=True) or "",
                        "function_name": _coerce_string(block.get("name")) or "",
                        "arguments": dict(block.get("input")) if isinstance(block.get("input"), dict) else {},
                        "observation_text": None,
                        "observation_metadata": None,
                        "tool_result_metadata": None,
                    }
                )
        if timestamp_s is not None:
            if pending_turn["start_timestamp_s"] is None:
                pending_turn["start_timestamp_s"] = timestamp_s
            pending_turn["end_timestamp_s"] = timestamp_s

    if pending_turn is not None and _turn_has_content(pending_turn):
        merged_turns.append(pending_turn)

    prompt = _coerce_string(initial_prompt, allow_empty=False, allow_placeholder=True)
    if prompt is None or not merged_turns:
        return None
    _mark_detached_background_tools(merged_turns)
    if not agent_id:
        agent_id = trace_path.stem.removeprefix("agent-")
    return ParsedClaudeCodeSidechain(
        trace_path=trace_path,
        agent_id=agent_id,
        initial_prompt=prompt,
        model_name=str(merged_turns[0].get("model_name") or _DEFAULT_MODEL_NAME),
        agent_steps=tuple(merged_turns),
        agent_step_delays_ms=tuple(_compute_turn_delays_ms(merged_turns)),
    )


def _task_agent_id_from_tool(tool: dict[str, Any]) -> str | None:
    payload = _tool_result_payload(tool) or {}
    metadata = tool.get("observation_metadata")
    metadata_payload = metadata if isinstance(metadata, dict) else {}
    return _first_string(
        payload.get("agentId"),
        metadata_payload.get("agentId"),
    )


def _load_task_sidechains(
    trace_path: Path,
) -> tuple[dict[str, ParsedClaudeCodeSidechain], dict[str, ParsedClaudeCodeSidechain | None]]:
    payload = load_trace_payload(trace_path)
    by_agent_id: dict[str, ParsedClaudeCodeSidechain] = {}
    sessions_root = trace_path.parent / "sessions" / "projects"
    if sessions_root.exists():
        for subagent_path in sorted(sessions_root.glob("**/subagents/agent-*.jsonl")):
            parsed = _parse_sidechain_jsonl(subagent_path)
            if parsed is None:
                continue
            by_agent_id[parsed.agent_id] = parsed

    prompt_to_sidechain: dict[str, ParsedClaudeCodeSidechain | None] = {}
    for step in _iter_agent_steps(payload):
        for tool in _extract_tool_entries(step):
            if tool.get("function_name") != "Task":
                continue
            prompt = _coerce_string(tool["arguments"].get("prompt"))
            if prompt is None:
                continue
            prompt_key = _normalize_prompt_key(prompt)
            agent_id = _task_agent_id_from_tool(tool)
            if agent_id is not None:
                sidechain = by_agent_id.get(agent_id)
                if sidechain is not None and _normalize_prompt_key(sidechain.initial_prompt) != prompt_key:
                    logger.warning(
                        "Task prompt mismatch for agent_id=%s trace=%s",
                        agent_id,
                        trace_path,
                    )
                prompt_to_sidechain[prompt_key] = sidechain
                continue
            prompt_to_sidechain.setdefault(prompt_key, None)

    for sidechain in by_agent_id.values():
        prompt_to_sidechain.setdefault(_normalize_prompt_key(sidechain.initial_prompt), sidechain)

    return by_agent_id, prompt_to_sidechain


def _estimate_input_tokens(payload: dict[str, Any]) -> int:
    # Claude Code only needs a stable token-count response to avoid its
    # fallback bootstrap request consuming the first replay turn. A rough
    # estimate is enough here.
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    estimated = max(1, len(serialized) // 4)
    tools = payload.get("tools")
    if isinstance(tools, list):
        estimated += max(0, len(tools))
    return estimated


def _anthropic_count_tokens_response(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "input_tokens": _estimate_input_tokens(payload),
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }


def _coerce_string(
    value: object,
    *,
    allow_empty: bool = False,
    allow_placeholder: bool = False,
) -> str | None:
    if not isinstance(value, str):
        return None
    if not allow_placeholder and _PLACEHOLDER_RE.fullmatch(value.strip()):
        return None
    if not allow_empty and value == "":
        return None
    return value


def _first_string(
    *candidates: object,
    allow_empty: bool = False,
    allow_placeholder: bool = False,
) -> str | None:
    for value in candidates:
        resolved = _coerce_string(
            value,
            allow_empty=allow_empty,
            allow_placeholder=allow_placeholder,
        )
        if resolved is not None:
            return resolved
    return None


def _request_context_from_payload(
    payload: dict[str, Any],
    *,
    background_task_ids_by_tool_use_id: dict[str, str],
    recorded_observations_by_tool_use_id: dict[str, str],
) -> ReplayRequestContext:
    return ReplayRequestContext(
        recovered_git_commit_hash=_detect_recovered_git_commit_hash(payload),
        background_task_aliases=_detect_background_task_aliases(
            payload,
            background_task_ids_by_tool_use_id=background_task_ids_by_tool_use_id,
        ),
        filename_aliases=_detect_filename_aliases(
            payload,
            recorded_observations_by_tool_use_id=recorded_observations_by_tool_use_id,
        ),
    )


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def _patch_to_old_new(structured_patch: list[dict[str, Any]]) -> tuple[str, str]:
    old_lines: list[str] = []
    new_lines: list[str] = []
    for hunk in structured_patch:
        if not isinstance(hunk, dict):
            continue
        for line in hunk.get("lines", []):
            if not isinstance(line, str) or not line:
                continue
            marker, content = line[0], line[1:]
            if marker == "-":
                old_lines.append(content)
            elif marker == "+":
                new_lines.append(content)
            elif marker == " ":
                old_lines.append(content)
                new_lines.append(content)
    return ("\n".join(old_lines), "\n".join(new_lines))


def _build_noop_bash(tool_name: str, *, reason: str) -> tuple[str, dict[str, Any]]:
    del tool_name
    return (
        "Bash",
        {
            "command": f"printf '%s\\n' {_shell_quote(reason)} >/dev/null",
            "run_in_background": False,
            "timeout": _NOOP_BASH_TIMEOUT_MS,
        },
    )


def _python_command(script: str) -> str:
    return f"python3 -c {_shell_quote(script)}"


def _b64_text(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _bash_spec(command: str, **kwargs: Any) -> tuple[str, dict[str, Any]]:
    bash_input: dict[str, Any] = {"command": command, "run_in_background": False}
    bash_input.update(kwargs)
    return ("Bash", bash_input)


def _normalize_runtime_specific_bash_command(command: str) -> str:
    command = command.replace(
        'postconf -e "relay_domains = hash:/var/lib/mailman3/data/postfix_domains"',
        "true",
    )
    command = command.replace(
        "postconf -e 'relay_domains = hash:/var/lib/mailman3/data/postfix_domains'",
        "true",
    )
    stripped = " ".join(command.split())
    if stripped == "postfix start 2>&1 || postfix reload 2>&1":
        return "service postfix start 2>&1 || service postfix restart 2>&1 || postfix start 2>&1 || postfix reload 2>&1"
    if stripped == "postfix reload 2>&1":
        return "service postfix reload 2>&1 || service postfix start 2>&1 || postfix reload 2>&1"
    if "postfix reload 2>&1" in command:
        return command.replace(
            "postfix reload 2>&1",
            "service postfix reload 2>&1 || service postfix start 2>&1 || postfix reload 2>&1",
        )
    if stripped == "mailman --run-as-root start 2>&1":
        return (
            "/usr/lib/mailman3/bin/mailman -C /etc/mailman3/mailman.cfg --run-as-root start 2>&1 "
            "|| mailman --run-as-root start 2>&1"
        )
    if stripped.startswith("cp ") and "/usr/local/bin/" in command and "install -d -m 755 /usr/local/bin" not in command:
        command = f"install -d -m 755 /usr/local/bin && {command}"
    if "apt-get" in command or re.search(r"(^|\\s)apt\\s", command):
        return _wrap_replay_apt_command(command)
    return command


def _normalize_runtime_specific_file_content(*, file_path: str | None, content: str) -> str:
    if file_path != "/etc/postfix/main.cf":
        return content
    return _MAILMAN_RELAY_DOMAINS_LINE_RE.sub("", content)


def _wrap_replay_apt_command(command: str) -> str:
    if "crab_retry_apt_command" in command:
        return command
    wrapped = _shell_quote(command)
    return (
        "wait_for_apt_lock() { "
        "while pgrep -x apt-get >/dev/null 2>&1 || pgrep -x apt >/dev/null 2>&1 || pgrep -x dpkg >/dev/null 2>&1; "
        "do sleep 1; done; "
        "}; "
        "crab_retry_apt_command() { "
        "attempts=0; "
        "while true; do "
        "wait_for_apt_lock; "
        "/bin/bash -lc \"$1\"; "
        "status=$?; "
        "if [ \"$status\" -eq 0 ]; then return 0; fi; "
        "attempts=$((attempts + 1)); "
        "if [ \"$attempts\" -ge 3 ]; then return \"$status\"; fi; "
        "sleep \"$attempts\"; "
        "done; "
        "}; "
        f"crab_retry_apt_command {wrapped}"
    )


def _controlled_background_task_ids(turns: list[dict[str, Any]]) -> set[str]:
    controlled_ids: set[str] = set()
    for turn in turns:
        tools = turn.get("tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            if tool.get("function_name") not in {"TaskOutput", "TaskStop"}:
                continue
            arguments = tool.get("arguments")
            if not isinstance(arguments, dict):
                continue
            task_id = _coerce_string(arguments.get("task_id"), allow_placeholder=True)
            if task_id:
                controlled_ids.add(task_id)
    return controlled_ids


def _mark_detached_background_tools(turns: list[dict[str, Any]]) -> None:
    controlled_ids = _controlled_background_task_ids(turns)
    for turn in turns:
        tools = turn.get("tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict) or tool.get("function_name") != "Bash":
                continue
            arguments = tool.get("arguments")
            if not isinstance(arguments, dict) or not isinstance(arguments.get("run_in_background"), bool):
                continue
            if not arguments["run_in_background"]:
                continue
            recorded_task_id = _recorded_background_task_id(tool)
            if recorded_task_id is not None and recorded_task_id in controlled_ids:
                continue
            tool["crab_detach_background"] = True


def _iter_tool_result_blocks(payload: dict[str, Any]) -> list[tuple[str | None, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    blocks: list[tuple[str | None, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            blocks.append((None, content))
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            block_content = block.get("content")
            if isinstance(block_content, str):
                tool_use_id = block.get("tool_use_id")
                blocks.append((tool_use_id if isinstance(tool_use_id, str) else None, block_content))
    return blocks


def _iter_tool_result_texts(payload: dict[str, Any]) -> list[str]:
    return [content for _, content in _iter_tool_result_blocks(payload)]


_GIT_REFLOG_COMMIT_RE = re.compile(r"(?m)\b([0-9a-f]{7,40})\s+HEAD@\{\d+\}:\s+commit:\s+([^\n]+)")
_GIT_HASH_TOKEN_RE = re.compile(r"\b([0-9a-f]{7,40})\b")
_HASH_SENSITIVE_GIT_COMMAND_PREFIXES = (
    "git cherry-pick ",
    "git diff ",
    "git log --oneline ",
    "git merge ",
    "git show --stat ",
)


def _detect_recovered_git_commit_hash(payload: dict[str, Any]) -> str | None:
    matches: list[str] = []
    for text in _iter_tool_result_texts(payload):
        for match in _GIT_REFLOG_COMMIT_RE.finditer(text):
            matches.append(match.group(1))
    unique_hashes = tuple(dict.fromkeys(matches))
    if len(unique_hashes) == 1:
        return unique_hashes[0]
    return None


def _extract_background_task_id_from_text(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    metadata = _parse_metadata_from_observation_text(value)
    if isinstance(metadata, dict):
        return _first_string(
            metadata.get("backgroundTaskId"),
            metadata.get("task_id"),
            ((metadata.get("task") or {}).get("task_id") if isinstance(metadata.get("task"), dict) else None),
            allow_placeholder=True,
        )
    xml_match = re.search(r"<task_id>\s*([A-Za-z0-9_-]+)\s*</task_id>", value)
    if xml_match is not None:
        return xml_match.group(1)
    match = _BACKGROUND_TASK_ID_RE.search(value)
    if match is None:
        return None
    return match.group(1)


def _recorded_background_task_id(tool: dict[str, Any]) -> str | None:
    payload = _tool_result_payload(tool) or {}
    metadata = tool.get("observation_metadata")
    metadata_payload = metadata if isinstance(metadata, dict) else {}
    return _first_string(
        payload.get("backgroundTaskId"),
        metadata_payload.get("backgroundTaskId"),
        allow_placeholder=True,
    )


def _detached_background_log_path(tool: dict[str, Any]) -> str:
    tool_use_id = _coerce_string(
        tool.get("tool_call_id"),
        allow_empty=True,
        allow_placeholder=True,
    ) or uuid.uuid4().hex[:12]
    safe_tool_use_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", tool_use_id)[:64] or "task"
    return f"/tmp/crab-bg-{safe_tool_use_id}.log"


def _detach_background_bash_command(command: str, *, tool: dict[str, Any]) -> str:
    log_path = _detached_background_log_path(tool)
    return (
        f"crab_bg_log={_shell_quote(log_path)}; "
        f"nohup setsid /bin/bash -lc {_shell_quote(command)} "
        '> "$crab_bg_log" 2>&1 < /dev/null & '
        'crab_bg_pid=$!; '
        'printf "Command detached into sandbox with PID: %s. Output is being written to: %s\\n" '
        '"$crab_bg_pid" "$crab_bg_log"'
    )


def _recorded_background_task_ids_by_tool_use_id(turns: tuple[dict[str, Any], ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for turn in turns:
        tools = turn.get("tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_use_id = _coerce_string(tool.get("tool_call_id"), allow_empty=True, allow_placeholder=True)
            recorded_task_id = _recorded_background_task_id(tool)
            if recorded_task_id is None and tool.get("function_name") in {"TaskOutput", "TaskStop"}:
                arguments = tool.get("arguments")
                if isinstance(arguments, dict):
                    recorded_task_id = _coerce_string(arguments.get("task_id"), allow_placeholder=True)
            if tool_use_id and recorded_task_id:
                mapping[tool_use_id] = recorded_task_id
    return mapping


def _recorded_observations_by_tool_use_id(turns: tuple[dict[str, Any], ...]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for turn in turns:
        tools = turn.get("tools")
        if not isinstance(tools, list):
            continue
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_use_id = _coerce_string(tool.get("tool_call_id"), allow_empty=True, allow_placeholder=True)
            observation_text = tool.get("observation_text")
            if tool_use_id and isinstance(observation_text, str) and observation_text:
                mapping[tool_use_id] = observation_text
    return mapping


_LS_LISTING_ENTRY_RE = re.compile(
    r"^[bcdlps-][rwxStTs-]{9}\s+\d+\s+\S+\s+\S+\s+(\d+)\s+\w{3}\s+\d{1,2}\s+(?:\d{2}:\d{2}|\d{4})\s+(.+)$"
)


def _parse_ls_listing_entries(value: str | None) -> list[tuple[str, int, str]]:
    if not isinstance(value, str) or not value:
        return []
    entries: list[tuple[str, int, str]] = []
    for line in value.splitlines():
        match = _LS_LISTING_ENTRY_RE.match(line.strip())
        if match is None:
            continue
        try:
            size = int(match.group(1))
        except ValueError:
            continue
        name = match.group(2).strip()
        if not name or name in {".", ".."}:
            continue
        suffix = Path(name).suffix.lower()
        entries.append((name, size, suffix))
    return entries


def _unique_names_by_listing_signature(entries: list[tuple[str, int, str]]) -> dict[tuple[int, str], str]:
    names_by_signature: dict[tuple[int, str], set[str]] = {}
    for name, size, suffix in entries:
        names_by_signature.setdefault((size, suffix), set()).add(name)
    return {
        signature: next(iter(names))
        for signature, names in names_by_signature.items()
        if len(names) == 1
    }


def _infer_filename_aliases_from_listings(recorded_listing: str, live_listing: str) -> dict[str, str]:
    recorded_entries = _parse_ls_listing_entries(recorded_listing)
    live_entries = _parse_ls_listing_entries(live_listing)
    if len(recorded_entries) < 2 or len(live_entries) < 2:
        return {}
    recorded_by_sig = _unique_names_by_listing_signature(recorded_entries)
    live_by_sig = _unique_names_by_listing_signature(live_entries)
    aliases: dict[str, str] = {}
    for signature, recorded_name in recorded_by_sig.items():
        live_name = live_by_sig.get(signature)
        if live_name is None or live_name == recorded_name:
            continue
        aliases.setdefault(recorded_name, live_name)
    return aliases


def _detect_filename_aliases(
    payload: dict[str, Any],
    *,
    recorded_observations_by_tool_use_id: dict[str, str],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for tool_use_id, live_content in _iter_tool_result_blocks(payload):
        if tool_use_id is None:
            continue
        recorded_content = recorded_observations_by_tool_use_id.get(tool_use_id)
        if recorded_content is None:
            continue
        inferred = _infer_filename_aliases_from_listings(recorded_content, live_content)
        for recorded_name, live_name in inferred.items():
            aliases.setdefault(recorded_name, live_name)
    return aliases


def _detect_background_task_aliases(
    payload: dict[str, Any],
    *,
    background_task_ids_by_tool_use_id: dict[str, str],
) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for tool_use_id, content in _iter_tool_result_blocks(payload):
        if tool_use_id is None:
            continue
        recorded_task_id = background_task_ids_by_tool_use_id.get(tool_use_id)
        if recorded_task_id is None:
            continue
        live_task_id = _extract_background_task_id_from_text(content)
        if live_task_id is None:
            continue
        aliases[recorded_task_id] = live_task_id
    return aliases


def _replace_single_git_hash_token(value: str, replacement_hash: str | None) -> str:
    if replacement_hash is None:
        return value
    tokens = tuple(dict.fromkeys(match.group(1) for match in _GIT_HASH_TOKEN_RE.finditer(value)))
    if len(tokens) != 1 or tokens[0] == replacement_hash:
        return value
    token = tokens[0]
    return re.sub(rf"\b{re.escape(token)}\b", replacement_hash, value)


def _normalize_git_command_literals(command: str, *, replacement_hash: str | None) -> str:
    stripped = command.strip()
    if not any(stripped.startswith(prefix) for prefix in _HASH_SENSITIVE_GIT_COMMAND_PREFIXES):
        return command
    return _replace_single_git_hash_token(command, replacement_hash)


def _normalize_conflict_marker_hashes(value: str, *, replacement_hash: str | None) -> str:
    if ">>>>>>>" not in value:
        return value
    return _replace_single_git_hash_token(value, replacement_hash)


def _apply_filename_aliases(value: str, aliases: dict[str, str]) -> str:
    if not aliases:
        return value
    rewritten = value
    for recorded_name in sorted(aliases, key=len, reverse=True):
        rewritten = rewritten.replace(recorded_name, aliases[recorded_name])
    return rewritten


def _apply_request_context_to_step(step: dict[str, Any], context: ReplayRequestContext) -> None:
    replacement_hash = context.recovered_git_commit_hash
    filename_aliases = context.filename_aliases
    for tool in step.get("tools", []):
        if not isinstance(tool, dict):
            continue
        arguments = tool.get("arguments")
        if not isinstance(arguments, dict):
            continue
        function_name = tool.get("function_name")
        if function_name in {"TaskOutput", "TaskStop"}:
            task_id = arguments.get("task_id")
            if isinstance(task_id, str):
                arguments["task_id"] = context.background_task_aliases.get(task_id, task_id)
        for key in ("file_path",):
            value = arguments.get(key)
            if isinstance(value, str):
                arguments[key] = _apply_filename_aliases(value, filename_aliases)
        if replacement_hash is None:
            if function_name == "Bash":
                command = arguments.get("command")
                if isinstance(command, str):
                    arguments["command"] = _apply_filename_aliases(command, filename_aliases)
            elif function_name == "Edit":
                for key in ("old_string", "new_string"):
                    value = arguments.get(key)
                    if isinstance(value, str):
                        arguments[key] = _apply_filename_aliases(value, filename_aliases)
            elif function_name == "Write":
                value = arguments.get("content")
                if isinstance(value, str):
                    arguments["content"] = _apply_filename_aliases(value, filename_aliases)
            continue
        if function_name == "Bash":
            command = arguments.get("command")
            if isinstance(command, str):
                arguments["command"] = _normalize_git_command_literals(
                    _apply_filename_aliases(command, filename_aliases),
                    replacement_hash=replacement_hash,
                )
        elif function_name == "Edit":
            for key in ("old_string", "new_string"):
                value = arguments.get(key)
                if isinstance(value, str):
                    arguments[key] = _normalize_conflict_marker_hashes(
                        _apply_filename_aliases(value, filename_aliases),
                        replacement_hash=replacement_hash,
                    )
        elif function_name == "Write":
            value = arguments.get("content")
            if isinstance(value, str):
                arguments["content"] = _normalize_conflict_marker_hashes(
                    _apply_filename_aliases(value, filename_aliases),
                    replacement_hash=replacement_hash,
                )


def _resolve_read_fallbacks(tool: dict[str, Any]) -> tuple[str | None, int | None, int | None]:
    args = tool["arguments"]
    tool_result = _tool_result_payload(tool) or {}
    file_result = tool_result.get("file")
    file_payload = file_result if isinstance(file_result, dict) else {}
    file_path = _first_string(
        args.get("file_path"),
        file_payload.get("filePath"),
    )
    offset = _coerce_int(args.get("offset"))
    limit = _coerce_int(args.get("limit"))
    if offset is None:
        start_line = _coerce_int(file_payload.get("startLine"))
        if start_line is not None:
            offset = start_line
    if limit is None:
        num_lines = _coerce_int(file_payload.get("numLines"))
        if num_lines is not None:
            limit = num_lines
    return file_path, offset, limit


def _resolve_write_fallbacks(tool: dict[str, Any]) -> tuple[str | None, str | None]:
    args = tool["arguments"]
    metadata = tool.get("observation_metadata")
    metadata_payload = metadata if isinstance(metadata, dict) else {}
    tool_result = _tool_result_payload(tool) or {}
    file_path = _first_string(
        args.get("file_path"),
        metadata_payload.get("filePath"),
        tool_result.get("filePath"),
    )
    content = _first_string(
        args.get("content"),
        metadata_payload.get("content"),
        tool_result.get("content"),
        allow_empty=True,
    )
    return file_path, content


def _resolve_edit_fallbacks(tool: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    args = tool["arguments"]
    metadata = tool.get("observation_metadata")
    metadata_payload = metadata if isinstance(metadata, dict) else {}
    tool_result = _tool_result_payload(tool) or {}
    structured_patch = None
    for candidate in (
        metadata_payload.get("structuredPatch"),
        tool_result.get("structuredPatch"),
    ):
        if isinstance(candidate, list) and candidate:
            structured_patch = candidate
            break
    patch_old = patch_new = None
    if structured_patch is not None:
        patch_old, patch_new = _patch_to_old_new(structured_patch)
    file_path = _first_string(
        args.get("file_path"),
        metadata_payload.get("filePath"),
        tool_result.get("filePath"),
    )
    old_string = _first_string(
        args.get("old_string"),
        metadata_payload.get("oldString"),
        tool_result.get("oldString"),
        patch_old,
        allow_empty=True,
    )
    new_string = _first_string(
        args.get("new_string"),
        metadata_payload.get("newString"),
        tool_result.get("newString"),
        patch_new,
        allow_empty=True,
    )
    return file_path, old_string, new_string


def _tool_to_replay_spec(
    tool: dict[str, Any],
    *,
    duplicate_catchup: bool = False,
) -> tuple[str, dict[str, Any]] | None:
    function_name = tool["function_name"]
    args = tool["arguments"]

    if function_name in _SKIP_TOOLS:
        return None

    if function_name == "Task":
        prompt = _coerce_string(args.get("prompt"))
        if prompt is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved Task prompt")
        task_input = dict(args)
        task_input["prompt"] = prompt
        description = _coerce_string(args.get("description"))
        if description is not None:
            task_input["description"] = description
        else:
            task_input.pop("description", None)
        subagent_type = _coerce_string(args.get("subagent_type"))
        if subagent_type is not None:
            task_input["subagent_type"] = subagent_type
        return ("Task", task_input)

    if function_name == "TaskOutput":
        task_id = _coerce_string(args.get("task_id"))
        if task_id is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved TaskOutput task_id")
        task_input: dict[str, Any] = {"task_id": task_id}
        block = args.get("block")
        if isinstance(block, bool):
            task_input["block"] = block
        timeout = _coerce_int(args.get("timeout"))
        if timeout is not None:
            task_input["timeout"] = timeout
        return ("TaskOutput", task_input)

    if function_name == "TaskStop":
        task_id = _coerce_string(args.get("task_id"))
        if task_id is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved TaskStop task_id")
        return ("TaskStop", {"task_id": task_id})

    if function_name == "Bash":
        command = _coerce_string(args.get("command"))
        if command is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved Bash command")
        normalized_command = _normalize_runtime_specific_bash_command(command)
        bash_input: dict[str, Any] = {"command": normalized_command, "run_in_background": False}
        description = _coerce_string(args.get("description"))
        if description is not None:
            bash_input["description"] = description
        timeout = _coerce_int(args.get("timeout"))
        if timeout is not None:
            bash_input["timeout"] = timeout
        run_in_background = args.get("run_in_background")
        if isinstance(run_in_background, bool):
            bash_input["run_in_background"] = run_in_background
        if bool(tool.get("crab_detach_background")):
            bash_input["command"] = _detach_background_bash_command(normalized_command, tool=tool)
            bash_input["run_in_background"] = False
        return ("Bash", bash_input)

    if function_name == "Read":
        file_path, offset, limit = _resolve_read_fallbacks(tool)
        if file_path is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved Read target")
        script_lines = [
            "from pathlib import Path",
            f"path = Path({file_path!r})",
            "text = path.read_text(encoding='utf-8')",
            "lines = text.splitlines()",
        ]
        if offset is not None:
            script_lines.append(f"start = max({offset} - 1, 0)")
        else:
            script_lines.append("start = 0")
        if limit is not None:
            script_lines.append(f"selected = lines[start:start + max({limit}, 0)]")
        else:
            script_lines.append("selected = lines[start:]")
        script_lines.extend(
            [
                "import sys",
                "output = '\\n'.join(selected)",
                "if output:",
                "    sys.stdout.write(output)",
                "    if text.endswith('\\n'):",
                "        sys.stdout.write('\\n')",
            ]
        )
        return _bash_spec(_python_command("\n".join(script_lines)), timeout=120_000)

    if function_name == "Edit":
        file_path, old_string, new_string = _resolve_edit_fallbacks(tool)
        if file_path is None or old_string is None or new_string is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved Edit payload")
        old_string = _normalize_runtime_specific_file_content(file_path=file_path, content=old_string)
        new_string = _normalize_runtime_specific_file_content(file_path=file_path, content=new_string)
        replace_all = args.get("replace_all")
        replace_all_flag = isinstance(replace_all, bool) and replace_all
        encoded_old = _b64_text(old_string)
        encoded_new = _b64_text(new_string)
        script_lines = [
            "import base64",
            "from pathlib import Path",
            f"path = Path({file_path!r})",
            f"old = base64.b64decode('{encoded_old}').decode('utf-8')",
            f"new = base64.b64decode('{encoded_new}').decode('utf-8')",
            "text = path.read_text(encoding='utf-8')",
        ]
        if duplicate_catchup:
            script_lines.extend(
                [
                    "if old not in text:",
                    "    if new in text:",
                    "        raise SystemExit(0)",
                    "    raise SystemExit(1)",
                ]
            )
        else:
            script_lines.extend(
                [
                    "if old not in text:",
                    "    raise SystemExit(1)",
                ]
            )
        script_lines.extend(
            [
                (
                    "updated = text.replace(old, new)"
                    if replace_all_flag
                    else "updated = text.replace(old, new, 1)"
                ),
                "path.write_text(updated, encoding='utf-8')",
            ]
        )
        return _bash_spec(_python_command("\n".join(script_lines)), timeout=120_000)

    if function_name == "Write":
        file_path, content = _resolve_write_fallbacks(tool)
        if file_path is None or content is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved Write payload")
        content = _normalize_runtime_specific_file_content(file_path=file_path, content=content)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        parent_dir = str(Path(file_path).parent) or "."
        bash_command = (
            f"mkdir -p {_shell_quote(parent_dir)} && "
            f"printf '%s' {_shell_quote(encoded)} | base64 -d > {_shell_quote(file_path)}"
        )
        return _bash_spec(bash_command, timeout=120_000)

    if function_name == "Glob":
        pattern = _coerce_string(args.get("pattern"))
        if pattern is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved Glob pattern")
        path = _coerce_string(args.get("path")) or "."
        script = (
            "import glob, os, sys;"
            f"path={path!r};"
            f"pattern={pattern!r};"
            "full_pattern = os.path.join(path, pattern) if path else pattern;"
            "matches = sorted(glob.glob(full_pattern, recursive=True));"
            "sys.stdout.write('\\n'.join(matches))"
        )
        return _bash_spec(f"python3 -c {_shell_quote(script)}", timeout=120_000)

    if function_name == "Grep":
        pattern = _coerce_string(args.get("pattern"))
        if pattern is None:
            return _build_noop_bash(function_name, reason="replay skipped unresolved Grep pattern")
        path = _coerce_string(args.get("path")) or "."
        parts = ["grep", "-R"]
        if args.get("output_mode") == "files_with_matches":
            parts.append("-l")
        else:
            parts.append("-n")
        if args.get("-i"):
            parts.append("-i")
        context = _coerce_int(args.get("context"))
        if context is None:
            context = _coerce_int(args.get("-C"))
        if context is not None:
            parts.extend(["-C", str(context)])
        glob_value = args.get("glob")
        if isinstance(glob_value, str) and glob_value:
            parts.append(f"--include={_shell_quote(glob_value)}")
        parts.append(_shell_quote(pattern))
        parts.append(_shell_quote(path))
        command = " ".join(parts) + " 2>/dev/null"
        head_limit = _coerce_int(args.get("head_limit"))
        if head_limit is not None and head_limit > 0:
            command += f" | head -n {head_limit}"
        return _bash_spec(command, timeout=120_000)

    logger.warning("Unknown Claude Code trace tool %r, replaying as no-op Bash", function_name)
    return _build_noop_bash(function_name, reason=f"replay skipped unknown tool {function_name}")


def _tool_strict_errors(tool: dict[str, Any], *, step_index: int) -> list[str]:
    function_name = tool["function_name"]
    if function_name in _STRICT_NON_REPLAYABLE_SKIP_TOOLS:
        return [f"step {step_index}: {function_name} tool side effects are not strictly replayable"]
    if function_name in _SKIP_TOOLS or function_name in {"Glob", "Grep"}:
        return []
    errors: list[str] = []
    if function_name == "Bash":
        command = _coerce_string(tool["arguments"].get("command"))
        if command is None:
            errors.append(f"step {step_index}: unresolved Bash command")
        return errors
    if function_name == "Read":
        file_path, _, _ = _resolve_read_fallbacks(tool)
        if file_path is None:
            errors.append(f"step {step_index}: unresolved Read file_path")
        return errors
    if function_name == "Write":
        file_path, content = _resolve_write_fallbacks(tool)
        if file_path is None:
            errors.append(f"step {step_index}: unresolved Write file_path")
        if content is None:
            errors.append(f"step {step_index}: unresolved Write content")
        return errors
    if function_name == "Edit":
        file_path, old_string, new_string = _resolve_edit_fallbacks(tool)
        if file_path is None:
            errors.append(f"step {step_index}: unresolved Edit file_path")
        if old_string is None:
            errors.append(f"step {step_index}: unresolved Edit old_string")
        if new_string is None:
            errors.append(f"step {step_index}: unresolved Edit new_string")
        return errors
    return []


def _step_to_content_blocks(step: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for text in step.get("text_segments", []):
        if isinstance(text, str) and text.strip():
            blocks.append({"type": "text", "text": text})

    for tool in step.get("tools", []):
        if not isinstance(tool, dict):
            continue
        replay_spec = _tool_to_replay_spec(tool)
        if replay_spec is None:
            continue
        tool_name, tool_input = replay_spec
        tool_use_id = _coerce_string(
            tool.get("tool_call_id"),
            allow_empty=True,
            allow_placeholder=True,
        ) or f"toolu_{uuid.uuid4().hex[:24]}"
        blocks.append(
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": tool_input,
            }
        )

    if not blocks:
        blocks.append({"type": "text", "text": ""})

    return blocks


def _duplicate_catchup_content_blocks(step: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for text in step.get("text_segments", []):
        if isinstance(text, str) and text.strip():
            blocks.append({"type": "text", "text": text})

    tools = step.get("tools", [])
    if isinstance(tools, list):
        for tool in tools:
            replay_spec = _duplicate_tool_replay_spec(tool)
            if replay_spec is None:
                continue
            tool_name, tool_input = replay_spec
            blocks.append(
                {
                    "type": "tool_use",
                    "id": f"toolu_dup_{uuid.uuid4().hex[:20]}",
                    "name": tool_name,
                    "input": tool_input,
                }
            )

    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _duplicate_tool_replay_spec(tool: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(tool, dict):
        return None
    return _tool_to_replay_spec(tool, duplicate_catchup=True)


def _anthropic_response_from_content_blocks(
    content_blocks: list[dict[str, Any]],
    *,
    model_name: str,
    is_last: bool,
    is_streaming: bool,
) -> dict[str, Any]:
    has_tool_use = any(block["type"] == "tool_use" for block in content_blocks)
    stop_reason = "end_turn" if is_last or not has_tool_use else "tool_use"
    response = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model_name or _DEFAULT_MODEL_NAME,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
    if is_streaming:
        response["_streaming"] = True
    return response


def _anthropic_duplicate_response(
    step: dict[str, Any],
    *,
    model_name: str,
    is_streaming: bool,
) -> dict[str, Any]:
    return _anthropic_response_from_content_blocks(
        _duplicate_catchup_content_blocks(step),
        model_name=model_name,
        is_last=False,
        is_streaming=is_streaming,
    )


def _anthropic_response(
    step: dict[str, Any],
    *,
    model_name: str,
    trace_index: int,
    is_last: bool,
    is_streaming: bool,
) -> dict[str, Any]:
    del trace_index
    return _anthropic_response_from_content_blocks(
        _step_to_content_blocks(step),
        model_name=model_name,
        is_last=is_last,
        is_streaming=is_streaming,
    )


def _end_turn_response(*, model_name: str, is_streaming: bool) -> dict[str, Any]:
    response = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "Task complete."}],
        "model": model_name or _DEFAULT_MODEL_NAME,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
    if is_streaming:
        response["_streaming"] = True
    return response


def _helper_model_response(*, model_name: str | None, is_streaming: bool) -> dict[str, Any]:
    response = {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
        "model": model_name or _DEFAULT_MODEL_NAME,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    }
    if is_streaming:
        response["_streaming"] = True
    return response

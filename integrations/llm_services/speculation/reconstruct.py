"""Reconstruct the per-turn LLM input from a recorded trajectory.

The replay services in ``terminus_trace_replay`` and ``mini_swe_trace_replay``
already parse the *responses* out of trajectories. For collecting real-draft
speculative decisions we additionally need the *inputs* — the chat-message list
the original model saw immediately before each recorded assistant turn.

Terminus trajectories store agent steps with an ``observation`` field that
carries the terminal output the harness fed back into the next turn. Mini-SWE
trajectories already serialise the full chat history, so reconstruction is
just a slice.

Claude Code trajectories use the same per-step shape as Terminus (a list of
``steps`` with ``source`` and per-step ``observation``) but split a single
model response across multiple steps: a text-only "thinking" step is
typically followed by a tool-call step. The Claude Code reconstructor
coalesces a thinking step (or several consecutive ones) with the immediately
following tool-call step into ONE turn — that is the unit a draft model is
asked to predict.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from integrations.llm_services.speculation.claude_code_tools import (
    CLAUDE_CODE_SYSTEM_PROMPT,
)

from datetime import datetime

from integrations.llm_services.terminus_trace_replay.service import (
    _build_terminus_response_json,
)


def _parse_iso_timestamp(value: object) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(raw).timestamp()
    except ValueError:
        return None


def _mini_swe_message_timestamp(message: dict[str, Any]) -> float | None:
    extra = message.get("extra")
    if not isinstance(extra, dict):
        return None
    timestamp = extra.get("timestamp")
    if not isinstance(timestamp, (int, float)):
        return None
    return float(timestamp)


@dataclass(frozen=True)
class ReconstructedTurn:
    """One reconstructable assistant turn from a trajectory.

    ``input_messages`` is the chat-completions ``messages`` payload the model
    saw immediately before producing ``oracle_response_content``. For terminus
    traces ``input_messages`` is a flat user/assistant alternation (the
    initial user message bakes the system instructions in). For mini_swe
    traces the first message is a real ``system`` role.

    ``oracle_latency_ms`` is the wall-clock the original recorded model spent
    producing this assistant turn (timestamp delta between the previous step
    and this one). It is ``None`` for the first turn (no prior timestamp to
    diff against) or when the trajectory lacks usable timestamps.
    """

    turn_index: int
    input_messages: list[dict[str, str]]
    oracle_response_content: str
    oracle_latency_ms: float | None = None


def reconstruct_terminus_turns(trace_path: Path) -> list[ReconstructedTurn]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"terminus trace {trace_path} did not decode to an object")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"terminus trace {trace_path} is missing a non-empty steps list")

    initial_user: str | None = None
    prefix: list[dict[str, str]] = []
    turns: list[ReconstructedTurn] = []
    pending_observation: str | None = None
    previous_timestamp: float | None = None
    agent_index = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        source = str(step.get("source", "")).lower()
        timestamp = _parse_iso_timestamp(step.get("timestamp"))
        if source == "user" and initial_user is None:
            raw_message = step.get("message")
            if isinstance(raw_message, str) and raw_message.strip():
                initial_user = raw_message
                prefix = [{"role": "user", "content": initial_user}]
            if timestamp is not None:
                previous_timestamp = timestamp
            continue
        if source != "agent":
            if timestamp is not None:
                previous_timestamp = timestamp
            continue
        if initial_user is None:
            raise ValueError(
                f"terminus trace {trace_path} has an agent step before any initial user message"
            )

        # Apply observation from the prior agent step as the user message that
        # preceded *this* one. The very first agent step has no preceding
        # observation — the initial user message is its only input.
        input_messages = list(prefix)
        if pending_observation:
            input_messages.append({"role": "user", "content": pending_observation})

        response_content, _ = _build_terminus_response_json(step)
        latency_ms: float | None = None
        if timestamp is not None and previous_timestamp is not None:
            latency_ms = max(0.0, (timestamp - previous_timestamp) * 1000.0)
        turns.append(
            ReconstructedTurn(
                turn_index=agent_index,
                input_messages=input_messages,
                oracle_response_content=response_content,
                oracle_latency_ms=latency_ms,
            )
        )

        # Advance the prefix: append this turn's user-observation pair so the
        # next turn sees it.
        if pending_observation:
            prefix.append({"role": "user", "content": pending_observation})
        prefix.append({"role": "assistant", "content": response_content})
        pending_observation = _terminus_observation_text(step)
        if timestamp is not None:
            previous_timestamp = timestamp
        agent_index += 1

    if not turns:
        raise ValueError(f"terminus trace {trace_path} has no replayable assistant turns")
    return turns


def _terminus_observation_text(step: dict[str, Any]) -> str | None:
    observation = step.get("observation")
    if not isinstance(observation, dict):
        return None
    results = observation.get("results")
    if not isinstance(results, list):
        return None
    chunks: list[str] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if isinstance(content, str) and content:
            chunks.append(content)
    if not chunks:
        return None
    return "\n".join(chunks)


def reconstruct_mini_swe_turns(trace_path: Path) -> list[ReconstructedTurn]:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"mini-swe trace {trace_path} did not decode to an object")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError(f"mini-swe trace {trace_path} is missing a non-empty messages list")

    prefix: list[dict[str, str]] = []
    turns: list[ReconstructedTurn] = []
    previous_timestamp: float | None = None
    agent_index = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        timestamp = _mini_swe_message_timestamp(message)
        if role == "assistant":
            if not content.strip():
                # Skip empty assistant turns — replay parser does the same.
                if timestamp is not None:
                    previous_timestamp = timestamp
                continue
            latency_ms: float | None = None
            if timestamp is not None and previous_timestamp is not None:
                latency_ms = max(0.0, (timestamp - previous_timestamp) * 1000.0)
            turns.append(
                ReconstructedTurn(
                    turn_index=agent_index,
                    input_messages=list(prefix),
                    oracle_response_content=content,
                    oracle_latency_ms=latency_ms,
                )
            )
            agent_index += 1
            prefix.append({"role": "assistant", "content": content})
        else:
            prefix.append({"role": role, "content": content})
        if timestamp is not None:
            previous_timestamp = timestamp

    if not turns:
        raise ValueError(f"mini-swe trace {trace_path} has no replayable assistant turns")
    return turns


# ---------------------------------------------------------------------------
# Claude Code


def reconstruct_claude_code_turns(trace_path: Path) -> list[ReconstructedTurn]:
    """Reconstruct per-turn LLM inputs for a Claude Code trajectory.

    Claude Code splits one assistant response across multiple ``steps``:
    text-only "thinking" content blocks followed by a single tool-use block
    (or, for the last assistant message of a session, a final text block
    with no tool call). This reconstructor groups consecutive thinking-only
    steps with the next tool-call step into ONE turn — the unit the draft
    model is asked to predict.

    Each turn's ``oracle_response_content`` is a JSON envelope of the shape::

        {"thinking": "<accumulated text>",
         "tool_calls": [{"name": "Bash", "arguments": {"command": "..."}}]}

    For a final response (no tool call after the last thinking blocks), the
    ``tool_calls`` array is empty.

    The ``input_messages`` payload is OpenAI-compatible: ``system`` +
    ``user`` initial task + alternating ``assistant`` (with ``tool_calls``)
    and ``tool`` (with ``tool_call_id``+result) messages.
    """
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"claude_code trace {trace_path} did not decode to an object")
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"claude_code trace {trace_path} is missing a non-empty steps list")

    initial_user: str | None = None
    prefix: list[dict[str, Any]] = [
        {"role": "system", "content": CLAUDE_CODE_SYSTEM_PROMPT}
    ]
    turns: list[ReconstructedTurn] = []
    pending_thinking: list[str] = []
    pending_thinking_first_ts: float | None = None
    previous_timestamp: float | None = None
    agent_index = 0

    def _emit_turn(tool_calls: list[dict[str, Any]] | None, end_timestamp: float | None) -> None:
        """Commit pending thinking + an optional tool call as one turn."""
        nonlocal agent_index, pending_thinking, pending_thinking_first_ts
        thinking_text = "\n".join(t for t in pending_thinking if t).strip()
        envelope = {
            "thinking": thinking_text,
            "tool_calls": _normalize_oracle_tool_calls(tool_calls or []),
        }
        oracle_content = json.dumps(envelope, ensure_ascii=False)
        latency_ms: float | None = None
        # Latency = (end timestamp of the action step) - (timestamp of the
        # previous step that fed this turn — i.e. the prior tool result or
        # the initial user message).
        if end_timestamp is not None and previous_timestamp is not None:
            latency_ms = max(0.0, (end_timestamp - previous_timestamp) * 1000.0)
        turns.append(
            ReconstructedTurn(
                turn_index=agent_index,
                input_messages=copy.deepcopy(prefix),
                oracle_response_content=oracle_content,
                oracle_latency_ms=latency_ms,
            )
        )
        agent_index += 1
        pending_thinking = []
        pending_thinking_first_ts = None

    for step in steps:
        if not isinstance(step, dict):
            continue
        source = str(step.get("source", "")).lower()
        timestamp = _parse_iso_timestamp(step.get("timestamp"))

        if source == "user":
            if initial_user is None:
                raw_message = step.get("message")
                if isinstance(raw_message, str) and raw_message.strip():
                    initial_user = raw_message
                    prefix.append({"role": "user", "content": initial_user})
            else:
                # A second user step inside the trajectory is rare but real
                # (the human nudges the agent mid-run). Flush any pending
                # thinking as a final-text turn first, then add the new
                # user message as input for the next turn.
                if pending_thinking:
                    _emit_turn(tool_calls=None, end_timestamp=previous_timestamp)
                raw_message = step.get("message")
                if isinstance(raw_message, str) and raw_message:
                    prefix.append({"role": "user", "content": raw_message})
            if timestamp is not None:
                previous_timestamp = timestamp
            continue

        if source != "agent":
            if timestamp is not None:
                previous_timestamp = timestamp
            continue

        if initial_user is None:
            raise ValueError(
                f"claude_code trace {trace_path} has an agent step before any initial user message"
            )

        tool_calls = step.get("tool_calls") or []
        message = step.get("message") if isinstance(step.get("message"), str) else ""

        if not tool_calls:
            # Pure thinking / text content — accumulate until the next
            # tool-call step or end-of-trace.
            if message:
                pending_thinking.append(message)
            if pending_thinking_first_ts is None and timestamp is not None:
                pending_thinking_first_ts = timestamp
            continue

        # We have a tool-call step. Commit a turn with whatever thinking
        # accumulated plus this tool call, then advance the prefix with the
        # assistant message (carrying tool_calls) and the tool result.
        _emit_turn(tool_calls=tool_calls, end_timestamp=timestamp)
        prefix.append(_assistant_message_with_tool_calls(message, tool_calls))
        tool_messages = _tool_result_messages(step, tool_calls)
        prefix.extend(tool_messages)
        if timestamp is not None:
            previous_timestamp = timestamp

    # Trailing thinking with no tool call = final-response turn.
    if pending_thinking:
        _emit_turn(tool_calls=None, end_timestamp=previous_timestamp)

    if not turns:
        raise ValueError(
            f"claude_code trace {trace_path} has no replayable assistant turns"
        )
    return turns


def _normalize_oracle_tool_calls(
    tool_calls: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pull out (name, arguments) pairs from a recorded step's tool_calls."""
    normalized: list[dict[str, Any]] = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("function_name")
        if not isinstance(name, str):
            continue
        arguments = tc.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        normalized.append({"name": name, "arguments": arguments})
    return normalized


def _assistant_message_with_tool_calls(
    thinking_text: str, tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build an OpenAI-format assistant message carrying tool_calls."""
    api_tool_calls = []
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        name = tc.get("function_name")
        if not isinstance(name, str):
            continue
        arguments = tc.get("arguments") if isinstance(tc.get("arguments"), dict) else {}
        api_tool_calls.append(
            {
                "id": tc.get("tool_call_id") or _synthesize_call_id(),
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        )
    message: dict[str, Any] = {"role": "assistant"}
    # OpenAI lets content be null when tool_calls is present, but DeepSeek
    # tolerates an empty string and that round-trips through JSON encoders
    # more cleanly.
    message["content"] = thinking_text or ""
    if api_tool_calls:
        message["tool_calls"] = api_tool_calls
    return message


def _tool_result_messages(
    step: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Translate a step's observation into one OpenAI ``tool`` message per call."""
    observation = step.get("observation")
    results: list[dict[str, Any]] = []
    if isinstance(observation, dict):
        raw_results = observation.get("results")
        if isinstance(raw_results, list):
            for r in raw_results:
                if isinstance(r, dict):
                    results.append(r)
    messages: list[dict[str, Any]] = []
    by_call_id: dict[str, str] = {}
    for r in results:
        cid = r.get("source_call_id")
        content = r.get("content")
        if isinstance(cid, str) and isinstance(content, str):
            by_call_id[cid] = content
    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        cid = tc.get("tool_call_id")
        if not isinstance(cid, str):
            continue
        content = by_call_id.get(cid, "")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": cid,
                "content": content,
            }
        )
    return messages


_SYNTH_CALL_ID_COUNTER = [0]


def _synthesize_call_id() -> str:
    _SYNTH_CALL_ID_COUNTER[0] += 1
    return f"crab_synth_{_SYNTH_CALL_ID_COUNTER[0]}"

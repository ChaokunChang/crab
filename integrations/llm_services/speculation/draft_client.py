"""Minimal OpenAI-compatible chat-completions client for draft models.

Works with any endpoint that accepts a POST to ``/chat/completions`` with the
OpenAI request body shape: ``{"model": ..., "messages": [...], **params}``.
This includes DeepSeek (``https://api.deepseek.com``), OpenRouter, vLLM/SGLang
servers, Ollama's OpenAI shim, and Together / Fireworks compatibility
endpoints.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class DraftResult:
    """Outcome of one draft-model chat-completions call.

    ``content`` is always a string. For tool-calling responses (where the API
    returns ``message.content == null`` and a populated ``message.tool_calls``)
    we serialise the tool call(s) into a JSON envelope so downstream scoring
    can treat every response uniformly:

        {"thinking": "<text the model emitted before the tool call, if any>",
         "tool_calls": [{"name": "Bash", "arguments": {"command": "ls"}}, ...]}

    For plain text responses ``content`` is the raw text (no envelope).
    """

    content: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


class DraftRequestError(RuntimeError):
    """Raised when the draft endpoint returns an unrecoverable error."""


class OpenAICompatibleDraftClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        model: str,
        params: dict[str, Any] | None = None,
        timeout_s: float = 120.0,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if not model:
            raise ValueError("model is required")
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._params = dict(params or {})
        self._timeout_s = float(timeout_s)

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
    ) -> DraftResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            **self._params,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_s) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise DraftRequestError(
                f"draft endpoint returned HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise DraftRequestError(f"draft endpoint unreachable: {exc.reason}") from exc
        elapsed_ms = (time.monotonic() - started) * 1000.0

        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise DraftRequestError(
                f"draft endpoint returned non-JSON body: {body[:500]!r}"
            ) from exc

        choices = decoded.get("choices")
        if not isinstance(choices, list) or not choices:
            raise DraftRequestError(f"draft response had no choices: {body[:500]}")
        first = choices[0]
        if not isinstance(first, dict):
            raise DraftRequestError("draft response choice was not an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise DraftRequestError("draft response choice had no message object")

        raw_content = message.get("content")
        thinking_text = raw_content if isinstance(raw_content, str) else ""
        tool_calls_raw = message.get("tool_calls")

        normalized_calls: list[dict[str, Any]] = []
        if isinstance(tool_calls_raw, list):
            for tc in tool_calls_raw:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
                name = fn.get("name") if isinstance(fn.get("name"), str) else ""
                args_raw = fn.get("arguments")
                if isinstance(args_raw, str):
                    try:
                        arguments = json.loads(args_raw) if args_raw.strip() else {}
                    except json.JSONDecodeError:
                        arguments = {"_raw_arguments": args_raw}
                elif isinstance(args_raw, dict):
                    arguments = args_raw
                else:
                    arguments = {}
                normalized_calls.append({"name": name, "arguments": arguments})

        if tools:
            # In tools-enabled mode, always emit the structured envelope so
            # downstream scoring can distinguish "no tool call → final
            # response" (tool_calls=[]) from "extractor failed to parse".
            envelope = {
                "thinking": thinking_text,
                "tool_calls": normalized_calls,
            }
            content = json.dumps(envelope, ensure_ascii=False)
        elif normalized_calls:
            envelope = {
                "thinking": thinking_text,
                "tool_calls": normalized_calls,
            }
            content = json.dumps(envelope, ensure_ascii=False)
        elif isinstance(raw_content, str):
            content = raw_content
        else:
            # Some endpoints return null content + no tool calls when the
            # model could not produce a structured response. Surface this
            # as an empty string so scoring records a reject rather than
            # erroring out.
            content = ""

        usage = decoded.get("usage") if isinstance(decoded.get("usage"), dict) else {}
        return DraftResult(
            content=content,
            latency_ms=elapsed_ms,
            prompt_tokens=_optional_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_int(usage.get("completion_tokens")),
        )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

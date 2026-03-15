from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .tool_catalog import TOOL_DEFINITIONS, allowed_tool_names, default_input_for_tool


class SimulatedLLMState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._turns: dict[tuple[str, str], int] = defaultdict(int)

    def next_turn(self, sandbox_id: str, provider: str) -> int:
        with self._lock:
            key = (sandbox_id, provider)
            value = self._turns[key]
            self._turns[key] += 1
            return value

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {f"{sandbox_id}:{provider}": turn for (sandbox_id, provider), turn in self._turns.items()}


def _sandbox_id_from_request(headers, payload: dict[str, Any]) -> str:
    sandbox_id = headers.get("X-Agent-Sandbox-Id", "").strip()
    if sandbox_id:
        return sandbox_id
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("sandbox_id"), str):
        return metadata["sandbox_id"]
    return "sandbox-unknown"


def choose_tool(provider: str, headers, payload: dict[str, Any], state: SimulatedLLMState) -> tuple[str, int]:
    sandbox_id = _sandbox_id_from_request(headers, payload)
    turn_index = state.next_turn(sandbox_id, provider)
    allowed = allowed_tool_names(provider, payload)
    names = allowed or [tool.name for tool in TOOL_DEFINITIONS]
    name = names[turn_index % len(names)]
    return name, turn_index


def build_openai_response(tool_name: str, turn_index: int) -> dict[str, Any]:
    arguments = default_input_for_tool(tool_name, turn_index)
    return {
        "id": f"chatcmpl-sim-{turn_index}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "simulated-openai",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments, sort_keys=True),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def build_anthropic_response(tool_name: str, turn_index: int) -> dict[str, Any]:
    arguments = default_input_for_tool(tool_name, turn_index)
    return {
        "id": f"msg_sim_{turn_index}",
        "type": "message",
        "role": "assistant",
        "model": "simulated-anthropic",
        "content": [
            {
                "type": "tool_use",
                "id": f"toolu_{uuid.uuid4().hex[:12]}",
                "name": tool_name,
                "input": arguments,
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


class SimulatedLLMHandler(BaseHTTPRequestHandler):
    state = SimulatedLLMState()
    response_delay_ms = 0

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_error(404)
            return
        body = b'{"ok":true}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path not in {"/v1/chat/completions", "/v1/messages"}:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        response = handle_request(
            path=self.path,
            headers=dict(self.headers.items()),
            payload=payload,
            state=self.state,
            response_delay_ms=self.response_delay_ms,
        )
        body = json.dumps(response, sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        _ = (format, args)
        return


def serve(*, host: str, port: int, response_delay_ms: int) -> ThreadingHTTPServer:
    SimulatedLLMHandler.response_delay_ms = response_delay_ms
    server = ThreadingHTTPServer((host, port), SimulatedLLMHandler)
    return server


def handle_request(
    *,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    state: SimulatedLLMState | None = None,
    response_delay_ms: int = 0,
) -> dict[str, Any]:
    provider = "openai" if path == "/v1/chat/completions" else "anthropic"
    llm_state = state or SimulatedLLMState()
    tool_name, turn_index = choose_tool(provider, headers, payload, llm_state)
    if response_delay_ms > 0:
        time.sleep(response_delay_ms / 1000.0)
    if provider == "openai":
        return build_openai_response(tool_name, turn_index)
    return build_anthropic_response(tool_name, turn_index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--response-delay-ms", type=int, default=500)
    args = parser.parse_args()

    server = serve(host=args.host, port=args.port, response_delay_ms=args.response_delay_ms)
    server.serve_forever()


if __name__ == "__main__":
    main()

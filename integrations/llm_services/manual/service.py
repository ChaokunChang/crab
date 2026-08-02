from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict, deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def _sandbox_id_from_request(headers: dict[str, str], payload: dict[str, Any]) -> str:
    sandbox_id = headers.get("X-Agent-Sandbox-Id", "").strip()
    if sandbox_id:
        return sandbox_id
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("sandbox_id"), str):
        return metadata["sandbox_id"]
    return "sandbox-unknown"


def _require_sandbox_id_from_request(headers: dict[str, str], payload: dict[str, Any]) -> str:
    sandbox_id = _sandbox_id_from_request(headers, payload).strip()
    if sandbox_id and sandbox_id != "sandbox-unknown":
        return sandbox_id
    raise ValueError("missing sandbox identity")


def _tool_response(*, content: str, turn: int, tool_name: str | None = None, tool_input: dict[str, Any] | None = None) -> dict[str, Any]:
    if tool_name is None:
        return {
            "id": f"chatcmpl-iflow-final-{turn}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "crab-iflow-manual",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    return {
        "id": f"chatcmpl-iflow-{turn}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "crab-iflow-manual",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(tool_input or {}, sort_keys=True),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _response_for_turn(response: dict[str, Any], *, turn: int) -> dict[str, Any]:
    cloned = json.loads(json.dumps(response))
    cloned["created"] = int(time.time())
    cloned["id"] = f"chatcmpl-manual-{turn}-{uuid.uuid4().hex[:8]}"
    return cloned


class ManualLLMState:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._turns: dict[str, int] = defaultdict(int)
        self._events: list[dict[str, Any]] = []
        self._queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)

    def enqueue_run_shell_command(
        self,
        *,
        command: str,
        sandbox_id: str,
        content: str | None = None,
        response_delay_ms: int = 0,
    ) -> dict[str, Any]:
        return self._enqueue_step(
            sandbox_id=sandbox_id,
            phase="manual_run_shell_command",
            response_delay_ms=response_delay_ms,
            response=_tool_response(
                content=content or "Run the requested shell command and report the result.",
                turn=0,
                tool_name="run_shell_command",
                tool_input={"command": command},
            ),
        )

    def enqueue_final_response(
        self,
        *,
        content: str,
        sandbox_id: str,
        response_delay_ms: int = 0,
    ) -> dict[str, Any]:
        return self._enqueue_step(
            sandbox_id=sandbox_id,
            phase="manual_final_response",
            response_delay_ms=response_delay_ms,
            response=_tool_response(content=content, turn=0),
        )

    def next_response(
        self,
        *,
        path: str,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if path != "/v1/chat/completions":
            raise ValueError(f"unsupported path: {path}")
        sandbox_id = _require_sandbox_id_from_request(headers, payload)
        with self._condition:
            turn = self._turns[sandbox_id]
            self._turns[sandbox_id] += 1
            self._events.append(
                {
                    "event": "request",
                    "path": path,
                    "sandbox_id": sandbox_id,
                    "turn": turn,
                    "phase": "manual_wait",
                    "headers": headers,
                    "payload": payload,
                }
            )
            while not self._queues[sandbox_id]:
                self._condition.wait(timeout=0.2)
            item = self._queues[sandbox_id].popleft()
        response = _response_for_turn(item["response"], turn=turn)
        if int(item["response_delay_ms"]) > 0:
            time.sleep(int(item["response_delay_ms"]) / 1000.0)
        with self._condition:
            self._events.append(
                {
                    "event": "response",
                    "sandbox_id": sandbox_id,
                    "turn": turn,
                    "phase": item["phase"],
                    "response": response,
                }
            )
        return response

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "events": list(self._events),
                "queue_depths": {sandbox_id: len(queue) for sandbox_id, queue in self._queues.items()},
                "turns": dict(self._turns),
            }

    def _enqueue_step(
        self,
        *,
        sandbox_id: str,
        phase: str,
        response_delay_ms: int,
        response: dict[str, Any],
    ) -> dict[str, Any]:
        target_sandbox_id = sandbox_id.strip()
        if not target_sandbox_id:
            raise ValueError("sandbox_id is required")
        with self._condition:
            self._queues[target_sandbox_id].append(
                {
                    "phase": phase,
                    "response_delay_ms": int(response_delay_ms),
                    "response": response,
                }
            )
            self._events.append(
                {
                    "event": "control_enqueue",
                    "sandbox_id": target_sandbox_id,
                    "phase": phase,
                    "response_delay_ms": int(response_delay_ms),
                    "queue_depth": len(self._queues[target_sandbox_id]),
                }
            )
            self._condition.notify_all()
            return {
                "sandbox_id": target_sandbox_id,
                "phase": phase,
                "queue_depth": len(self._queues[target_sandbox_id]),
            }


def handle_control_request(*, path: str, payload: dict[str, Any], state: ManualLLMState) -> dict[str, Any]:
    if path == "/control/run_shell_command":
        result = state.enqueue_run_shell_command(
            command=str(payload["command"]),
            sandbox_id=str(payload["sandbox_id"]),
            content=None if payload.get("content") is None else str(payload["content"]),
            response_delay_ms=int(payload.get("response_delay_ms", 0)),
        )
        return {"ok": True, "result": result}
    if path == "/control/final_response":
        result = state.enqueue_final_response(
            content=str(payload["content"]),
            sandbox_id=str(payload["sandbox_id"]),
            response_delay_ms=int(payload.get("response_delay_ms", 0)),
        )
        return {"ok": True, "result": result}
    raise ValueError(f"unsupported control path: {path}")


def serve_manual(*, host: str, port: int) -> ThreadingHTTPServer:
    state = ManualLLMState()

    class ManualLLMHandler(BaseHTTPRequestHandler):
        manual_state = state

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _write_json(self, payload: dict[str, Any], *, code: int = 200) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                snapshot = self.manual_state.snapshot()
                self._write_json({"ok": True, "events": len(snapshot["events"])})
                return
            if self.path == "/control/state":
                self._write_json({"ok": True, "state": self.manual_state.snapshot()})
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            payload = self._read_json()
            try:
                if self.path == "/v1/chat/completions":
                    response = self.manual_state.next_response(
                        path=self.path,
                        headers=dict(self.headers.items()),
                        payload=payload,
                    )
                    self._write_json(response)
                    return
                response = handle_control_request(path=self.path, payload=payload, state=self.manual_state)
                self._write_json(response)
            except (KeyError, ValueError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, format: str, *args) -> None:
            _ = (format, args)
            return

    server = ThreadingHTTPServer((host, port), ManualLLMHandler)
    server.manual_state = state  # type: ignore[attr-defined]
    return server

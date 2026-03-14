from __future__ import annotations

import argparse
import json
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass(frozen=True)
class ScriptStep:
    phase: str
    content: str
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    response_delay_ms: int = 0
    expected_process_changed: bool | None = None
    expected_filesystem_changed: bool | None = None


def default_script_steps(*, idle_delay_ms: int = 2000) -> list[ScriptStep]:
    return [
        ScriptStep(
            phase="transient_process",
            content="Run a short no-op shell command and report completion.",
            tool_name="run_shell_command",
            tool_input={"command": "sh -lc \"printf noop >/dev/null\""},
            response_delay_ms=idle_delay_ms,
            expected_process_changed=False,
            expected_filesystem_changed=False,
        ),
        ScriptStep(
            phase="filesystem_write",
            content="Create a deterministic filesystem artifact for verification.",
            tool_name="run_shell_command",
            tool_input={
                "command": "sh -lc \"mkdir -p /work/iflow-probe && printf iflow-artifact >/work/iflow-probe/artifact.txt\""
            },
            response_delay_ms=idle_delay_ms,
            expected_process_changed=False,
            expected_filesystem_changed=True,
        ),
        ScriptStep(
            phase="detached_daemon",
            content="Start a detached HTTP daemon and persist its PID for validation.",
            tool_name="run_shell_command",
            tool_input={
                "command": (
                    "sh -lc \"mkdir -p /work/iflow-probe && "
                    "python3 -m http.server 8123 >/work/iflow-probe/http.log 2>&1 & "
                    "echo $! >/work/iflow-probe/http.pid\""
                )
            },
            response_delay_ms=idle_delay_ms,
            expected_process_changed=True,
            expected_filesystem_changed=True,
        ),
        ScriptStep(
            phase="final_response",
            content="The verification scenario is complete. Summarize what you observed and stop.",
            response_delay_ms=idle_delay_ms,
            expected_process_changed=None,
            expected_filesystem_changed=None,
        ),
    ]


class ScriptedLLMState:
    def __init__(self, steps: list[ScriptStep] | None = None) -> None:
        self._lock = threading.Lock()
        self._steps = list(steps or default_script_steps())
        self._turns: dict[str, int] = defaultdict(int)
        self._events: list[dict[str, Any]] = []

    def next_step(self, sandbox_id: str) -> tuple[int, ScriptStep]:
        with self._lock:
            turn = self._turns[sandbox_id]
            if turn >= len(self._steps):
                step = self._steps[-1]
            else:
                step = self._steps[turn]
            self._turns[sandbox_id] += 1
            return turn, step

    def record(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(dict(payload))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "steps": [asdict(step) for step in self._steps],
                "events": list(self._events),
                "turns": dict(self._turns),
            }


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
                ScriptStep(
                    phase="manual_run_shell_command",
                    content=content or "Run the requested shell command and report the result.",
                    tool_name="run_shell_command",
                    tool_input={"command": command},
                ),
                turn=0,
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
            response=_tool_response(
                ScriptStep(
                    phase="manual_final_response",
                    content=content,
                ),
                turn=0,
            ),
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


def _tool_response(step: ScriptStep, turn: int) -> dict[str, Any]:
    if step.tool_name is None:
        return {
            "id": f"chatcmpl-iflow-final-{turn}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "agent-cr-iflow-scripted",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": step.content,
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
        "model": "agent-cr-iflow-scripted",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": step.content,
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": step.tool_name,
                                "arguments": json.dumps(step.tool_input or {}, sort_keys=True),
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


def handle_request(
    *,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    state: ScriptedLLMState,
) -> dict[str, Any]:
    if path != "/v1/chat/completions":
        raise ValueError(f"unsupported path: {path}")
    sandbox_id = _sandbox_id_from_request(headers, payload)
    turn, step = state.next_step(sandbox_id)
    state.record(
        {
            "event": "request",
            "path": path,
            "sandbox_id": sandbox_id,
            "turn": turn,
            "phase": step.phase,
            "headers": headers,
            "payload": payload,
        }
    )
    if step.response_delay_ms > 0:
        time.sleep(step.response_delay_ms / 1000.0)
    response = _tool_response(step, turn)
    state.record(
        {
            "event": "response",
            "sandbox_id": sandbox_id,
            "turn": turn,
            "phase": step.phase,
            "response": response,
            "expected_process_changed": step.expected_process_changed,
            "expected_filesystem_changed": step.expected_filesystem_changed,
        }
    )
    return response


class ScriptedLLMHandler(BaseHTTPRequestHandler):
    state = ScriptedLLMState()

    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_error(404)
            return
        snapshot = self.state.snapshot()
        body = json.dumps({"ok": True, "events": len(snapshot["events"])}, sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
        response = handle_request(path=self.path, headers=dict(self.headers.items()), payload=payload, state=self.state)
        body = json.dumps(response, sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        _ = (format, args)
        return


def serve(*, host: str, port: int, steps: list[ScriptStep] | None = None) -> ThreadingHTTPServer:
    ScriptedLLMHandler.state = ScriptedLLMState(steps)
    server = ThreadingHTTPServer((host, port), ScriptedLLMHandler)
    server.scripted_state = ScriptedLLMHandler.state  # type: ignore[attr-defined]
    return server


def serve_manual(
    *,
    host: str,
    port: int,
) -> ThreadingHTTPServer:
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
                if self.path == "/control/run_shell_command":
                    result = self.manual_state.enqueue_run_shell_command(
                        command=str(payload["command"]),
                        sandbox_id=str(payload["sandbox_id"]),
                        content=None if payload.get("content") is None else str(payload["content"]),
                        response_delay_ms=int(payload.get("response_delay_ms", 0)),
                    )
                    self._write_json({"ok": True, "result": result})
                    return
                if self.path == "/control/final_response":
                    result = self.manual_state.enqueue_final_response(
                        content=str(payload["content"]),
                        sandbox_id=str(payload["sandbox_id"]),
                        response_delay_ms=int(payload.get("response_delay_ms", 0)),
                    )
                    self._write_json({"ok": True, "result": result})
                    return
                self.send_error(404)
            except (KeyError, ValueError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, format: str, *args) -> None:
            _ = (format, args)
            return

    server = ThreadingHTTPServer((host, port), ManualLLMHandler)
    server.manual_state = state  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--idle-delay-ms", type=int, default=2000)
    parser.add_argument("--manual", action="store_true")
    args = parser.parse_args()

    if args.manual:
        server = serve_manual(host=args.host, port=args.port)
    else:
        server = serve(host=args.host, port=args.port, steps=default_script_steps(idle_delay_ms=args.idle_delay_ms))
    server.serve_forever()


if __name__ == "__main__":
    main()

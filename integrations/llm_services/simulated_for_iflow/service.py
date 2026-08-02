from __future__ import annotations

import argparse
import json
import shlex
import threading
import time
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


DEFAULT_RESPONSE_DELAY_MS = 2000
DEFAULT_MAX_TOOL_CALLS_BEFORE_FINISH = 3

_MODEL_NAME = "crab-iflow-simulated"
_FINAL_RESPONSE_STEP = {
    "phase": "final_response",
    "content": "Done.",
    "tool_name": None,
    "tool_input": None,
}


def _shell_command(script: str) -> str:
    return f"sh -lc {shlex.quote(script)}"


_DEFAULT_RESPONSE_STEPS: tuple[dict[str, Any], ...] = (
    {
        "phase": "transient_process",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": 'sh -lc "printf transient-process && (sleep 1 & wait $!)"',
        },
    },
    {
        "phase": "filesystem_write",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": """sh -lc "mkdir -p /work/iflow-probe && cat <<'EOF' >/work/iflow-probe/artifact.txt
iflow artifact
phase=filesystem_write
EOF"
""".strip(),
        },
    },
    {
        "phase": "detached_daemon",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": 'sh -lc "mkdir -p /work/iflow-probe/site && printf simulator-http >/work/iflow-probe/site/index.txt && python3 -m http.server 8123 -d /work/iflow-probe/site >/work/iflow-probe/http.log 2>&1 & echo $! >/work/iflow-probe/http.pid"',
        },
    },
    {
        "phase": "workspace_inventory",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": 'sh -lc "printf workdir=%s\\\\n \\"$PWD\\" && find /work -maxdepth 2 -mindepth 1 | sort"',
        },
    },
    {
        "phase": "environment_probe",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": 'sh -lc "printf shell=%s\\\\n \\"${SHELL:-unknown}\\" && env | sort | sed -n \'1,12p\'"',
        },
    },
    {
        "phase": "stateful_counter_update",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": """sh -lc "mkdir -p /work/iflow-probe
counter=/work/iflow-probe/counter.txt
value=$(cat \"$counter\" 2>/dev/null || printf 0)
printf '%s\\n' \"$((value + 1))\" >\"$counter\"
printf counter-updated=%s\\n \"$((value + 1))\""
""".strip(),
        },
    },
    {
        "phase": "append_journal",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": """sh -lc "mkdir -p /work/iflow-probe
printf 'journal-entry %s\\n' \"$(date -u +%Y%m%dT%H%M%SZ)\" >>/work/iflow-probe/journal.log
tail -n 3 /work/iflow-probe/journal.log"
""".strip(),
        },
    },
    {
        "phase": "expensive_digest_scan",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": """sh -lc "find /app /work -maxdepth 4 -type f 2>/dev/null | sort | head -n 200 >/tmp/iflow-scan-files.txt
if [ -s /tmp/iflow-scan-files.txt ]; then
  xargs -r sha256sum </tmp/iflow-scan-files.txt | sha256sum
else
  printf 'no-files\\n'
fi"
""".strip(),
        },
    },
    {
        "phase": "expensive_stateful_blob",
        "content": None,
        "tool_name": "run_shell_command",
        "tool_input": {
            "command": """sh -lc "mkdir -p /work/iflow-probe
dd if=/dev/zero of=/work/iflow-probe/heavy-blob.bin bs=262144 count=8 status=none
sha256sum /work/iflow-probe/heavy-blob.bin >/work/iflow-probe/heavy-blob.bin.sha256
wc -c /work/iflow-probe/heavy-blob.bin"
""".strip(),
        },
    },
)


def _sandbox_id_from_request(headers: dict[str, str], payload: dict[str, Any]) -> str:
    sandbox_id = headers.get("X-Agent-Sandbox-Id", "").strip()
    if sandbox_id:
        return sandbox_id
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("sandbox_id"), str):
        return metadata["sandbox_id"]
    return "sandbox-unknown"


def _sender_key_from_request(headers: dict[str, str], payload: dict[str, Any]) -> str:
    return _sandbox_id_from_request(headers, payload)


def _allowed_tool_names(payload: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for item in payload.get("tools", []):
        function = dict(item.get("function", {}))
        name = function.get("name")
        if isinstance(name, str):
            names.append(name)
    return names


def _response_steps_for_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    allowed_names = set(_allowed_tool_names(payload))
    if not allowed_names:
        return list(_DEFAULT_RESPONSE_STEPS)
    filtered = [step for step in _DEFAULT_RESPONSE_STEPS if step["tool_name"] in allowed_names]
    return filtered or list(_DEFAULT_RESPONSE_STEPS)


def _finish_response(turn: int, *, content: str = "Done.") -> dict[str, Any]:
    return {
        "id": f"chatcmpl-iflow-final-{turn}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                    "tool_calls": [],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _tool_response(step: dict[str, Any], turn: int) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-iflow-{turn}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": step["content"],
                    "tool_calls": [
                        {
                            "id": f"call_{uuid.uuid4().hex[:12]}",
                            "type": "function",
                            "function": {
                                "name": step["tool_name"],
                                "arguments": json.dumps(step["tool_input"], sort_keys=True),
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class SimulatedLLMState:
    def __init__(
        self,
        *,
        response_delay_ms: int = DEFAULT_RESPONSE_DELAY_MS,
        max_tool_calls_before_finish: int = DEFAULT_MAX_TOOL_CALLS_BEFORE_FINISH,
    ) -> None:
        self._lock = threading.Lock()
        self._response_delay_ms = max(0, int(response_delay_ms))
        self._max_tool_calls_before_finish = max(1, int(max_tool_calls_before_finish))
        self._turns: dict[str, int] = defaultdict(int)
        self._next_step_index: dict[str, int] = defaultdict(int)
        self._tool_calls_since_finish: dict[str, int] = defaultdict(int)
        self._events: list[dict[str, Any]] = []

    @property
    def response_delay_ms(self) -> int:
        return self._response_delay_ms

    @property
    def max_tool_calls_before_finish(self) -> int:
        return self._max_tool_calls_before_finish

    def next_response(self, sender_key: str, step_count: int) -> tuple[int, str, int | None, int]:
        with self._lock:
            turn = self._turns[sender_key]
            self._turns[sender_key] += 1
            tool_calls_since_finish = self._tool_calls_since_finish[sender_key]
            if tool_calls_since_finish >= self._max_tool_calls_before_finish:
                self._tool_calls_since_finish[sender_key] = 0
                self._next_step_index[sender_key] = 0
                return turn, "finish", None, tool_calls_since_finish

            step_index = self._next_step_index[sender_key] % max(1, step_count)
            self._next_step_index[sender_key] = (step_index + 1) % max(1, step_count)
            self._tool_calls_since_finish[sender_key] = tool_calls_since_finish + 1
            return turn, "tool_call", step_index, self._tool_calls_since_finish[sender_key]

    def record(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._events.append(dict(payload))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            senders = {
                sender_key: {
                    "turn": self._turns[sender_key],
                    "next_tool_index": self._next_step_index[sender_key],
                    "tool_calls_since_finish": self._tool_calls_since_finish[sender_key],
                }
                for sender_key in sorted(set(self._turns) | set(self._next_step_index) | set(self._tool_calls_since_finish))
            }
            return {
                "events": list(self._events),
                "response_delay_ms": self._response_delay_ms,
                "max_tool_calls_before_finish": self._max_tool_calls_before_finish,
                "senders": senders,
            }


def handle_request(
    *,
    path: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    state: SimulatedLLMState,
) -> dict[str, Any]:
    if path != "/v1/chat/completions":
        raise ValueError(f"unsupported path: {path}")

    sender_key = _sender_key_from_request(headers, payload)
    sandbox_id = _sandbox_id_from_request(headers, payload)
    steps = _response_steps_for_payload(payload)
    turn, response_kind, step_index, tool_calls_since_finish = state.next_response(sender_key, len(steps))
    step = _FINAL_RESPONSE_STEP if response_kind == "finish" else steps[int(step_index)]

    state.record(
        {
            "event": "request",
            "path": path,
            "phase": step["phase"],
            "sandbox_id": sandbox_id,
            "sender_key": sender_key,
            "turn": turn,
            "response_kind": response_kind,
            "headers": headers,
            "payload": payload,
        }
    )
    if state.response_delay_ms > 0:
        time.sleep(state.response_delay_ms / 1000.0)

    if response_kind == "finish":
        response = _finish_response(turn, content=str(step["content"]))
    else:
        response = _tool_response(step, turn)

    state.record(
        {
            "event": "response",
            "phase": step["phase"],
            "sandbox_id": sandbox_id,
            "sender_key": sender_key,
            "turn": turn,
            "response_kind": response_kind,
            "tool_calls_since_finish": tool_calls_since_finish,
            "response": response,
        }
    )
    return response


class SimulatedLLMHandler(BaseHTTPRequestHandler):
    state = SimulatedLLMState()

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


def serve(
    *,
    host: str,
    port: int,
    response_delay_ms: int = DEFAULT_RESPONSE_DELAY_MS,
    max_tool_calls_before_finish: int = DEFAULT_MAX_TOOL_CALLS_BEFORE_FINISH,
) -> ThreadingHTTPServer:
    SimulatedLLMHandler.state = SimulatedLLMState(
        response_delay_ms=response_delay_ms,
        max_tool_calls_before_finish=max_tool_calls_before_finish,
    )
    server = ThreadingHTTPServer((host, port), SimulatedLLMHandler)
    server.simulated_state = SimulatedLLMHandler.state  # type: ignore[attr-defined]
    return server


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--response-delay-ms", type=int, default=DEFAULT_RESPONSE_DELAY_MS)
    parser.add_argument("--max-tool-calls-before-finish", type=int, default=DEFAULT_MAX_TOOL_CALLS_BEFORE_FINISH)
    args = parser.parse_args()

    server = serve(
        host=args.host,
        port=args.port,
        response_delay_ms=args.response_delay_ms,
        max_tool_calls_before_finish=args.max_tool_calls_before_finish,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

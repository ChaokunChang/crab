from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


WORK_DIR = Path(os.environ.get("AGENT_WORK_DIR", "/work"))
PROXY_URL = os.environ["PROXY_URL"].rstrip("/")
STATUS_PORT = int(os.environ.get("STATUS_PORT", "19180"))
POLL_INTERVAL_S = float(os.environ.get("POLL_INTERVAL_S", "0.2"))

STATE_PATH = WORK_DIR / "agent_state.json"
SIDE_EFFECT_PATH = WORK_DIR / "side_effects.log"
ACTION_LOG_PATH = WORK_DIR / "actions.log"

WORK_DIR.mkdir(parents=True, exist_ok=True)

lock = threading.Lock()
runtime_id = str(uuid.uuid4())
started_at = time.time()
state = {
    "runtime_id": runtime_id,
    "started_at": started_at,
    "total_actions": 0,
    "stateless_actions": 0,
    "stateful_actions": 0,
    "side_effectful_actions": 0,
    "memory_notes": [],
    "last_action_id": None,
}


def persist_state() -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, sort_keys=True))
    tmp.replace(STATE_PATH)


def append_log(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def fetch_action() -> dict[str, object]:
    payload = json.dumps(
        {
            "runtime_id": state["runtime_id"],
            "total_actions": state["total_actions"],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        PROXY_URL + "/next_action",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode("utf-8"))


def execute_action(action: dict[str, object]) -> None:
    category = str(action["category"])
    action_id = str(action["action_id"])
    payload = dict(action.get("payload", {}))
    state["total_actions"] += 1
    state["last_action_id"] = action_id
    append_log(ACTION_LOG_PATH, json.dumps({"ts": time.time(), "action": action}, sort_keys=True))

    if category == "stateless":
        state["stateless_actions"] += 1
        return
    if category == "stateful":
        state["stateful_actions"] += 1
        state["memory_notes"].append(str(payload.get("note", "")))
        persist_state()
        return
    if category == "side_effectful":
        state["side_effectful_actions"] += 1
        filename = str(payload.get("filename", "tool_artifact.txt"))
        content = str(payload.get("content", "artifact updated"))
        (WORK_DIR / filename).write_text(content + "\n", encoding="utf-8")
        append_log(SIDE_EFFECT_PATH, f"{time.time():.6f} wrote {filename}")
        persist_state()
        return
    raise ValueError(f"unknown action category: {category}")


def agent_loop() -> None:
    while True:
        with lock:
            action = fetch_action()
            execute_action(action)
        time.sleep(POLL_INTERVAL_S)


class StatusHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/status":
            self.send_error(404)
            return
        with lock:
            payload = dict(state)
        payload["side_effect_log_exists"] = SIDE_EFFECT_PATH.exists()
        payload["state_file_exists"] = STATE_PATH.exists()
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        _ = (format, args)
        return


def main() -> None:
    persist_state()
    threading.Thread(target=agent_loop, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", STATUS_PORT), StatusHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()

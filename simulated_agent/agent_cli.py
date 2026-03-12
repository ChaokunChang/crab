from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from simulated_agent.tool_catalog import TOOL_DEFINITIONS, get_tool, provider_tools


def parse_openai_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for choice in payload.get("choices", []):
        message = dict(choice.get("message", {}))
        for tool_call in message.get("tool_calls", []):
            function = dict(tool_call.get("function", {}))
            raw_arguments = function.get("arguments", "{}")
            arguments = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
            calls.append(
                {
                    "id": str(tool_call.get("id", "")),
                    "name": str(function["name"]),
                    "input": arguments,
                }
            )
    return calls


def parse_anthropic_tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in payload.get("content", []):
        if block.get("type") != "tool_use":
            continue
        calls.append(
            {
                "id": str(block.get("id", "")),
                "name": str(block["name"]),
                "input": dict(block.get("input", {})),
            }
        )
    return calls


class AgentRuntime:
    def __init__(
        self,
        *,
        provider: str,
        interceptor_url: str,
        sandbox_id: str,
        work_dir: Path,
        poll_interval_s: float,
        status_port: int,
    ) -> None:
        self.provider = provider
        self.interceptor_url = interceptor_url.rstrip("/")
        self.sandbox_id = sandbox_id
        self.work_dir = work_dir
        self.poll_interval_s = poll_interval_s
        self.status_port = status_port
        self.state_path = self.work_dir / "agent_state.json"
        self.action_log_path = self.work_dir / "actions.log"
        self.tool_activity_path = self.work_dir / "tool_activity.log"
        self.tool_artifact_path = self.work_dir / "tool_artifact.txt"
        self.journal_path = self.work_dir / "journal.log"
        self.lock = threading.Lock()
        self.memory_notes: list[str] = []
        self.state = {
            "runtime_id": str(uuid.uuid4()),
            "started_at": time.time(),
            "provider": self.provider,
            "total_requests": 0,
            "completed_requests": 0,
            "total_actions": 0,
            "stateless_actions": 0,
            "stateful_actions": 0,
            "filesystem_actions": 0,
            "process_actions": 0,
            "network_actions": 0,
            "idempotent_actions": 0,
            "non_idempotent_actions": 0,
            "last_tool_name": None,
            "last_tool_input": None,
            "last_tool_result": None,
        }
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.persist_state()

    def persist_state(self) -> None:
        payload = dict(self.state)
        payload["memory_notes"] = list(self.memory_notes)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def build_request(self) -> tuple[str, dict[str, str], bytes]:
        with self.lock:
            total_actions = self.state["total_actions"]
        metadata = {
            "sandbox_id": self.sandbox_id,
            "total_actions": total_actions,
        }
        if self.provider == "openai":
            path = "/v1/chat/completions"
            body = {
                "model": "simulated-openai",
                "messages": [{"role": "user", "content": "continue"}],
                "tools": provider_tools("openai"),
                "tool_choice": "required",
                "metadata": metadata,
            }
        elif self.provider == "anthropic":
            path = "/v1/messages"
            body = {
                "model": "simulated-anthropic",
                "max_tokens": 64,
                "messages": [{"role": "user", "content": "continue"}],
                "tools": provider_tools("anthropic"),
                "metadata": metadata,
            }
        else:
            raise ValueError(f"unsupported provider: {self.provider}")
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Sandbox-Id": self.sandbox_id,
            "X-Request-Id": str(uuid.uuid4()),
        }
        return path, headers, json.dumps(body, sort_keys=True).encode("utf-8")

    def fetch_tool_calls(self) -> list[dict[str, Any]]:
        path, headers, body = self.build_request()
        req = urllib.request.Request(
            self.interceptor_url + path,
            data=body,
            headers=headers,
            method="POST",
        )
        with self.lock:
            self.state["total_requests"] += 1
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        with self.lock:
            self.state["completed_requests"] += 1
        if self.provider == "openai":
            return parse_openai_tool_calls(payload)
        return parse_anthropic_tool_calls(payload)

    def run_tool(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        tool = get_tool(name)
        result: dict[str, Any]
        if name == "read_workdir":
            result = {"entries": sorted(path.name for path in self.work_dir.iterdir())}
        elif name == "show_pwd":
            result = {"cwd": str(self.work_dir)}
        elif name == "remember_note":
            note = str(tool_input["note"])
            self.memory_notes.append(note)
            result = {"remembered": note}
        elif name == "overwrite_artifact":
            self.tool_artifact_path.write_text(str(tool_input["content"]) + "\n", encoding="utf-8")
            result = {"path": str(self.tool_artifact_path.name)}
        elif name == "append_journal":
            with self.journal_path.open("a", encoding="utf-8") as fh:
                fh.write(str(tool_input["line"]) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            result = {"path": str(self.journal_path.name)}
        elif name == "mkdir_cache":
            target = self.work_dir / str(tool_input["dirname"])
            target.mkdir(parents=True, exist_ok=True)
            result = {"path": str(target.relative_to(self.work_dir))}
        elif name == "spawn_probe":
            proc = subprocess.run(
                ["/bin/sh", "-lc", f"printf '%s' {json.dumps(str(tool_input['message']))} >/dev/null"],
                check=False,
                cwd=str(self.work_dir),
            )
            result = {"returncode": proc.returncode}
        elif name == "fetch_proxy_health":
            with urllib.request.urlopen(self.interceptor_url + str(tool_input["path"]), timeout=10.0) as resp:
                result = {"status": int(resp.status), "body": resp.read().decode("utf-8")}
        else:
            raise ValueError(f"unsupported tool: {name}")

        self.state["total_actions"] += 1
        if tool.stateless:
            self.state["stateless_actions"] += 1
        else:
            self.state["stateful_actions"] += 1
        if tool.changes_filesystem:
            self.state["filesystem_actions"] += 1
        if tool.changes_process_state:
            self.state["process_actions"] += 1
        if tool.uses_network:
            self.state["network_actions"] += 1
        if tool.idempotent:
            self.state["idempotent_actions"] += 1
        else:
            self.state["non_idempotent_actions"] += 1
        self.state["last_tool_name"] = name
        self.state["last_tool_input"] = dict(tool_input)
        self.state["last_tool_result"] = dict(result)
        event = {
            "ts": time.time(),
            "tool_name": name,
            "input": dict(tool_input),
            "result": dict(result),
            "effects": tool.effect_metadata(),
        }
        self.append_jsonl(self.action_log_path, event)
        self.append_jsonl(self.tool_activity_path, event)
        self.persist_state()
        return result

    def loop_forever(self) -> None:
        while True:
            calls = self.fetch_tool_calls()
            for call in calls:
                with self.lock:
                    self.run_tool(str(call["name"]), dict(call["input"]))
            time.sleep(self.poll_interval_s)

    def status_payload(self) -> dict[str, Any]:
        payload = dict(self.state)
        payload["memory_notes"] = list(self.memory_notes)
        payload["state_file_exists"] = self.state_path.exists()
        payload["tool_artifact_exists"] = self.tool_artifact_path.exists()
        payload["journal_exists"] = self.journal_path.exists()
        return payload


class StatusHandler(BaseHTTPRequestHandler):
    runtime: AgentRuntime

    def do_GET(self) -> None:
        if self.path != "/status":
            self.send_error(404)
            return
        with self.runtime.lock:
            body = json.dumps(self.runtime.status_payload(), sort_keys=True).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        _ = (format, args)
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--provider", choices=["openai", "anthropic"], default=os.environ.get("AGENT_PROVIDER", "openai"))
    args = parser.parse_args()

    runtime = AgentRuntime(
        provider=args.provider,
        interceptor_url=os.environ["INTERCEPTOR_URL"],
        sandbox_id=os.environ.get("AGENT_SANDBOX_ID", "sandbox-unknown"),
        work_dir=Path(os.environ.get("AGENT_WORK_DIR", "/work")),
        poll_interval_s=float(os.environ.get("POLL_INTERVAL_S", "0.2")),
        status_port=int(os.environ.get("STATUS_PORT", "19180")),
    )
    StatusHandler.runtime = runtime
    threading.Thread(target=runtime.loop_forever, daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", runtime.status_port), StatusHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()

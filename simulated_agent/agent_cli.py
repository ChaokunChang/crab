from __future__ import annotations

import argparse
import json
import logging
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
        self.log_path = self.work_dir / "agent_cli.log"
        self.lock = threading.Lock()
        self.memory_notes: list[str] = []
        self.state = {
            "runtime_id": str(uuid.uuid4()),
            "started_at": time.time(),
            "provider": self.provider,
            "total_requests": 0,
            "completed_requests": 0,
            "request_errors": 0,
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
            "last_error": None,
        }
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._build_logger()
        self._load_persisted_state()
        self.persist_state()
        self.logger.info(
            "agent runtime started sandbox_id=%s provider=%s work_dir=%s poll_interval_s=%.3f status_port=%d",
            self.sandbox_id,
            self.provider,
            self.work_dir,
            self.poll_interval_s,
            self.status_port,
        )

    def _build_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"simulated_agent.agent_cli.{self.sandbox_id}.{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        for existing in list(logger.handlers):
            logger.removeHandler(existing)
            existing.close()
        handler = logging.FileHandler(self.log_path, encoding="utf-8")
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        return logger

    def _load_persisted_state(self) -> None:
        if not self.state_path.exists():
            self.logger.debug("no persisted state found at %s", self.state_path)
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self.logger.warning("failed to load persisted state from %s", self.state_path, exc_info=True)
            return
        if not isinstance(payload, dict):
            self.logger.warning("ignored non-dict persisted state from %s", self.state_path)
            return
        memory_notes = payload.get("memory_notes", [])
        if isinstance(memory_notes, list):
            self.memory_notes = [str(note) for note in memory_notes]
        for key in self.state:
            if key in {"runtime_id", "started_at", "provider"}:
                continue
            if key in payload:
                self.state[key] = payload[key]
        self.logger.debug(
            "reloaded persisted state total_actions=%s completed_requests=%s memory_notes=%d",
            self.state["total_actions"],
            self.state["completed_requests"],
            len(self.memory_notes),
        )

    def persist_state(self) -> None:
        payload = dict(self.state)
        payload["memory_notes"] = list(self.memory_notes)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)
        self.logger.debug(
            "persisted state total_requests=%s completed_requests=%s total_actions=%s",
            self.state["total_requests"],
            self.state["completed_requests"],
            self.state["total_actions"],
        )

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
        self.logger.debug(
            "fetching tool calls path=%s request_id=%s total_actions=%s",
            path,
            headers["X-Request-Id"],
            self.state["total_actions"],
        )
        req = urllib.request.Request(
            self.interceptor_url + path,
            data=body,
            headers=headers,
            method="POST",
        )
        with self.lock:
            self.state["total_requests"] += 1
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        with self.lock:
            self.state["completed_requests"] += 1
        if self.provider == "openai":
            calls = parse_openai_tool_calls(payload)
        else:
            calls = parse_anthropic_tool_calls(payload)
        self.logger.info(
            "received tool calls count=%d provider=%s sandbox_id=%s",
            len(calls),
            self.provider,
            self.sandbox_id,
        )
        return calls

    def record_error(self, stage: str, exc: Exception) -> None:
        self.state["request_errors"] += 1
        self.state["last_error"] = {
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
            "ts": time.time(),
        }
        self.logger.warning(
            "runtime error stage=%s type=%s message=%s",
            stage,
            type(exc).__name__,
            exc,
        )
        self.persist_state()

    def run_tool(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        tool = get_tool(name)
        self.logger.info("running tool name=%s", name)
        self.logger.debug("tool input name=%s payload=%s", name, json.dumps(tool_input, sort_keys=True))
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
        self.logger.debug("tool result name=%s payload=%s", name, json.dumps(result, sort_keys=True))
        return result

    def run_cycle(self) -> None:
        self.logger.debug("starting agent cycle sandbox_id=%s", self.sandbox_id)
        try:
            calls = self.fetch_tool_calls()
        except Exception as exc:
            with self.lock:
                self.record_error("fetch_tool_calls", exc)
            return
        if not calls:
            self.logger.debug("no tool calls returned for sandbox_id=%s", self.sandbox_id)
        for call in calls:
            try:
                with self.lock:
                    self.run_tool(str(call["name"]), dict(call["input"]))
            except Exception as exc:
                with self.lock:
                    self.record_error("run_tool", exc)
                return

    def loop_forever(self) -> None:
        self.logger.info("entering agent loop sandbox_id=%s", self.sandbox_id)
        while True:
            self.run_cycle()
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

"""Run a real runc-backed iFlow sandbox through the SDK.

This example intentionally uses a tiny local OpenAI-compatible server so it
does not need external credentials. The request still travels through the SDK
interceptor and forwarder before reaching the local upstream.

    PYTHONPATH=. python3 examples/sdk/05_iflow_runc.py
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent_cr import Engine, EngineConfig, Sandbox
from agent_cr.agents_builtin.iflow import IFlowAgent


class _OneShotOpenAI(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _json(self, payload: dict[str, object], *, code: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._json({"ok": True})
            return
        if self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": "agent-cr-iflow-sdk"}]})
            return
        self._json({"error": self.path}, code=404)

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        if length:
            self.rfile.read(length)
        if self.path != "/v1/chat/completions":
            self._json({"error": self.path}, code=404)
            return
        self._json(
            {
                "id": "chatcmpl-sdk-example",
                "object": "chat.completion",
                "created": 0,
                "model": "agent-cr-iflow-sdk",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done.", "tool_calls": []},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        )

    def log_message(self, fmt: str, *args: object) -> None:
        return


def _serve_fake_llm() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _OneShotOpenAI)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, f"http://127.0.0.1:{server.server_address[1]}"


def main() -> None:
    server, thread, llm_url = _serve_fake_llm()
    try:
        with Engine.start(EngineConfig(runtime="runc")) as engine:
            sbx = Sandbox(engine=engine, name="sdk-iflow-demo")
            agent = IFlowAgent(timeout=180).bind(sbx, llm_url=llm_url)
            try:
                result = agent.run("Say Done and do not call tools.")
                print("task exit:", result.exit_code)
                print("task output:", result.output.strip())

                sbx.commands.run("printf before >/work/sdk-demo.txt", check=True)
                ckpt = sbx.checkpoint(label="after-iflow")
                sbx.commands.run("printf after >/work/sdk-demo.txt", check=True)
                sbx.restore(ckpt)
                restored = sbx.commands.run("cat /work/sdk-demo.txt", check=True)
                print("restored file:", restored.stdout.strip())
            finally:
                sbx.kill()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


if __name__ == "__main__":
    main()

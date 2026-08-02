"""Test the per-sandbox upstream URL routing used by the SDK Engine.

The SDK reuses the harness pattern: a single interceptor sends all traffic
to a single forwarder; the forwarder reads `X-Agent-Sandbox-Id` and
dispatches to per-sandbox real LLM URLs. This test spins up two fake
upstream HTTP servers, registers them under two sandbox ids, and verifies
that each sandbox's request lands on the right upstream.
"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from crab.engine import Engine, EngineConfig, shutdown_default_engine
from crab.ids import SandboxId


class _LoggingUpstream:
    """Tiny HTTP server that records the bodies it receives."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.received: list[dict[str, object]] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b"{}"
                payload = {
                    "path": self.path,
                    "body": body.decode("utf-8", errors="replace"),
                    "sandbox_header": self.headers.get("X-Agent-Sandbox-Id"),
                }
                with outer._lock:
                    outer.received.append(payload)
                response = json.dumps({"upstream": outer.tag, "ok": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)


class TestPerSandboxUpstreamRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.up_a = _LoggingUpstream("upstream-A")
        self.up_b = _LoggingUpstream("upstream-B")
        self.engine = Engine.start(EngineConfig(runtime="docker"))

    def tearDown(self) -> None:
        self.engine.stop()
        shutdown_default_engine()
        self.up_a.stop()
        self.up_b.stop()

    def _post_via_interceptor(self, sandbox_id: str, path: str, body: bytes) -> bytes:
        req = urllib.request.Request(
            f"{self.engine.interceptor_base_url}{path}",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Agent-Sandbox-Id": sandbox_id,
            },
        )
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return resp.read()

    def test_two_sandboxes_route_to_distinct_upstreams(self) -> None:
        sid_a = SandboxId("sbx-a")
        sid_b = SandboxId("sbx-b")
        self.engine.register_upstream(sid_a, self.up_a.url)
        self.engine.register_upstream(sid_b, self.up_b.url)

        resp_a = self._post_via_interceptor("sbx-a", "/v1/chat/completions", b'{"model":"x"}')
        resp_b = self._post_via_interceptor("sbx-b", "/v1/chat/completions", b'{"model":"y"}')

        self.assertEqual(json.loads(resp_a)["upstream"], "upstream-A")
        self.assertEqual(json.loads(resp_b)["upstream"], "upstream-B")

        # Each upstream should have seen exactly its own sandbox's traffic.
        self.assertEqual(len(self.up_a.received), 1)
        self.assertEqual(self.up_a.received[0]["sandbox_header"], "sbx-a")
        self.assertEqual(len(self.up_b.received), 1)
        self.assertEqual(self.up_b.received[0]["sandbox_header"], "sbx-b")

    def test_unregister_yields_502(self) -> None:
        sid_a = SandboxId("sbx-a")
        self.engine.register_upstream(sid_a, self.up_a.url)
        self.engine.unregister_upstream(sid_a)
        # No upstream registered → forwarder returns 502 explaining the
        # sandbox is unknown. The interceptor faithfully relays that.
        import urllib.error

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self._post_via_interceptor("sbx-a", "/v1/chat/completions", b'{"model":"x"}')
        self.assertEqual(ctx.exception.code, 502)


if __name__ == "__main__":
    unittest.main()

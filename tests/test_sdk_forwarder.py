"""Direct tests for the SDK LLM forwarder.

These exercise `SdkLLMForwarder` and `serve_sdk_llm_forwarder` without the
interceptor in the loop, so we can pin down the dispatch behavior on its
own. The forwarder follows the same per-sandbox-dispatch pattern as the
benchmark router (`integrations.llm_services.router.BenchmarkLLMRouter`).
"""
from __future__ import annotations

import json
import threading
import unittest
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agent_cr.sdk_llm_forwarder import SdkLLMForwarder, serve_sdk_llm_forwarder


class _LoggingUpstream:
    def __init__(self, tag: str, *, status: int = 200) -> None:
        self.tag = tag
        self.received: list[dict[str, object]] = []
        self._status = status
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length) if length else b""
                payload = {
                    "path": self.path,
                    "body": body.decode("utf-8", errors="replace"),
                    "sandbox_header": self.headers.get("X-Agent-Sandbox-Id"),
                    "extra_header": self.headers.get("X-Test-Header"),
                }
                with outer._lock:
                    outer.received.append(payload)
                response = json.dumps({"upstream": outer.tag}).encode("utf-8")
                self.send_response(outer._status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                _ = (format, args)

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


class TestSdkLLMForwarderUnit(unittest.TestCase):
    def test_register_unregister_resolve(self) -> None:
        fwd = SdkLLMForwarder()
        fwd.register("sbx-1", "https://api.example/v1")
        self.assertEqual(fwd.resolve("sbx-1"), "https://api.example/v1")
        self.assertEqual(fwd.registered_count(), 1)
        fwd.unregister("sbx-1")
        self.assertIsNone(fwd.resolve("sbx-1"))

    def test_register_rejects_bad_scheme(self) -> None:
        fwd = SdkLLMForwarder()
        with self.assertRaises(ValueError):
            fwd.register("sbx", "ftp://nope")

    def test_register_rejects_empty(self) -> None:
        fwd = SdkLLMForwarder()
        with self.assertRaises(ValueError):
            fwd.register("", "https://x")
        with self.assertRaises(ValueError):
            fwd.register("sbx", "")


class TestSdkLLMForwarderHTTP(unittest.TestCase):
    def setUp(self) -> None:
        self.up_a = _LoggingUpstream("A")
        self.up_b = _LoggingUpstream("B")
        self.server, self.forwarder = serve_sdk_llm_forwarder()
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)
        self.up_a.stop()
        self.up_b.stop()

    def _post(self, sandbox_id: str, path: str, body: bytes, *, extra_headers: dict[str, str] | None = None) -> tuple[int, bytes]:
        headers = {"Content-Type": "application/json", "X-Agent-Sandbox-Id": sandbox_id}
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method="POST",
            headers=headers,
        )
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read() if exc.fp else b""

    def test_routes_per_sandbox(self) -> None:
        self.forwarder.register("sbx-a", self.up_a.url)
        self.forwarder.register("sbx-b", self.up_b.url)
        code_a, body_a = self._post("sbx-a", "/v1/chat/completions", b'{"q":1}')
        code_b, body_b = self._post("sbx-b", "/v1/messages", b'{"q":2}')
        self.assertEqual(code_a, 200)
        self.assertEqual(code_b, 200)
        self.assertEqual(json.loads(body_a)["upstream"], "A")
        self.assertEqual(json.loads(body_b)["upstream"], "B")
        self.assertEqual(len(self.up_a.received), 1)
        self.assertEqual(self.up_a.received[0]["sandbox_header"], "sbx-a")
        self.assertEqual(len(self.up_b.received), 1)
        self.assertEqual(self.up_b.received[0]["path"], "/v1/messages")

    def test_openai_base_url_with_v1_does_not_duplicate_prefix(self) -> None:
        self.forwarder.register("sbx-a", f"{self.up_a.url}/v1")
        code, _ = self._post("sbx-a", "/v1/chat/completions", b'{"q":1}')
        self.assertEqual(code, 200)
        self.assertEqual(len(self.up_a.received), 1)
        self.assertEqual(self.up_a.received[0]["path"], "/v1/chat/completions")

    def test_unknown_sandbox_returns_502(self) -> None:
        code, body = self._post("sbx-unknown", "/v1/messages", b'{"q":1}')
        self.assertEqual(code, 502)
        self.assertIn(b"no upstream registered", body)

    def test_missing_sandbox_header_returns_400(self) -> None:
        req = urllib.request.Request(
            f"{self.base_url}/v1/messages",
            data=b'{"q":1}',
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5.0)
        self.assertEqual(ctx.exception.code, 400)

    def test_unknown_path_returns_404(self) -> None:
        self.forwarder.register("sbx-a", self.up_a.url)
        req = urllib.request.Request(
            f"{self.base_url}/v1/something-else",
            data=b'{}',
            method="POST",
            headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-a"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5.0)
        self.assertEqual(ctx.exception.code, 404)

    def test_propagates_request_headers(self) -> None:
        self.forwarder.register("sbx-a", self.up_a.url)
        self._post("sbx-a", "/v1/messages", b'{}', extra_headers={"X-Test-Header": "alpha"})
        self.assertEqual(self.up_a.received[0]["extra_header"], "alpha")

    def test_healthz(self) -> None:
        self.forwarder.register("sbx-a", self.up_a.url)
        self.forwarder.register("sbx-b", self.up_b.url)
        req = urllib.request.Request(f"{self.base_url}/healthz")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read())
        self.assertEqual(payload["ok"], True)
        self.assertEqual(payload["registered_sandboxes"], 2)


if __name__ == "__main__":
    unittest.main()

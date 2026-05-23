"""Per-sandbox LLM forwarder for the SDK Engine.

This is the SDK analogue of `integrations/llm_services/router.py`'s
`BenchmarkLLMRouter` / `serve_benchmark_llm_router`. It plays the same
architectural role:

    sandbox → AgentCRRequestInterceptorServer → [this forwarder] → real LLM

The interceptor is unchanged. Like the harness, it points at a single
upstream URL — but instead of pointing at the benchmark router (which
dispatches to simulated/replay services), it points at this forwarder
(which dispatches to per-sandbox real LLM endpoints).

Identification of which sandbox owns a given request is done via the
existing `sandbox_id_resolver` callback on the interceptor (header-first,
IP-fallback via the network manager) — exactly the same mechanism the
harness uses. The interceptor stamps `X-Agent-Sandbox-Id` on every forwarded
request, and this forwarder reads it.

We use `urllib.request` instead of `ThreadLocalHttpClient` because real LLM
endpoints are HTTPS and `ThreadLocalHttpClient` is http-only. This keeps the
existing client class untouched.
"""
from __future__ import annotations

import logging
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

from .http_utils import PooledHTTPServer

logger = logging.getLogger(__name__)


# Paths the forwarder accepts — identical to the interceptor's allowlist.
_ACCEPTED_PATHS = frozenset(
    {"/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"}
)


# Headers we never propagate to the real upstream. `Host` is rewritten by
# urllib based on the target URL. `Content-Length` is recomputed. Hop-by-hop
# headers per RFC 7230 §6.1 are stripped to avoid confusing the upstream.
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "transfer-encoding",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "upgrade",
    }
)


def _join_upstream_url(upstream: str, path: str) -> str:
    """Join provider base URLs with interceptor paths.

    OpenAI-compatible APIs are commonly configured as `https://host/v1`, while
    the interceptor forwards paths like `/v1/chat/completions`. Accept that
    conventional base URL without producing `/v1/v1/...`.
    """
    if upstream.endswith("/v1") and path.startswith("/v1/"):
        return upstream + path[len("/v1"):]
    return upstream + path


class SdkLLMForwarder:
    """Per-sandbox upstream URL registry + raw-bytes passthrough."""

    def __init__(self, *, request_timeout_seconds: float = 600.0) -> None:
        self._lock = threading.Lock()
        self._upstreams: dict[str, str] = {}
        self._request_timeout_seconds = float(request_timeout_seconds)

    def register(self, sandbox_id: str, upstream_url: str) -> None:
        if not sandbox_id:
            raise ValueError("sandbox_id must be non-empty")
        if not upstream_url:
            raise ValueError("upstream_url must be non-empty")
        parsed = urlsplit(upstream_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(
                f"upstream_url must be http(s); got scheme={parsed.scheme!r} for {upstream_url!r}"
            )
        normalized = upstream_url.rstrip("/")
        with self._lock:
            self._upstreams[sandbox_id] = normalized

    def unregister(self, sandbox_id: str) -> None:
        with self._lock:
            self._upstreams.pop(sandbox_id, None)

    def resolve(self, sandbox_id: str) -> str | None:
        with self._lock:
            return self._upstreams.get(sandbox_id)

    def registered_count(self) -> int:
        with self._lock:
            return len(self._upstreams)

    def forward(
        self,
        *,
        sandbox_id: str,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        upstream = self.resolve(sandbox_id)
        if upstream is None:
            payload = (
                f'{{"error":"no upstream registered for sandbox_id={sandbox_id!r}"}}'
            ).encode("utf-8")
            return 502, [("Content-Type", "application/json")], payload
        target_url = _join_upstream_url(upstream, path)
        forwarded_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in _HOP_BY_HOP_HEADERS
        }
        request = urllib.request.Request(
            target_url,
            data=body,
            headers=forwarded_headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self._request_timeout_seconds
            ) as response:
                response_body = response.read()
                response_headers = [
                    (str(k), str(v)) for k, v in response.getheaders()
                ]
                return int(response.status), response_headers, response_body
        except urllib.error.HTTPError as exc:
            error_body = exc.read() if exc.fp is not None else b""
            response_headers = [(str(k), str(v)) for k, v in exc.headers.items()]
            return int(exc.code), response_headers, error_body
        except urllib.error.URLError as exc:
            logger.warning(
                "Forwarder upstream connection failed: sandbox=%s upstream=%s error=%s",
                sandbox_id,
                upstream,
                exc,
            )
            payload = f'{{"error":"upstream connection failed: {exc.reason}"}}'.encode(
                "utf-8"
            )
            return 502, [("Content-Type", "application/json")], payload


def serve_sdk_llm_forwarder(
    *,
    host: str = "127.0.0.1",
    port: int = 0,
    forwarder: SdkLLMForwarder | None = None,
    max_workers: int | None = None,
) -> tuple[PooledHTTPServer, SdkLLMForwarder]:
    """HTTP server in front of `SdkLLMForwarder`.

    Mirrors the shape of `integrations.llm_services.router.serve_benchmark_llm_router`:
    a `PooledHTTPServer` + per-request handler that delegates to the forwarder.
    Returns the server and the forwarder so callers can register/unregister
    upstreams without going through HTTP.
    """
    instance = forwarder or SdkLLMForwarder()

    class ForwarderHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        sdk_forwarder: SdkLLMForwarder = instance  # type: ignore[assignment]

        def end_headers(self) -> None:
            self.send_header("Connection", "close")
            self.close_connection = True
            super().end_headers()

        def _send_payload(
            self,
            status_code: int,
            headers: list[tuple[str, str]],
            body: bytes,
        ) -> None:
            self.send_response(status_code)
            for key, value in headers:
                if key.lower() in _HOP_BY_HOP_HEADERS:
                    continue
                self.send_header(key, value)
            if not any(k.lower() == "content-length" for k, _ in headers):
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                logger.debug("Client disconnected while writing forwarder response")

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                payload = (
                    f'{{"ok":true,"registered_sandboxes":{self.sdk_forwarder.registered_count()}}}'
                ).encode("utf-8")
                self._send_payload(200, [("Content-Type", "application/json")], payload)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            request_path = self.path.split("?")[0]
            if request_path not in _ACCEPTED_PATHS:
                self.send_error(404)
                return
            sandbox_id = self.headers.get("X-Agent-Sandbox-Id", "").strip()
            if not sandbox_id:
                self.send_error(400, "missing X-Agent-Sandbox-Id header")
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length) if length else b""
            headers = {str(k): str(v) for k, v in self.headers.items()}
            status_code, response_headers, response_body = self.sdk_forwarder.forward(
                sandbox_id=sandbox_id,
                path=request_path,
                headers=headers,
                body=body,
            )
            self._send_payload(status_code, response_headers, response_body)

        def log_message(self, format: str, *args) -> None:  # noqa: A002
            _ = (format, args)

    server = PooledHTTPServer((host, port), ForwarderHandler, max_workers=max_workers)
    server.sdk_forwarder = instance  # type: ignore[attr-defined]
    return server, instance


__all__ = ["SdkLLMForwarder", "serve_sdk_llm_forwarder"]

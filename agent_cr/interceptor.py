from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .contracts import RequestInterceptorHook, SandboxInspector, TelemetrySink
from .ids import SandboxId
from .models import RequestContext, RequestState, SandboxSnapshot, utc_now


class CompositeRequestInterceptorHook(RequestInterceptorHook):
    def __init__(self, hooks: list[RequestInterceptorHook] | None = None):
        self._hooks = list(hooks or [])

    def add_hook(self, hook: RequestInterceptorHook) -> None:
        self._hooks.append(hook)

    def on_request_start(self, context: RequestContext) -> None:
        for hook in self._hooks:
            hook.on_request_start(context)

    def on_request_end(self, context: RequestContext) -> None:
        for hook in self._hooks:
            hook.on_request_end(context)


class TelemetryRequestInterceptorHook(RequestInterceptorHook):
    def __init__(self, telemetry: TelemetrySink):
        self._telemetry = telemetry

    def on_request_start(self, context: RequestContext) -> None:
        self._telemetry.emit_event(
            "request.start",
            {
                "request_id": context.request_id,
                "sandbox_id": str(context.sandbox_id),
                "provider": context.metadata.get("provider", ""),
                "path": context.metadata.get("path", ""),
            },
        )

    def on_request_end(self, context: RequestContext) -> None:
        self._telemetry.emit_event(
            "request.end",
            {
                "request_id": context.request_id,
                "sandbox_id": str(context.sandbox_id),
                "provider": context.metadata.get("provider", ""),
                "path": context.metadata.get("path", ""),
            },
        )


class InMemoryRequestStateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._states: dict[SandboxId, RequestState] = {}

    def mark_request_start(self, context: RequestContext) -> RequestState:
        with self._lock:
            current = self._states.get(context.sandbox_id, RequestState(sandbox_id=context.sandbox_id))
            provider = None if context.metadata.get("provider") is None else str(context.metadata["provider"])
            updated = replace(
                current,
                active_llm_requests=current.active_llm_requests + 1,
                total_llm_requests=current.total_llm_requests + 1,
                last_request_id=context.request_id,
                last_llm_provider=provider,
                last_llm_request_started_at=context.started_at,
            )
            self._states[context.sandbox_id] = updated
            return updated

    def mark_request_end(self, context: RequestContext) -> RequestState:
        with self._lock:
            current = self._states.get(context.sandbox_id, RequestState(sandbox_id=context.sandbox_id))
            provider = None if context.metadata.get("provider") is None else str(context.metadata["provider"])
            updated = replace(
                current,
                active_llm_requests=max(0, current.active_llm_requests - 1),
                completed_llm_requests=current.completed_llm_requests + 1,
                last_request_id=context.request_id,
                last_llm_provider=provider,
                last_llm_request_ended_at=utc_now(),
            )
            self._states[context.sandbox_id] = updated
            return updated

    def get(self, sandbox_id: SandboxId) -> RequestState:
        with self._lock:
            return self._states.get(sandbox_id, RequestState(sandbox_id=sandbox_id))


class RequestAwareSandboxInspector(SandboxInspector):
    def __init__(self, base: SandboxInspector, request_state_store: InMemoryRequestStateStore) -> None:
        self._base = base
        self._request_state_store = request_state_store

    def __getattr__(self, name: str):
        return getattr(self._base, name)

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        base_snapshot = self._base.inspect(sandbox_id)
        request_state = self._request_state_store.get(sandbox_id)
        observed_at = base_snapshot.observed_at
        if request_state.last_llm_request_started_at is not None:
            observed_at = max(observed_at, request_state.last_llm_request_started_at)
        if request_state.last_llm_request_ended_at is not None:
            observed_at = max(observed_at, request_state.last_llm_request_ended_at)
        return replace(
            base_snapshot,
            observed_at=observed_at,
            metadata={**base_snapshot.metadata, **request_state.to_metadata()},
        )


class AgentCRRequestInterceptor:
    def __init__(
        self,
        *,
        upstream_transport: Callable[[str, dict[str, str], bytes], tuple[int, list[tuple[str, str]], bytes]],
        request_state_store: InMemoryRequestStateStore,
        hook: RequestInterceptorHook | None = None,
        on_state_change: Callable[[SandboxId], None] | None = None,
    ) -> None:
        self._upstream_transport = upstream_transport
        self._request_state_store = request_state_store
        self._hook = hook or CompositeRequestInterceptorHook()
        self._on_state_change = on_state_change

    def intercept(
        self,
        *,
        path: str,
        headers: dict[str, str],
        body: bytes,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        sandbox_id_raw = headers.get("X-Agent-Sandbox-Id", "").strip()
        if not sandbox_id_raw:
            raise ValueError("missing X-Agent-Sandbox-Id")
        provider = "openai" if path == "/v1/chat/completions" else "anthropic"
        context = RequestContext(
            request_id=headers.get("X-Request-Id", "").strip() or str(uuid.uuid4()),
            sandbox_id=SandboxId(sandbox_id_raw),
            started_at=utc_now(),
            metadata={"provider": provider, "path": path},
        )
        self._hook.on_request_start(context)
        self._request_state_store.mark_request_start(context)
        self._notify(context.sandbox_id)
        try:
            return self._upstream_transport(path, headers, body)
        finally:
            self._request_state_store.mark_request_end(context)
            self._hook.on_request_end(context)
            self._notify(context.sandbox_id)

    def _notify(self, sandbox_id: SandboxId) -> None:
        if self._on_state_change is not None:
            try:
                self._on_state_change(sandbox_id)
            except Exception:
                return


class AgentCRRequestInterceptorServer:
    def __init__(
        self,
        *,
        upstream_url: str,
        request_state_store: InMemoryRequestStateStore,
        hook: RequestInterceptorHook | None = None,
        on_state_change: Callable[[SandboxId], None] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._upstream_url = upstream_url.rstrip("/")
        self._interceptor = AgentCRRequestInterceptor(
            upstream_transport=self._forward,
            request_state_store=request_state_store,
            hook=hook,
            on_state_change=on_state_change,
        )
        self._server = ThreadingHTTPServer((host, port), self._build_handler())
        self._thread: threading.Thread | None = None

    @property
    def host(self) -> str:
        return str(self._server.server_address[0])

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _build_handler(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/healthz":
                    self.send_error(404)
                    return
                body = json.dumps(
                    {
                        "ok": True,
                        "upstream_url": outer._upstream_url,
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                if self.path not in {"/v1/chat/completions", "/v1/messages"}:
                    self.send_error(404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    body = self.rfile.read(length) if length else b"{}"
                    status_code, headers, body = outer._interceptor.intercept(
                        path=self.path,
                        headers=dict(self.headers.items()),
                        body=body,
                    )
                    self.send_response(status_code)
                    for key, value in headers:
                        if key.lower() == "transfer-encoding":
                            continue
                        self.send_header(key, value)
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except BrokenPipeError:
                        return
                except ValueError as exc:
                    self.send_error(400, str(exc))
                except urllib.error.HTTPError as exc:
                    body = exc.read()
                    self.send_response(exc.code)
                    for key, value in exc.headers.items():
                        if key.lower() == "transfer-encoding":
                            continue
                        self.send_header(key, value)
                    self.end_headers()
                    try:
                        self.wfile.write(body)
                    except BrokenPipeError:
                        return

            def log_message(self, format: str, *args) -> None:
                _ = (format, args)
                return

        return Handler

    def _forward(
        self,
        path: str,
        headers: dict[str, str],
        payload: bytes,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"host", "content-length"}
        }
        req = urllib.request.Request(
            self._upstream_url + path,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            return int(resp.status), list(resp.headers.items()), resp.read()

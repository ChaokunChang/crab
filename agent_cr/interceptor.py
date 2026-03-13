from __future__ import annotations

from collections import deque
import json
import threading
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from .contracts import RequestInterceptorHook, SandboxInspector, TelemetrySink
from .ids import SandboxId
from .models import RequestContext, RequestState, RequestStateChange, SandboxSnapshot, utc_now


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
        self._condition = threading.Condition(self._lock)
        self._states: dict[SandboxId, RequestState] = {}
        self._changes: deque[RequestStateChange] = deque()

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
            self._changes.append(
                RequestStateChange(
                    sandbox_id=context.sandbox_id,
                    event_type="request_start",
                    request_id=context.request_id,
                    observed_at=context.started_at,
                )
            )
            self._condition.notify_all()
            return updated

    def mark_request_end(self, context: RequestContext) -> RequestState:
        with self._lock:
            current = self._states.get(context.sandbox_id, RequestState(sandbox_id=context.sandbox_id))
            ended_at = utc_now()
            provider = None if context.metadata.get("provider") is None else str(context.metadata["provider"])
            updated = replace(
                current,
                active_llm_requests=max(0, current.active_llm_requests - 1),
                completed_llm_requests=current.completed_llm_requests + 1,
                last_request_id=context.request_id,
                last_llm_provider=provider,
                last_llm_request_ended_at=ended_at,
            )
            self._states[context.sandbox_id] = updated
            self._changes.append(
                RequestStateChange(
                    sandbox_id=context.sandbox_id,
                    event_type="request_end",
                    request_id=context.request_id,
                    observed_at=ended_at,
                )
            )
            self._condition.notify_all()
            return updated

    def get(self, sandbox_id: SandboxId) -> RequestState:
        with self._lock:
            return self._states.get(sandbox_id, RequestState(sandbox_id=sandbox_id))

    def wait_for_change(self, timeout: float | None = None) -> RequestStateChange | None:
        with self._condition:
            if not self._changes:
                self._condition.wait(timeout=timeout)
            if not self._changes:
                return None
            return self._changes.popleft()

    def notify_waiters(self) -> None:
        with self._condition:
            self._condition.notify_all()


@dataclass(frozen=True)
class PendingSandboxResponse:
    sandbox_id: SandboxId
    request_id: str
    generation: int


class SandboxResponseGateRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._enabled = False
        self._states: dict[SandboxId, dict[str, int | bool | str | None | threading.Condition]] = {}

    def enable(self) -> None:
        with self._lock:
            self._enabled = True

    def disable(self) -> None:
        with self._lock:
            self._enabled = False
            conditions = [state["condition"] for state in self._states.values()]
            for state in self._states.values():
                state["active"] = False
                state["released_generation"] = state["generation"]
                state["request_id"] = None
        for condition in conditions:
            assert isinstance(condition, threading.Condition)
            with condition:
                condition.notify_all()

    def arm(self, sandbox_id: SandboxId, request_id: str) -> int | None:
        with self._lock:
            if not self._enabled:
                return None
            state = self._states.get(sandbox_id)
            if state is None:
                condition = threading.Condition()
                state = {
                    "generation": 0,
                    "released_generation": 0,
                    "active": False,
                    "request_id": None,
                    "condition": condition,
                }
                self._states[sandbox_id] = state
            state["generation"] = int(state["generation"]) + 1
            state["active"] = True
            state["request_id"] = request_id
            return int(state["generation"])

    def wait_for_release(self, sandbox_id: SandboxId, generation: int | None, timeout: float | None = None) -> None:
        if generation is None:
            return
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return
            condition = state["condition"]
            assert isinstance(condition, threading.Condition)
        with condition:
            condition.wait_for(
                lambda: self._is_generation_released(sandbox_id, generation),
                timeout=timeout,
            )

    def get_pending(self, sandbox_id: SandboxId) -> PendingSandboxResponse | None:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None or not bool(state["active"]):
                return None
            request_id = state.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                return None
            return PendingSandboxResponse(
                sandbox_id=sandbox_id,
                request_id=request_id,
                generation=int(state["generation"]),
            )

    def release_pending(self, sandbox_id: SandboxId, *, request_id: str, generation: int | None = None) -> bool:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None or not bool(state["active"]):
                return False
            current_request_id = state.get("request_id")
            current_generation = int(state["generation"])
            if current_request_id != request_id:
                return False
            if generation is not None and current_generation != generation:
                return False
            state["active"] = False
            state["released_generation"] = current_generation
            state["request_id"] = None
            condition = state["condition"]
            assert isinstance(condition, threading.Condition)
        with condition:
            condition.notify_all()
        return True

    def release(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return
            state["active"] = False
            state["released_generation"] = int(state["generation"])
            state["request_id"] = None
            condition = state["condition"]
            assert isinstance(condition, threading.Condition)
        with condition:
            condition.notify_all()

    def _is_generation_released(self, sandbox_id: SandboxId, generation: int) -> bool:
        with self._lock:
            state = self._states.get(sandbox_id)
            if state is None:
                return True
            if not self._enabled:
                return True
            return int(state["released_generation"]) >= generation


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

    def mark_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        *,
        process: bool,
        filesystem: bool,
        at,
    ) -> None:
        self._base.mark_checkpoint_complete(
            sandbox_id,
            process=process,
            filesystem=filesystem,
            at=at,
        )


class AgentCRRequestInterceptor:
    def __init__(
        self,
        *,
        upstream_transport: Callable[[str, dict[str, str], bytes], tuple[int, list[tuple[str, str]], bytes]],
        request_state_store: InMemoryRequestStateStore,
        hook: RequestInterceptorHook | None = None,
        on_state_change: Callable[[SandboxId], None] | None = None,
        response_gate_registry: SandboxResponseGateRegistry | None = None,
        sandbox_id_resolver: Callable[[str | None, dict[str, str], bytes], str | None] | None = None,
    ) -> None:
        self._upstream_transport = upstream_transport
        self._request_state_store = request_state_store
        self._hook = hook or CompositeRequestInterceptorHook()
        self._on_state_change = on_state_change
        self._response_gate_registry = response_gate_registry
        self._sandbox_id_resolver = sandbox_id_resolver

    def intercept(
        self,
        *,
        path: str,
        headers: dict[str, str],
        body: bytes,
        client_host: str | None = None,
    ) -> tuple[int, list[tuple[str, str]], bytes]:
        sandbox_id_raw = headers.get("X-Agent-Sandbox-Id", "").strip()
        if self._sandbox_id_resolver is not None:
            resolved = self._sandbox_id_resolver(client_host, headers, body)
            if resolved is not None and resolved.strip():
                sandbox_id_raw = resolved.strip()
        if not sandbox_id_raw:
            raise ValueError("missing X-Agent-Sandbox-Id")
        upstream_headers = dict(headers)
        upstream_headers["X-Agent-Sandbox-Id"] = sandbox_id_raw
        provider = "openai" if path == "/v1/chat/completions" else "anthropic"
        context = RequestContext(
            request_id=headers.get("X-Request-Id", "").strip() or str(uuid.uuid4()),
            sandbox_id=SandboxId(sandbox_id_raw),
            started_at=utc_now(),
            metadata={"provider": provider, "path": path},
        )
        self._hook.on_request_start(context)
        self._request_state_store.mark_request_start(context)
        gate_generation = None
        if self._response_gate_registry is not None:
            gate_generation = self._response_gate_registry.arm(context.sandbox_id, context.request_id)
        self._notify(context.sandbox_id)
        try:
            response = self._upstream_transport(path, upstream_headers, body)
            if self._response_gate_registry is not None:
                self._response_gate_registry.wait_for_release(context.sandbox_id, gate_generation)
            return response
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
        response_gate_registry: SandboxResponseGateRegistry | None = None,
        sandbox_id_resolver: Callable[[str | None, dict[str, str], bytes], str | None] | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self._upstream_url = upstream_url.rstrip("/")
        self._interceptor = AgentCRRequestInterceptor(
            upstream_transport=self._forward,
            request_state_store=request_state_store,
            hook=hook,
            on_state_change=on_state_change,
            response_gate_registry=response_gate_registry,
            sandbox_id_resolver=sandbox_id_resolver,
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
                        client_host=str(self.client_address[0]),
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

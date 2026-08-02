from __future__ import annotations

import argparse
import logging
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from typing import Any, Protocol
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from crab import TelemetryConfig
from crab.http_utils import HttpStatusError, PooledHTTPServer, ThreadLocalHttpClient
from crab.json_codec import get_json_codec
from crab.telemetry import (
    DEFAULT_TELEMETRY_BATCH_MAX_RECORDS,
    DEFAULT_TELEMETRY_FLUSH_INTERVAL_MS,
    DEFAULT_TELEMETRY_OVERFLOW_POLICY,
    DEFAULT_TELEMETRY_QUEUE_CAPACITY,
    DEFAULT_TELEMETRY_SERIALIZER,
    DEFAULT_TELEMETRY_WRITER_MODE,
    build_configured_telemetry_sink,
    start_operation,
)

from integrations.llm_services.iflow_trace_replay.service import TraceReplayLLMState
from integrations.llm_services.claude_code_trace_replay.service import TraceReplayLLMState as ClaudeCodeTraceReplayLLMState
from integrations.llm_services.mini_swe_trace_replay.service import TraceReplayLLMState as MiniSWETraceReplayLLMState
from integrations.llm_services.mini_swe_spec_trace_replay.service import TraceReplayLLMState as MiniSWESpecTraceReplayLLMState
from integrations.llm_services.terminus_trace_replay.service import TraceReplayLLMState as TerminusTraceReplayLLMState
from integrations.llm_services.terminus_spec_trace_replay.service import TraceReplayLLMState as TerminusSpecTraceReplayLLMState
from integrations.llm_services.manual.service import ManualLLMState, handle_control_request
from integrations.llm_services.simulated.service import SimulatedLLMState, handle_request as handle_simulated_request
from integrations.llm_services.simulated_for_iflow.service import (
    SimulatedLLMState as SimulatedIFlowLLMState,
    handle_request as handle_iflow_simulated_request,
)

logger = logging.getLogger(__name__)

_IFLOW_BENCHMARK_MAX_TOOL_CALLS_BEFORE_FINISH = 4096
_JSON_CODEC = get_json_codec("auto")


def _sandbox_id_from_request(headers: dict[str, str], payload: dict[str, Any]) -> str:
    sandbox_id = headers.get("X-Agent-Sandbox-Id", "").strip()
    if sandbox_id:
        return sandbox_id
    metadata = payload.get("metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("sandbox_id"), str):
        return metadata["sandbox_id"]
    return "sandbox-unknown"


def default_llm_service_type_for_agent(agent_type: str) -> str:
    if agent_type == "iflow":
        return "simulated_for_iflow"
    if agent_type == "mini_swe":
        return "mini_swe_trace_replay"
    if agent_type == "claude_code":
        return "claude_code_trace_replay"
    if agent_type == "terminus":
        return "terminus_trace_replay"
    return "simulated"


def validate_llm_service_type(*, provider: str, llm_service_type: str) -> None:
    openai_only = {
        "manual",
        "simulated_for_iflow",
        "iflow_trace_replay",
        "mini_swe_trace_replay",
        "mini_swe_spec_trace_replay",
        "terminus_trace_replay",
        "terminus_spec_trace_replay",
    }
    supported = {
        "simulated",
        "manual",
        "simulated_for_iflow",
        "iflow_trace_replay",
        "mini_swe_trace_replay",
        "mini_swe_spec_trace_replay",
        "claude_code_trace_replay",
        "terminus_trace_replay",
        "terminus_spec_trace_replay",
    }
    if llm_service_type not in supported:
        raise ValueError(f"unsupported llm service type: {llm_service_type}")
    if provider == "anthropic" and llm_service_type in openai_only:
        raise ValueError(f"llm_service_type={llm_service_type} only supports provider=openai")


class LLMServiceState(Protocol):
    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]: ...

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]: ...

    def reset(self) -> None: ...

    def restore(self, *, consumed_response_count: int) -> None: ...


class SimulatedServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        _ = llm_service_config
        self._state = SimulatedLLMState()

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return handle_simulated_request(path=path, headers=headers, payload=payload, state=self._state, response_delay_ms=250)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        _ = include_events
        return {"turns": self._state.snapshot()}

    def reset(self) -> None:
        return

    def restore(self, *, consumed_response_count: int) -> None:
        _ = consumed_response_count
        return


class ManualServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        _ = llm_service_config
        self._state = ManualLLMState()

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._state.next_response(path=path, headers=headers, payload=payload)

    def handle_control(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return handle_control_request(path=path, payload=payload, state=self._state)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        _ = include_events
        return self._state.snapshot()

    def reset(self) -> None:
        return

    def restore(self, *, consumed_response_count: int) -> None:
        _ = consumed_response_count
        return


class SimulatedForIFlowServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        _ = llm_service_config
        self._state = SimulatedIFlowLLMState(
            response_delay_ms=250,
            max_tool_calls_before_finish=_IFLOW_BENCHMARK_MAX_TOOL_CALLS_BEFORE_FINISH,
        )

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return handle_iflow_simulated_request(path=path, headers=headers, payload=payload, state=self._state)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        _ = include_events
        return self._state.snapshot()

    def reset(self) -> None:
        return

    def restore(self, *, consumed_response_count: int) -> None:
        _ = consumed_response_count
        return


class IFlowTraceReplayServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        self._state = TraceReplayLLMState(llm_service_config=llm_service_config)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._state.handle_request(path=path, headers=headers, payload=payload)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        return self._state.snapshot(include_events=include_events)

    def reset(self) -> None:
        self._state.reset()

    def restore(self, *, consumed_response_count: int) -> None:
        self._state.restore(consumed_response_count=consumed_response_count)


class MiniSWETraceReplayServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        self._state = MiniSWETraceReplayLLMState(llm_service_config=llm_service_config)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._state.handle_request(path=path, headers=headers, payload=payload)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        return self._state.snapshot(include_events=include_events)

    def reset(self) -> None:
        self._state.reset()

    def restore(self, *, consumed_response_count: int) -> None:
        self._state.restore(consumed_response_count=consumed_response_count)


class MiniSWESpecTraceReplayServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        self._state = MiniSWESpecTraceReplayLLMState(llm_service_config=llm_service_config)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._state.handle_request(path=path, headers=headers, payload=payload)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        return self._state.snapshot(include_events=include_events)

    def reset(self) -> None:
        self._state.reset()

    def restore(self, *, consumed_response_count: int) -> None:
        self._state.restore(consumed_response_count=consumed_response_count)

    def export_state(self) -> dict[str, Any]:
        return self._state.export_state()

    def import_state(self, state: dict[str, Any]) -> None:
        self._state.import_state(state)


class ClaudeCodeTraceReplayServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        self._state = ClaudeCodeTraceReplayLLMState(llm_service_config=llm_service_config)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._state.handle_request(path=path, headers=headers, payload=payload)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        return self._state.snapshot(include_events=include_events)

    def reset(self) -> None:
        self._state.reset()

    def restore(self, *, consumed_response_count: int) -> None:
        self._state.restore(consumed_response_count=consumed_response_count)


class TerminusTraceReplayServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        self._state = TerminusTraceReplayLLMState(llm_service_config=llm_service_config)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._state.handle_request(path=path, headers=headers, payload=payload)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        return self._state.snapshot(include_events=include_events)

    def reset(self) -> None:
        self._state.reset()

    def restore(self, *, consumed_response_count: int) -> None:
        self._state.restore(consumed_response_count=consumed_response_count)


class TerminusSpecTraceReplayServiceState:
    def __init__(self, *, llm_service_config: dict[str, object] | None = None) -> None:
        self._state = TerminusSpecTraceReplayLLMState(llm_service_config=llm_service_config)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._state.handle_request(path=path, headers=headers, payload=payload)

    def snapshot(self, *, include_events: bool = True) -> dict[str, Any]:
        return self._state.snapshot(include_events=include_events)

    def reset(self) -> None:
        self._state.reset()

    def restore(self, *, consumed_response_count: int) -> None:
        self._state.restore(consumed_response_count=consumed_response_count)

    def export_state(self) -> dict[str, Any]:
        return self._state.export_state()

    def import_state(self, state: dict[str, Any]) -> None:
        self._state.import_state(state)


def build_llm_service_registry() -> dict[str, type[LLMServiceState]]:
    return {
        "manual": ManualServiceState,
        "simulated": SimulatedServiceState,
        "simulated_for_iflow": SimulatedForIFlowServiceState,
        "iflow_trace_replay": IFlowTraceReplayServiceState,
        "mini_swe_trace_replay": MiniSWETraceReplayServiceState,
        "mini_swe_spec_trace_replay": MiniSWESpecTraceReplayServiceState,
        "claude_code_trace_replay": ClaudeCodeTraceReplayServiceState,
        "terminus_trace_replay": TerminusTraceReplayServiceState,
        "terminus_spec_trace_replay": TerminusSpecTraceReplayServiceState,
    }


@dataclass(frozen=True)
class RegisteredSandboxService:
    sandbox_id: str
    llm_service_type: str
    service_state: LLMServiceState
    llm_service_config: dict[str, object] | None = None


class BenchmarkLLMRouter:
    def __init__(self, registry: dict[str, type[LLMServiceState]] | None = None, telemetry=None) -> None:
        self._registry = registry or build_llm_service_registry()
        self._telemetry = telemetry
        self._lock = threading.Lock()
        self._services: dict[str, RegisteredSandboxService] = {}

    def register_sandbox(
        self,
        *,
        sandbox_id: str,
        llm_service_type: str,
        llm_service_config: dict[str, object] | None = None,
    ) -> None:
        if llm_service_type not in self._registry:
            raise ValueError(f"unsupported llm service type: {llm_service_type}")
        with self._lock:
            self._services[sandbox_id] = RegisteredSandboxService(
                sandbox_id=sandbox_id,
                llm_service_type=llm_service_type,
                service_state=self._registry[llm_service_type](llm_service_config=llm_service_config),
                llm_service_config=None if llm_service_config is None else dict(llm_service_config),
            )

    def unregister_sandbox(self, sandbox_id: str) -> None:
        with self._lock:
            self._services.pop(sandbox_id, None)

    def resolve_service(self, sandbox_id: str) -> RegisteredSandboxService:
        with self._lock:
            try:
                return self._services[sandbox_id]
            except KeyError as exc:
                raise ValueError(f"no llm service registered for sandbox_id={sandbox_id}") from exc

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        sandbox_id = _sandbox_id_from_request(headers, payload)
        service = self.resolve_service(sandbox_id)
        request_id = headers.get("X-Crab-Request-Id", "").strip()
        attributes = {
            "component": "llm_service",
            "request_id": request_id,
            "sandbox_id": sandbox_id,
            "path": path,
            "llm_service_type": service.llm_service_type,
        }
        operation = None if self._telemetry is None else start_operation(
            self._telemetry,
            "llm.service.request",
            attributes,
        )
        try:
            response = service.service_state.handle_request(path=path, headers=headers, payload=payload)
        except Exception:
            if operation is not None:
                operation.finish(status="failed")
            raise
        if operation is not None:
            operation.finish(status="succeeded")
        return response

    def handle_control_request(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        sandbox_id = str(payload.get("sandbox_id", "")).strip()
        if not sandbox_id:
            raise ValueError("sandbox_id is required")
        service = self.resolve_service(sandbox_id)
        if service.llm_service_type != "manual":
            raise ValueError(f"sandbox {sandbox_id} is not registered with llm_service_type=manual")
        manual_service = service.service_state
        assert isinstance(manual_service, ManualServiceState)
        return manual_service.handle_control(path=path, payload=payload)

    def _snapshot_registered_service(
        self,
        item: RegisteredSandboxService,
        *,
        include_events: bool,
    ) -> dict[str, Any]:
        return {
            "llm_service_type": item.llm_service_type,
            "llm_service_config": item.llm_service_config,
            "state": item.service_state.snapshot(include_events=include_events),
        }

    def snapshot(
        self,
        sandbox_id: str | None = None,
        *,
        include_events: bool = True,
    ) -> dict[str, Any] | None:
        if sandbox_id is not None:
            with self._lock:
                item = self._services.get(sandbox_id)
            if item is None:
                return None
            return self._snapshot_registered_service(item, include_events=include_events)

        with self._lock:
            services = list(self._services.items())
        return {
            registered_sandbox_id: self._snapshot_registered_service(item, include_events=include_events)
            for registered_sandbox_id, item in services
        }

    def registered_sandbox_count(self) -> int:
        with self._lock:
            return len(self._services)

    def reset_sandbox(self, sandbox_id: str) -> None:
        self.resolve_service(sandbox_id).service_state.reset()

    def restore_sandbox(self, sandbox_id: str, *, consumed_response_count: int) -> None:
        self.resolve_service(sandbox_id).service_state.restore(
            consumed_response_count=consumed_response_count
        )

    def copy_sandbox_state(self, *, source_sandbox_id: str, target_sandbox_id: str) -> None:
        source_service = self.resolve_service(source_sandbox_id)
        target_service = self.resolve_service(target_sandbox_id)
        if source_service.llm_service_type != target_service.llm_service_type:
            raise ValueError(
                "cannot copy llm service state across different service types: "
                f"{source_service.llm_service_type!r} != {target_service.llm_service_type!r}"
            )
        exporter = getattr(source_service.service_state, "export_state", None)
        importer = getattr(target_service.service_state, "import_state", None)
        if not callable(exporter) or not callable(importer):
            raise ValueError(
                f"llm service type {source_service.llm_service_type!r} does not support exact state copy"
            )
        importer(exporter())


@dataclass(frozen=True)
class BenchmarkLLMRouterClient:
    base_url: str
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_http_client",
            ThreadLocalHttpClient(self.base_url, timeout_seconds=self.timeout_seconds),
        )

    def register_sandbox(
        self,
        *,
        sandbox_id: str,
        llm_service_type: str,
        llm_service_config: dict[str, object] | None = None,
    ) -> dict[str, Any]:
        return self._post(
            "/control/register",
            {
                "sandbox_id": sandbox_id,
                "llm_service_type": llm_service_type,
                "llm_service_config": llm_service_config,
            },
        )

    def unregister_sandbox(self, sandbox_id: str) -> dict[str, Any]:
        return self._post("/control/unregister", {"sandbox_id": sandbox_id})

    def snapshot(self, sandbox_id: str | None = None) -> dict[str, Any] | None:
        query = "" if sandbox_id is None else "?" + urlencode({"sandbox_id": sandbox_id})
        payload = self._get(f"/control/state{query}")
        state = payload.get("state")
        if state is None:
            return None
        if isinstance(state, dict):
            return dict(state)
        raise ValueError("router state response must be an object")

    def reset_sandbox(self, sandbox_id: str) -> dict[str, Any]:
        return self._post("/control/reset", {"sandbox_id": sandbox_id})

    def restore_sandbox(self, sandbox_id: str, *, consumed_response_count: int) -> dict[str, Any]:
        return self._post(
            "/control/restore",
            {
                "sandbox_id": sandbox_id,
                "consumed_response_count": consumed_response_count,
            },
        )

    def copy_sandbox_state(self, *, source_sandbox_id: str, target_sandbox_id: str) -> dict[str, Any]:
        return self._post(
            "/control/copy_state",
            {
                "source_sandbox_id": source_sandbox_id,
                "target_sandbox_id": target_sandbox_id,
            },
        )

    def close(self) -> None:
        self._http_client.close()

    def _get(self, path: str) -> dict[str, Any]:
        return self._request("GET", path)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, payload=payload)

    def _request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            if method == "GET":
                return self._http_client.get_json(path)
            return self._http_client.post_json(path, payload or {})
        except HttpStatusError as exc:
            raise RuntimeError(f"router request failed status={exc.status_code} path={path}") from exc


def serve_benchmark_llm_router(
    *,
    host: str,
    port: int,
    registry: dict[str, type[LLMServiceState]] | None = None,
    telemetry=None,
    max_workers: int | None = None,
) -> PooledHTTPServer:
    router = BenchmarkLLMRouter(registry=registry, telemetry=telemetry)

    class RouterHandler(BaseHTTPRequestHandler):
        benchmark_llm_router = router
        protocol_version = "HTTP/1.1"

        def end_headers(self) -> None:
            self.send_header("Connection", "close")
            self.close_connection = True
            super().end_headers()

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            result = _JSON_CODEC.loads(self.rfile.read(length))
            if not isinstance(result, dict):
                raise ValueError("request body must decode to a JSON object")
            return dict(result)

        def _write_json(self, payload: dict[str, Any], *, code: int = 200) -> None:
            # Check if the response should be streamed as SSE (Anthropic Messages API streaming)
            if payload.pop("_streaming", False):
                self._write_anthropic_sse(payload)
                return
            body = _JSON_CODEC.dumps_bytes(payload)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write_anthropic_sse(self, response: dict[str, Any]) -> None:
            """Convert an Anthropic Messages API response to SSE events."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def _sse(event: str, data: dict[str, Any]) -> None:
                line = f"event: {event}\ndata: {_JSON_CODEC.dumps(data)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()

            content_blocks = response.get("content", [])
            msg_start = dict(response)
            msg_start["content"] = []
            msg_start.pop("_streaming", None)
            _sse("message_start", {"type": "message_start", "message": msg_start})

            for idx, block in enumerate(content_blocks):
                block_type = block.get("type", "text")
                if block_type == "text":
                    _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {"type": "text", "text": ""},
                    })
                    _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "text_delta", "text": block.get("text", "")},
                    })
                elif block_type == "tool_use":
                    _sse("content_block_start", {
                        "type": "content_block_start",
                        "index": idx,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.get("id", ""),
                            "name": block.get("name", ""),
                            "input": {},
                        },
                    })
                    input_json = _JSON_CODEC.dumps(block.get("input", {}))
                    _sse("content_block_delta", {
                        "type": "content_block_delta",
                        "index": idx,
                        "delta": {"type": "input_json_delta", "partial_json": input_json},
                    })
                _sse("content_block_stop", {"type": "content_block_stop", "index": idx})

            _sse("message_delta", {
                "type": "message_delta",
                "delta": {
                    "stop_reason": response.get("stop_reason", "end_turn"),
                    "stop_sequence": response.get("stop_sequence"),
                },
                "usage": {"output_tokens": response.get("usage", {}).get("output_tokens", 0)},
            })
            _sse("message_stop", {"type": "message_stop"})

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._write_json(
                    {
                        "ok": True,
                        "registered_sandboxes": self.benchmark_llm_router.registered_sandbox_count(),
                    }
                )
                return
            if self.path == "/v1/models":
                self._write_json({
                    "data": [{"id": "claude-opus-4-6-20250318", "object": "model"}, {"id": "claude-opus-4-6", "object": "model"}, {"id": "claude-sonnet-4-6-20250514", "object": "model"}],
                    "object": "list",
                })
                return
            if self.path.startswith("/control/state"):
                query = parse_qs(urlparse(self.path).query)
                sandbox_id = next(iter(query.get("sandbox_id", [])), "")
                if sandbox_id:
                    snapshot = self.benchmark_llm_router.snapshot(sandbox_id=sandbox_id, include_events=False)
                    self._write_json({"ok": True, "state": snapshot})
                    return
                snapshot = self.benchmark_llm_router.snapshot(include_events=False)
                self._write_json({"ok": True, "state": snapshot})
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            request_path = self.path.split("?")[0]
            payload = self._read_json()
            try:
                if request_path in {"/v1/chat/completions", "/v1/messages", "/v1/messages/count_tokens"}:
                    response = self.benchmark_llm_router.handle_request(
                        path=request_path,
                        headers=dict(self.headers.items()),
                        payload=payload,
                    )
                    self._write_json(response)
                    return
                if self.path in {"/control/run_shell_command", "/control/final_response"}:
                    response = self.benchmark_llm_router.handle_control_request(path=self.path, payload=payload)
                    self._write_json(response)
                    return
                if self.path == "/control/register":
                    sandbox_id = str(payload.get("sandbox_id", "")).strip()
                    llm_service_type = str(payload.get("llm_service_type", "")).strip()
                    llm_service_config = payload.get("llm_service_config")
                    if not sandbox_id:
                        raise ValueError("sandbox_id is required")
                    if not llm_service_type:
                        raise ValueError("llm_service_type is required")
                    if llm_service_config is not None and not isinstance(llm_service_config, dict):
                        raise ValueError("llm_service_config must be an object when provided")
                    self.benchmark_llm_router.register_sandbox(
                        sandbox_id=sandbox_id,
                        llm_service_type=llm_service_type,
                        llm_service_config=None if llm_service_config is None else dict(llm_service_config),
                    )
                    self._write_json({"ok": True})
                    return
                if self.path == "/control/unregister":
                    sandbox_id = str(payload.get("sandbox_id", "")).strip()
                    if not sandbox_id:
                        raise ValueError("sandbox_id is required")
                    self.benchmark_llm_router.unregister_sandbox(sandbox_id)
                    self._write_json({"ok": True})
                    return
                if self.path == "/control/reset":
                    sandbox_id = str(payload.get("sandbox_id", "")).strip()
                    if not sandbox_id:
                        raise ValueError("sandbox_id is required")
                    self.benchmark_llm_router.reset_sandbox(sandbox_id)
                    self._write_json({"ok": True})
                    return
                if self.path == "/control/restore":
                    sandbox_id = str(payload.get("sandbox_id", "")).strip()
                    if not sandbox_id:
                        raise ValueError("sandbox_id is required")
                    consumed_response_count = int(payload.get("consumed_response_count", 0))
                    self.benchmark_llm_router.restore_sandbox(
                        sandbox_id,
                        consumed_response_count=consumed_response_count,
                    )
                    self._write_json({"ok": True})
                    return
                if self.path == "/control/copy_state":
                    source_sandbox_id = str(payload.get("source_sandbox_id", "")).strip()
                    target_sandbox_id = str(payload.get("target_sandbox_id", "")).strip()
                    if not source_sandbox_id:
                        raise ValueError("source_sandbox_id is required")
                    if not target_sandbox_id:
                        raise ValueError("target_sandbox_id is required")
                    self.benchmark_llm_router.copy_sandbox_state(
                        source_sandbox_id=source_sandbox_id,
                        target_sandbox_id=target_sandbox_id,
                    )
                    self._write_json({"ok": True})
                    return
                self.send_error(404)
            except ValueError as exc:
                self.send_error(400, str(exc))
            except Exception:
                logger.exception(
                    "Benchmark LLM router request failed: path=%s client=%s",
                    self.path,
                    self.client_address[0] if self.client_address else "unknown",
                )
                self.send_error(500, "internal server error")

        def log_message(self, format: str, *args) -> None:
            _ = (format, args)
            return

    server = PooledHTTPServer((host, port), RouterHandler, max_workers=max_workers)
    server.benchmark_llm_router = router  # type: ignore[attr-defined]
    return server


def _router_telemetry_from_args(args: argparse.Namespace):
    if not args.telemetry_jsonl:
        return None
    return build_configured_telemetry_sink(
        TelemetryConfig(
            enabled=True,
            jsonl_path=Path(args.telemetry_jsonl),
            keep_in_memory_copy=bool(args.telemetry_keep_in_memory_copy),
            detail_level=args.telemetry_detail_level,
            capture_command_output=bool(args.telemetry_capture_command_output),
            max_text_attribute_bytes=int(args.telemetry_max_text_attribute_bytes),
            writer_mode=str(args.telemetry_writer_mode),
            queue_capacity=int(args.telemetry_queue_capacity),
            batch_max_records=int(args.telemetry_batch_max_records),
            flush_interval_ms=int(args.telemetry_flush_interval_ms),
            overflow_policy=str(args.telemetry_overflow_policy),
            serializer=str(args.telemetry_serializer),
        ),
        default_attributes={"run_id": args.run_id} if args.run_id else None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LLM router")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--telemetry-jsonl")
    parser.add_argument("--run-id")
    parser.add_argument("--telemetry-detail-level", default="basic")
    parser.add_argument("--telemetry-capture-command-output", action="store_true")
    parser.add_argument("--telemetry-max-text-attribute-bytes", type=int, default=2048)
    parser.add_argument("--telemetry-keep-in-memory-copy", action="store_true")
    parser.add_argument("--telemetry-writer-mode", default=DEFAULT_TELEMETRY_WRITER_MODE)
    parser.add_argument("--telemetry-queue-capacity", type=int, default=DEFAULT_TELEMETRY_QUEUE_CAPACITY)
    parser.add_argument("--telemetry-batch-max-records", type=int, default=DEFAULT_TELEMETRY_BATCH_MAX_RECORDS)
    parser.add_argument("--telemetry-flush-interval-ms", type=int, default=DEFAULT_TELEMETRY_FLUSH_INTERVAL_MS)
    parser.add_argument("--telemetry-overflow-policy", default=DEFAULT_TELEMETRY_OVERFLOW_POLICY)
    parser.add_argument("--telemetry-serializer", default=DEFAULT_TELEMETRY_SERIALIZER)
    parser.add_argument("--max-workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    telemetry = _router_telemetry_from_args(args)
    server = serve_benchmark_llm_router(
        host=str(args.host),
        port=int(args.port),
        telemetry=telemetry,
        max_workers=int(args.max_workers),
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if telemetry is not None:
            telemetry.close()


if __name__ == "__main__":
    main()

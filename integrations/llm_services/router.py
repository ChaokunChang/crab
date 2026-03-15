from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol
from urllib.parse import parse_qs, urlparse

from integrations.llm_services.manual.service import ManualLLMState, handle_control_request
from integrations.llm_services.simulated.service import SimulatedLLMState, handle_request as handle_simulated_request
from integrations.llm_services.simulated_for_iflow.service import (
    SimulatedLLMState as SimulatedIFlowLLMState,
    handle_request as handle_iflow_simulated_request,
)


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
    return "simulated"


def validate_llm_service_type(*, provider: str, llm_service_type: str) -> None:
    openai_only = {"manual", "simulated_for_iflow"}
    supported = {"simulated", "manual", "simulated_for_iflow"}
    if llm_service_type not in supported:
        raise ValueError(f"unsupported llm service type: {llm_service_type}")
    if provider == "anthropic" and llm_service_type in openai_only:
        raise ValueError(f"llm_service_type={llm_service_type} only supports provider=openai")


class LLMServiceState(Protocol):
    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]: ...

    def snapshot(self) -> dict[str, Any]: ...


class SimulatedServiceState:
    def __init__(self) -> None:
        self._state = SimulatedLLMState()

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return handle_simulated_request(path=path, headers=headers, payload=payload, state=self._state, response_delay_ms=250)

    def snapshot(self) -> dict[str, Any]:
        return {"turns": self._state.snapshot()}


class ManualServiceState:
    def __init__(self) -> None:
        self._state = ManualLLMState()

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return self._state.next_response(path=path, headers=headers, payload=payload)

    def handle_control(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return handle_control_request(path=path, payload=payload, state=self._state)

    def snapshot(self) -> dict[str, Any]:
        return self._state.snapshot()


class SimulatedForIFlowServiceState:
    def __init__(self) -> None:
        self._state = SimulatedIFlowLLMState(response_delay_ms=250, max_tool_calls_before_finish=3)

    def handle_request(self, *, path: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        return handle_iflow_simulated_request(path=path, headers=headers, payload=payload, state=self._state)

    def snapshot(self) -> dict[str, Any]:
        return self._state.snapshot()


def build_llm_service_registry() -> dict[str, type[LLMServiceState]]:
    return {
        "manual": ManualServiceState,
        "simulated": SimulatedServiceState,
        "simulated_for_iflow": SimulatedForIFlowServiceState,
    }


@dataclass(frozen=True)
class RegisteredSandboxService:
    sandbox_id: str
    llm_service_type: str
    service_state: LLMServiceState


class BenchmarkLLMRouter:
    def __init__(self, registry: dict[str, type[LLMServiceState]] | None = None) -> None:
        self._registry = registry or build_llm_service_registry()
        self._lock = threading.Lock()
        self._services: dict[str, RegisteredSandboxService] = {}

    def register_sandbox(self, *, sandbox_id: str, llm_service_type: str) -> None:
        if llm_service_type not in self._registry:
            raise ValueError(f"unsupported llm service type: {llm_service_type}")
        with self._lock:
            self._services[sandbox_id] = RegisteredSandboxService(
                sandbox_id=sandbox_id,
                llm_service_type=llm_service_type,
                service_state=self._registry[llm_service_type](),
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
        return service.service_state.handle_request(path=path, headers=headers, payload=payload)

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

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                sandbox_id: {
                    "llm_service_type": item.llm_service_type,
                    "state": item.service_state.snapshot(),
                }
                for sandbox_id, item in self._services.items()
            }


def serve_benchmark_llm_router(*, host: str, port: int, registry: dict[str, type[LLMServiceState]] | None = None) -> ThreadingHTTPServer:
    router = BenchmarkLLMRouter(registry=registry)

    class RouterHandler(BaseHTTPRequestHandler):
        benchmark_llm_router = router

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _write_json(self, payload: dict[str, Any], *, code: int = 200) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._write_json({"ok": True, "registered_sandboxes": len(self.benchmark_llm_router.snapshot())})
                return
            if self.path.startswith("/control/state"):
                query = parse_qs(urlparse(self.path).query)
                sandbox_id = next(iter(query.get("sandbox_id", [])), "")
                snapshot = self.benchmark_llm_router.snapshot()
                if sandbox_id:
                    self._write_json({"ok": True, "state": snapshot.get(sandbox_id)})
                    return
                self._write_json({"ok": True, "state": snapshot})
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            payload = self._read_json()
            try:
                if self.path in {"/v1/chat/completions", "/v1/messages"}:
                    response = self.benchmark_llm_router.handle_request(
                        path=self.path,
                        headers=dict(self.headers.items()),
                        payload=payload,
                    )
                    self._write_json(response)
                    return
                if self.path in {"/control/run_shell_command", "/control/final_response"}:
                    response = self.benchmark_llm_router.handle_control_request(path=self.path, payload=payload)
                    self._write_json(response)
                    return
                self.send_error(404)
            except ValueError as exc:
                self.send_error(400, str(exc))

        def log_message(self, format: str, *args) -> None:
            _ = (format, args)
            return

    server = ThreadingHTTPServer((host, port), RouterHandler)
    server.benchmark_llm_router = router  # type: ignore[attr-defined]
    return server

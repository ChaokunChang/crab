from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from agent_cr import (
    AgentCRRequestInterceptor,
    CompositeRequestInterceptorHook,
    InMemoryRequestStateStore,
    InMemoryTelemetrySink,
    SandboxId,
    StorageConfig,
    TelemetryRequestInterceptorHook,
    build_default_system,
)
from simulated_agent.service import SimulatedLLMState, handle_request


class InterceptorTests(unittest.TestCase):
    def test_interceptor_tracks_request_state_and_emits_telemetry(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        telemetry = InMemoryTelemetrySink()
        hook = CompositeRequestInterceptorHook([TelemetryRequestInterceptorHook(telemetry)])
        llm_state = SimulatedLLMState()
        interceptor = AgentCRRequestInterceptor(
            upstream_transport=lambda path, headers, body: (
                200,
                [("Content-Type", "application/json")],
                json.dumps(
                    handle_request(
                        path=path,
                        headers=headers,
                        payload=json.loads(body.decode("utf-8")),
                        state=llm_state,
                    ),
                    sort_keys=True,
                ).encode("utf-8"),
            ),
            request_state_store=request_state_store,
            hook=hook,
        )
        _, _, body = interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-1", "X-Request-Id": "req-1"},
            body=json.dumps(
                {
                    "model": "simulated-openai",
                    "messages": [{"role": "user", "content": "continue"}],
                    "tools": [
                        {
                            "type": "function",
                            "function": {
                                "name": "show_pwd",
                                "description": "x",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                }
            ).encode("utf-8"),
        )
        payload = json.loads(body.decode("utf-8"))

        self.assertEqual(payload["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "show_pwd")
        state = request_state_store.get(SandboxId("sbx-1"))
        self.assertEqual(state.total_llm_requests, 1)
        self.assertEqual(state.completed_llm_requests, 1)
        self.assertEqual(state.active_llm_requests, 0)
        self.assertEqual(state.last_llm_provider, "openai")
        event_names = [name for name, _ in telemetry.events]
        self.assertEqual(event_names.count("request.start"), 1)
        self.assertEqual(event_names.count("request.end"), 1)

    def test_interceptor_notifies_system_scheduler_asynchronously(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_interceptor_") as tmp:
            system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
            )
            seen: list[str] = []
            seen_event = threading.Event()
            system.checkpoint_if_due = lambda sandbox_id: seen.append(str(sandbox_id)) or seen_event.set() or None  # type: ignore[method-assign]

            interceptor = AgentCRRequestInterceptor(
                upstream_transport=lambda path, headers, body: (
                    200,
                    [("Content-Type", "application/json")],
                    json.dumps(
                        handle_request(
                            path=path,
                            headers=headers,
                            payload=json.loads(body.decode("utf-8")),
                            state=SimulatedLLMState(),
                            response_delay_ms=50,
                        ),
                        sort_keys=True,
                    ).encode("utf-8"),
                ),
                request_state_store=system.request_state_store or InMemoryRequestStateStore(),
                on_state_change=system.notify_interceptor_state_change,
            )
            interceptor.intercept(
                path="/v1/messages",
                headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-2"},
                body=json.dumps(
                    {
                        "model": "simulated-anthropic",
                        "messages": [{"role": "user", "content": "continue"}],
                        "tools": [
                            {
                                "name": "read_workdir",
                                "description": "x",
                                "input_schema": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                ).encode("utf-8"),
            )

            self.assertTrue(seen_event.wait(5.0))
            self.assertIn("sbx-2", seen)


if __name__ == "__main__":
    unittest.main()

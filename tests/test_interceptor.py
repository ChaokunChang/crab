from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_cr import (
    AgentCRRequestInterceptor,
    CompositeRequestInterceptorHook,
    InMemoryRequestStateStore,
    InMemoryTelemetrySink,
    SandboxId,
    SandboxSnapshot,
    StorageConfig,
    TelemetryRequestInterceptorHook,
    build_default_system,
)
from agent_cr.models import utc_now
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

    def test_interceptor_notifies_system_scheduler(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_interceptor_") as tmp:
            system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
            )

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

            self.assertTrue(system.has_pending_interceptor_signal(SandboxId("sbx-2")))
            event_names = [name for name, _ in system.telemetry.events]
            self.assertIn("interceptor.state_changed", event_names)
            system.executor.shutdown()

    def test_interceptor_can_use_default_sandbox_id_when_header_missing(self) -> None:
        request_state_store = InMemoryRequestStateStore()
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
                    ),
                    sort_keys=True,
                ).encode("utf-8"),
            ),
            request_state_store=request_state_store,
            default_sandbox_id=SandboxId("sbx-default"),
        )

        _, _, body = interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body=json.dumps(
                {
                    "model": "simulated-openai",
                    "messages": [{"role": "user", "content": "continue"}],
                }
            ).encode("utf-8"),
        )

        payload = json.loads(body.decode("utf-8"))
        self.assertIn("choices", payload)
        state = request_state_store.get(SandboxId("sbx-default"))
        self.assertEqual(state.total_llm_requests, 1)
        self.assertEqual(state.completed_llm_requests, 1)
        self.assertEqual(state.active_llm_requests, 0)

    def test_interceptor_waits_for_system_checkpoint_flow(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_interceptor_system_") as tmp:
            system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
            )
            sandbox_id = system.sandbox_manager.launch("docker")
            system.inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sandbox_id,
                    runtime_name="docker",
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=False,
                    observed_at=utc_now(),
                )
            )
            system.start()

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
                        ),
                        sort_keys=True,
                    ).encode("utf-8"),
                ),
                request_state_store=system.request_state_store or InMemoryRequestStateStore(),
                response_gate_registry=system.response_gate_registry,
            )
            _, _, body = interceptor.intercept(
                path="/v1/chat/completions",
                headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": str(sandbox_id)},
                body=json.dumps(
                    {
                        "model": "simulated-openai",
                        "messages": [{"role": "user", "content": "continue"}],
                    }
                ).encode("utf-8"),
            )

            payload = json.loads(body.decode("utf-8"))
            self.assertIn("choices", payload)
            state = system.request_state_store.get(sandbox_id) if system.request_state_store is not None else None
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.active_llm_requests, 0)
            event_names = [name for name, _ in system.telemetry.events]
            self.assertIn("scheduler.evaluate", event_names)
            self.assertIn("executor.job_finished", event_names)
            self.assertEqual(system.sandbox_manager.describe(sandbox_id).status, "running")
            system.stop()
            system.executor.shutdown()


if __name__ == "__main__":
    unittest.main()

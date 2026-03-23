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
    SandboxSnapshot,
    SandboxResponseGateRegistry,
    StorageConfig,
    TelemetryRequestInterceptorHook,
    build_default_system,
)
from agent_cr.models import utc_now
from integrations.llm_services.simulated.service import SimulatedLLMState, handle_request


class InterceptorTests(unittest.TestCase):
    def test_build_default_system_disables_restore_validation_by_default_and_allows_opt_in(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_interceptor_builder_") as tmp:
            default_system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
            )
            enabled_system = build_default_system(
                storage_root=tmp,
                runtime="docker",
                storage_config=StorageConfig(root_dir=Path(tmp)),
                enforce_restore_checkpoint_validation=True,
            )

            self.assertFalse(default_system.enforce_restore_checkpoint_validation)
            self.assertTrue(enabled_system.enforce_restore_checkpoint_validation)
            default_system.executor.shutdown()
            enabled_system.executor.shutdown()

    def test_response_gate_registry_tracks_request_identity(self) -> None:
        registry = SandboxResponseGateRegistry()
        sandbox_id = SandboxId("sbx-gate")
        registry.enable()

        generation = registry.arm(sandbox_id, "req-1")

        self.assertEqual(generation, 1)
        pending = registry.get_pending(sandbox_id)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.request_id, "req-1")
        self.assertEqual(pending.generation, 1)

    def test_response_gate_registry_targeted_release_does_not_release_newer_request(self) -> None:
        registry = SandboxResponseGateRegistry()
        sandbox_id = SandboxId("sbx-gate")
        registry.enable()

        first_generation = registry.arm(sandbox_id, "req-1")
        second_generation = registry.arm(sandbox_id, "req-2")

        self.assertFalse(
            registry.release_pending(sandbox_id, request_id="req-1", generation=first_generation),
        )
        pending = registry.get_pending(sandbox_id)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertEqual(pending.request_id, "req-2")
        self.assertEqual(pending.generation, second_generation)
        self.assertTrue(
            registry.release_pending(sandbox_id, request_id="req-2", generation=second_generation),
        )
        self.assertIsNone(registry.get_pending(sandbox_id))

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

    def test_interceptor_resolves_sandbox_id_from_client_host_and_overrides_forwarded_header(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        forwarded_headers: dict[str, str] = {}

        def _upstream_transport(path, headers, body):
            forwarded_headers.update(headers)
            return (
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
            )

        interceptor = AgentCRRequestInterceptor(
            upstream_transport=_upstream_transport,
            request_state_store=request_state_store,
            sandbox_id_resolver=lambda client_host, headers, body: "fork-1" if client_host == "10.250.0.8" else None,
        )

        interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "source-1", "X-Request-Id": "req-1"},
            body=json.dumps({"model": "simulated-openai", "messages": [{"role": "user", "content": "continue"}]}).encode(
                "utf-8"
            ),
            client_host="10.250.0.8",
        )

        self.assertEqual(request_state_store.get(SandboxId("fork-1")).total_llm_requests, 1)
        self.assertEqual(request_state_store.get(SandboxId("source-1")).total_llm_requests, 0)
        self.assertEqual(forwarded_headers["X-Agent-Sandbox-Id"], "fork-1")
        self.assertEqual(forwarded_headers["X-Request-Id"], "req-1")
        self.assertEqual(forwarded_headers["X-AgentCR-Request-Id"], "req-1")

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

    def test_interceptor_rejects_request_when_sandbox_identity_is_missing(self) -> None:
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
        )

        with self.assertRaisesRegex(ValueError, "missing sandbox identity"):
            interceptor.intercept(
                path="/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                body=json.dumps(
                    {
                        "model": "simulated-openai",
                        "messages": [{"role": "user", "content": "continue"}],
                    }
                ).encode("utf-8"),
            )

    def test_interceptor_resolver_maps_distinct_client_hosts_to_distinct_sandboxes(self) -> None:
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
            sandbox_id_resolver=lambda client_host, headers, body: {
                "172.17.0.240": "sbx-iflow-a",
                "172.17.0.241": "sbx-iflow-b",
            }.get("" if client_host is None else client_host),
        )

        interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"model": "simulated-openai", "messages": [{"role": "user", "content": "continue"}]}).encode(
                "utf-8"
            ),
            client_host="172.17.0.240",
        )
        interceptor.intercept(
            path="/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"model": "simulated-openai", "messages": [{"role": "user", "content": "continue"}]}).encode(
                "utf-8"
            ),
            client_host="172.17.0.241",
        )

        self.assertEqual(request_state_store.get(SandboxId("sbx-iflow-a")).total_llm_requests, 1)
        self.assertEqual(request_state_store.get(SandboxId("sbx-iflow-b")).total_llm_requests, 1)

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

    def test_interceptor_releases_buffered_response_for_matching_restored_request(self) -> None:
        request_state_store = InMemoryRequestStateStore()
        response_gate_registry = SandboxResponseGateRegistry()
        response_gate_registry.enable()
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
            response_gate_registry=response_gate_registry,
        )

        response_body: dict[str, object] = {}
        finished = threading.Event()

        def _run_request() -> None:
            _, _, body = interceptor.intercept(
                path="/v1/chat/completions",
                headers={"Content-Type": "application/json", "X-Agent-Sandbox-Id": "sbx-restore", "X-Request-Id": "req-live"},
                body=json.dumps(
                    {
                        "model": "simulated-openai",
                        "messages": [{"role": "user", "content": "continue"}],
                    }
                ).encode("utf-8"),
            )
            response_body["payload"] = json.loads(body.decode("utf-8"))
            finished.set()

        thread = threading.Thread(target=_run_request)
        thread.start()
        pending = None
        for _ in range(50):
            pending = response_gate_registry.get_pending(SandboxId("sbx-restore"))
            if pending is not None:
                break
            finished.wait(0.01)
        self.assertIsNotNone(pending)
        assert pending is not None
        self.assertTrue(
            response_gate_registry.release_pending(
                SandboxId("sbx-restore"),
                request_id="req-live",
                generation=pending.generation,
            )
        )
        thread.join(timeout=2.0)

        self.assertTrue(finished.is_set())
        self.assertIn("payload", response_body)
        state = request_state_store.get(SandboxId("sbx-restore"))
        self.assertEqual(state.completed_llm_requests, 1)


if __name__ == "__main__":
    unittest.main()

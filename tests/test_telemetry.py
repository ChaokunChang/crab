from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_cr import (
    AgentCRRequestInterceptor,
    CompositeTelemetrySink,
    InMemoryRequestStateStore,
    InMemoryTelemetrySink,
    JsonlTelemetrySink,
)


class TelemetryTests(unittest.TestCase):
    def test_jsonl_telemetry_sink_writes_event_and_metric_records(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_") as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            sink = JsonlTelemetrySink(path)

            sink.emit_event("sandbox.command", {"sandbox_id": "sbx-1", "success": True})
            sink.emit_metric("checkpoint.total_ms", 12.5, {"sandbox_id": "sbx-1"})

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["kind"], "event")
        self.assertEqual(rows[0]["name"], "sandbox.command")
        self.assertEqual(rows[1]["kind"], "metric")
        self.assertEqual(rows[1]["name"], "checkpoint.total_ms")
        self.assertEqual(rows[1]["value"], 12.5)

    def test_composite_telemetry_sink_fans_out_to_multiple_sinks(self) -> None:
        in_memory = InMemoryTelemetrySink()
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_") as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            composite = CompositeTelemetrySink([in_memory, JsonlTelemetrySink(path)])

            composite.emit_event("request.start", {"sandbox_id": "sbx-1"})

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(in_memory.events[0][0], "request.start")
        self.assertEqual(rows[0]["name"], "request.start")

    def test_interceptor_emits_total_request_latency_metric(self) -> None:
        telemetry = InMemoryTelemetrySink()
        interceptor = AgentCRRequestInterceptor(
            upstream_transport=lambda path, headers, body: (200, [], b"{}"),
            request_state_store=InMemoryRequestStateStore(),
            telemetry=telemetry,
        )

        interceptor.intercept(
            path="/v1/chat/completions",
            headers={"X-Agent-Sandbox-Id": "sbx-1"},
            body=b"{}",
        )

        metric_names = [name for name, _, _ in telemetry.metrics]
        self.assertIn("llm.request_total_ms", metric_names)


if __name__ == "__main__":
    unittest.main()

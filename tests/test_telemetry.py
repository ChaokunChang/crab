from __future__ import annotations

import json
import multiprocessing
import tempfile
import time
import unittest
from pathlib import Path

from agent_cr import (
    AgentCRRequestInterceptor,
    AsyncJsonlTelemetrySink,
    CompositeTelemetrySink,
    InMemoryRequestStateStore,
    InMemoryTelemetrySink,
    JsonlTelemetrySink,
    start_operation,
)


def _emit_jsonl_records(path_str: str, prefix: str) -> None:
    sink = JsonlTelemetrySink(Path(path_str))
    for index in range(25):
        sink.emit_event("worker.event", {"worker": prefix, "index": index})


class SlowJsonlTelemetrySink(JsonlTelemetrySink):
    def write_records(self, records: list[dict[str, object]]) -> None:
        time.sleep(0.02)
        super().write_records(records)


class SlowInMemoryTelemetrySink(InMemoryTelemetrySink):
    def emit_event(self, name: str, attributes: dict[str, object]) -> None:
        time.sleep(0.02)
        super().emit_event(name, attributes)

    def emit_metric(
        self,
        name: str,
        value: float,
        attributes: dict[str, object] | None = None,
    ) -> None:
        time.sleep(0.02)
        super().emit_metric(name, value, attributes)


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

    def test_jsonl_telemetry_sink_supports_cross_process_writes(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_") as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            processes = [
                multiprocessing.Process(target=_emit_jsonl_records, args=(str(path), "a")),
                multiprocessing.Process(target=_emit_jsonl_records, args=(str(path), "b")),
            ]

            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=5.0)
                self.assertEqual(process.exitcode, 0)

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 50)
        workers = {row["attributes"]["worker"] for row in rows}
        self.assertEqual(workers, {"a", "b"})

    def test_async_jsonl_telemetry_sink_drops_new_records_when_queue_is_full(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_") as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            sink = AsyncJsonlTelemetrySink(
                SlowJsonlTelemetrySink(path),
                queue_capacity=1,
                batch_max_records=1,
                flush_interval_ms=5,
                overflow_policy="drop_new",
            )
            for index in range(50):
                sink.emit_event("burst.event", {"index": index})
            sink.close()

            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertTrue(any(row["name"] == "telemetry.writer.dropped" for row in rows))
        written_events = [row for row in rows if row["name"] == "burst.event"]
        self.assertLess(len(written_events), 50)

    def test_telemetry_operation_duration_excludes_emit_overhead(self) -> None:
        telemetry = SlowInMemoryTelemetrySink()

        operation = start_operation(telemetry, "slow.operation", {"sandbox_id": "sbx-1"})
        time.sleep(0.01)
        duration_ms = operation.finish()

        self.assertGreater(duration_ms, 5.0)
        self.assertLess(duration_ms, 35.0)


if __name__ == "__main__":
    unittest.main()

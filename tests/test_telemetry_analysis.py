from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_cr.telemetry_analysis import analyze_telemetry_file, generate_report_bundle


class TelemetryAnalysisTests(unittest.TestCase):
    def test_analyze_telemetry_file_prefers_primary_run_and_canonical_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_analysis_") as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            records = [
                {
                    "timestamp": "2026-03-23T01:00:00+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.start",
                    "attributes": {"run_id": "run-a", "sandbox_id": "sbx-1", "task_id": "task-1"},
                },
                {
                    "timestamp": "2026-03-23T01:00:01+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:01+08:00",
                    "kind": "metric",
                    "name": "checkpoint.flow.duration_ms",
                    "value": 12.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "checkpoint",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:02+08:00",
                    "kind": "metric",
                    "name": "checkpoint.total_ms",
                    "value": 11.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "checkpoint",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:03+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.duration_ms",
                    "value": 120.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "benchmark",
                        "agent_type": "iflow",
                        "llm_service_type": "iflow_trace_replay",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:04+08:00",
                    "kind": "metric",
                    "name": "llm.agentcr_delay_ms",
                    "value": 30.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "interceptor",
                        "request_id": "req-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:05+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.duration_ms",
                    "value": 999.0,
                    "attributes": {
                        "run_id": "run-b",
                        "sandbox_id": "sbx-2",
                        "task_id": "task-2",
                        "component": "benchmark",
                    },
                },
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            analysis = analyze_telemetry_file(path)

            self.assertEqual(analysis.run_id, "run-a")
            summary_by_name = {item.metric_name: item for item in analysis.operation_summaries}
            self.assertIn("checkpoint.flow.duration_ms", summary_by_name)
            self.assertEqual(summary_by_name["checkpoint.flow.duration_ms"].source_metric_name, "checkpoint.flow.duration_ms")
            self.assertEqual(summary_by_name["checkpoint.flow.duration_ms"].count, 1)
            self.assertEqual(len(analysis.task_summaries), 1)
            self.assertEqual(analysis.task_summaries[0].task_id, "task-1")
            self.assertAlmostEqual(analysis.task_summaries[0].metrics["benchmark.task.duration_ms"], 120.0)
            self.assertEqual(analysis.distinct_requests, 1)

    def test_generate_report_bundle_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_report_") as tmp:
            root = Path(tmp)
            telemetry_path = root / "telemetry.jsonl"
            telemetry_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-03-23T01:00:00+08:00",
                                "kind": "metric",
                                "name": "benchmark.task.duration_ms",
                                "value": 50.0,
                                "attributes": {
                                    "run_id": "run-a",
                                    "sandbox_id": "sbx-1",
                                    "task_id": "task-1",
                                    "component": "benchmark",
                                    "agent_type": "iflow",
                                    "llm_service_type": "iflow_trace_replay",
                                },
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-03-23T01:00:01+08:00",
                                "kind": "metric",
                                "name": "llm.interceptor_total_ms",
                                "value": 10.0,
                                "attributes": {
                                    "run_id": "run-a",
                                    "sandbox_id": "sbx-1",
                                    "task_id": "task-1",
                                    "component": "interceptor",
                                },
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            analysis = generate_report_bundle(telemetry_path, output_dir=root / "report")

            self.assertEqual(analysis.run_id, "run-a")
            self.assertTrue((root / "report" / "summary.json").exists())
            self.assertTrue((root / "report" / "report.html").exists())
            self.assertTrue((root / "report" / "operation_summary.csv").exists())
            self.assertTrue((root / "report" / "task_summary.csv").exists())
            self.assertTrue((root / "report" / "slow_operations.csv").exists())


if __name__ == "__main__":
    unittest.main()

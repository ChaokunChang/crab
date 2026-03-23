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


    def test_exclude_failed_tasks_filters_sandboxes_with_zero_success_ratio(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_filter_") as tmp:
            path = Path(tmp) / "telemetry.jsonl"
            records = [
                # Sandbox sbx-1 succeeds (success_ratio=1.0)
                {
                    "timestamp": "2026-03-23T01:00:00+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.duration_ms",
                    "value": 100.0,
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
                    "timestamp": "2026-03-23T01:00:01+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.success_ratio",
                    "value": 1.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "benchmark",
                    },
                },
                # Sandbox sbx-2 fails (success_ratio=0.0)
                {
                    "timestamp": "2026-03-23T01:00:02+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.duration_ms",
                    "value": 200.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-2",
                        "task_id": "task-1",
                        "component": "benchmark",
                        "agent_type": "iflow",
                        "llm_service_type": "iflow_trace_replay",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:03+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.success_ratio",
                    "value": 0.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-2",
                        "task_id": "task-1",
                        "component": "benchmark",
                    },
                },
                # Sandbox sbx-3 succeeds with a different task
                {
                    "timestamp": "2026-03-23T01:00:04+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-3",
                        "task_id": "task-2",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:05+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-3",
                        "task_id": "task-2",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:05+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.duration_ms",
                    "value": 80.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-3",
                        "task_id": "task-2",
                        "component": "benchmark",
                        "agent_type": "iflow",
                        "llm_service_type": "iflow_trace_replay",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:06+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.success_ratio",
                    "value": 1.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-3",
                        "task_id": "task-2",
                        "component": "benchmark",
                    },
                },
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            # Without filtering: all 3 sandboxes appear
            analysis_all = analyze_telemetry_file(path, exclude_failed_tasks=False)
            self.assertEqual(analysis_all.distinct_sandboxes, 3)
            self.assertFalse(analysis_all.exclude_failed_tasks)
            self.assertEqual(len(analysis_all.excluded_sandbox_task_pairs), 0)
            all_task_ids = {t.task_id for t in analysis_all.task_summaries}
            self.assertIn("task-1", all_task_ids)

            # With filtering: sbx-2 is excluded
            analysis_filtered = analyze_telemetry_file(path, exclude_failed_tasks=True)
            self.assertTrue(analysis_filtered.exclude_failed_tasks)
            self.assertEqual(len(analysis_filtered.excluded_sandbox_task_pairs), 1)
            self.assertEqual(analysis_filtered.excluded_sandbox_task_pairs[0], ("sbx-2", "task-1"))
            self.assertEqual(analysis_filtered.distinct_sandboxes, 2)
            self.assertNotIn("sbx-2", {
                sid
                for task in analysis_filtered.task_summaries
                for sid in task.sandbox_ids
            })
            # task-1 still appears from sbx-1
            filtered_task_ids = {t.task_id for t in analysis_filtered.task_summaries}
            self.assertIn("task-1", filtered_task_ids)
            self.assertIn("task-2", filtered_task_ids)
            # task-1 duration should be 100 (from sbx-1 only), not averaged with sbx-2's 200
            task1 = next(t for t in analysis_filtered.task_summaries if t.task_id == "task-1")
            self.assertAlmostEqual(task1.metrics["benchmark.task.duration_ms"], 100.0)

    def test_exclude_failed_tasks_report_bundle_includes_exclusion_info(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_filter_report_") as tmp:
            root = Path(tmp)
            path = root / "telemetry.jsonl"
            records = [
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
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:01+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.success_ratio",
                    "value": 0.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "benchmark",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:02+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.duration_ms",
                    "value": 30.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-2",
                        "task_id": "task-2",
                        "component": "benchmark",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:03+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.success_ratio",
                    "value": 1.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-2",
                        "task_id": "task-2",
                        "component": "benchmark",
                    },
                },
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            analysis = generate_report_bundle(
                path, output_dir=root / "report", exclude_failed_tasks=True,
            )

            self.assertTrue(analysis.exclude_failed_tasks)
            self.assertEqual(len(analysis.excluded_sandbox_task_pairs), 1)
            self.assertEqual(analysis.distinct_sandboxes, 1)
            # Check report HTML contains exclusion info
            html = (root / "report" / "report.html").read_text(encoding="utf-8")
            self.assertIn("Excluded Failed Sandboxes", html)
            self.assertIn("sbx-1", html)
            # Check summary JSON
            summary = json.loads((root / "report" / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["exclude_failed_tasks"])
            self.assertEqual(len(summary["excluded_sandbox_task_pairs"]), 1)


if __name__ == "__main__":
    unittest.main()

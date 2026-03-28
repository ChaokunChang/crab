from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.telemetry_analysis import analyze_telemetry_file, generate_report_bundle
from benchmarks.telemetry_analysis import report as telemetry_report


class TelemetryAnalysisTests(unittest.TestCase):
    def test_counter_window_series_converts_cumulative_counters_to_per_window_values(self) -> None:
        self.assertEqual(
            telemetry_report._counter_window_series([(1.0, 100.0), (2.0, 140.0), (3.0, 130.0), (4.0, 170.0)]),
            [(1.0, 0.0), (2.0, 40.0), (3.0, 0.0), (4.0, 40.0)],
        )

    def test_aggregate_window_series_averages_values_within_each_bucket(self) -> None:
        self.assertEqual(
            telemetry_report._aggregate_window_series(
                [(1.0, 10.0), (4.0, 20.0), (9.0, 30.0), (13.0, 50.0)],
                window_size_seconds=10.0,
            ),
            [(9.0, 20.0), (13.0, 50.0)],
        )

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
            self.assertEqual(analysis.task_summaries[0].sandbox_id, "sbx-1")
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

            analysis = generate_report_bundle(telemetry_path, output_dir=root / "report", window_size_seconds=30.0)

            self.assertEqual(analysis.run_id, "run-a")
            self.assertTrue((root / "report" / "summary.json").exists())
            self.assertTrue((root / "report" / "report.html").exists())
            self.assertTrue((root / "report" / "phase_overview.csv").exists())
            self.assertTrue((root / "report" / "operation_summary.csv").exists())
            self.assertTrue((root / "report" / "task_summary.csv").exists())
            self.assertTrue((root / "report" / "slow_operations.csv").exists())
            self.assertTrue((root / "report" / "checkpoint_analysis.csv").exists())
            self.assertTrue((root / "report" / "restore_analysis.csv").exists())
            self.assertTrue((root / "report" / "resource_summary.csv").exists())

    def test_detailed_analysis_is_limited_to_run_phase_when_phase_markers_exist(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_run_scope_") as tmp:
            root = Path(tmp)
            path = root / "telemetry.jsonl"
            records = [
                {
                    "timestamp": "2026-03-23T01:00:00+08:00",
                    "kind": "metric",
                    "name": "benchmark.phase.setup.configured_max_workers",
                    "value": 8.0,
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "setup",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "component": "benchmark",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:00+08:00",
                    "kind": "event",
                    "name": "benchmark.phase.setup.start",
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "setup",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "configured_max_workers": 8,
                        "component": "benchmark",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:10+08:00",
                    "kind": "event",
                    "name": "benchmark.phase.setup.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "setup",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "configured_max_workers": 8,
                        "component": "benchmark",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:05+08:00",
                    "kind": "metric",
                    "name": "sandbox.runtime_pause.duration_ms",
                    "value": 999.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "runtime",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:00+08:00",
                    "kind": "metric",
                    "name": "benchmark.phase.run.configured_max_workers",
                    "value": 16.0,
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "run",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "component": "benchmark",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:00+08:00",
                    "kind": "event",
                    "name": "benchmark.phase.run.start",
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "run",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "configured_max_workers": 16,
                        "component": "benchmark",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:30+08:00",
                    "kind": "metric",
                    "name": "checkpoint.flow.duration_ms",
                    "value": 12.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "system",
                        "status": "skipped",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:35+08:00",
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
                    },
                },
                {
                    "timestamp": "2026-03-23T01:02:00+08:00",
                    "kind": "event",
                    "name": "benchmark.phase.run.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "run",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "configured_max_workers": 16,
                        "component": "benchmark",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:03:00+08:00",
                    "kind": "metric",
                    "name": "benchmark.phase.verification.configured_max_workers",
                    "value": 4.0,
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "verification",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "component": "benchmark",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:03:00+08:00",
                    "kind": "event",
                    "name": "benchmark.phase.verification.start",
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "verification",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "configured_max_workers": 4,
                        "component": "benchmark",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:03:20+08:00",
                    "kind": "event",
                    "name": "benchmark.phase.verification.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "phase": "verification",
                        "phase_scope": "run",
                        "sandbox_count": 2,
                        "configured_max_workers": 4,
                        "component": "benchmark",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:03:05+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.verify.duration_ms",
                    "value": 321.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "benchmark",
                    },
                },
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            analysis = analyze_telemetry_file(path)

            self.assertEqual(analysis.detail_scope_name, "run")
            self.assertEqual([item.phase for item in analysis.phase_overview], ["setup", "run", "verification"])
            self.assertEqual(analysis.detail_started_at, "2026-03-23T01:01:00+08:00")
            self.assertEqual(analysis.detail_finished_at, "2026-03-23T01:02:00+08:00")
            summary_by_name = {item.metric_name: item for item in analysis.operation_summaries}
            self.assertIn("checkpoint.flow.duration_ms", summary_by_name)
            self.assertIn("benchmark.task.duration_ms", summary_by_name)
            self.assertNotIn("sandbox.runtime_pause.duration_ms", summary_by_name)
            self.assertNotIn("benchmark.task.verify.duration_ms", summary_by_name)
            self.assertFalse(any(name.startswith("benchmark.phase.") for name in summary_by_name))

            bundle = generate_report_bundle(path, output_dir=root / "report")
            operation_csv = (root / "report" / "operation_summary.csv").read_text(encoding="utf-8")
            self.assertIn("checkpoint.flow.duration_ms", operation_csv)
            self.assertNotIn("sandbox.runtime_pause.duration_ms", operation_csv)
            self.assertNotIn("benchmark.task.verify.duration_ms", operation_csv)
            html = (root / "report" / "report.html").read_text(encoding="utf-8")
            self.assertIn("Phase Overview", html)
            self.assertIn("filtered to run-phase telemetry only", html)
            phase_csv = (root / "report" / "phase_overview.csv").read_text(encoding="utf-8")
            self.assertIn("setup", phase_csv)
            self.assertIn("run", phase_csv)
            self.assertIn("verification", phase_csv)
            self.assertEqual(bundle.detail_scope_name, "run")


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
            self.assertIn("sbx-2", {t.sandbox_id for t in analysis_all.task_summaries})

            # With filtering: sbx-2 is excluded
            analysis_filtered = analyze_telemetry_file(path, exclude_failed_tasks=True)
            self.assertTrue(analysis_filtered.exclude_failed_tasks)
            self.assertEqual(len(analysis_filtered.excluded_sandbox_task_pairs), 1)
            self.assertEqual(analysis_filtered.excluded_sandbox_task_pairs[0], ("sbx-2", "task-1"))
            self.assertEqual(analysis_filtered.distinct_sandboxes, 2)
            self.assertNotIn("sbx-2", {task.sandbox_id for task in analysis_filtered.task_summaries})
            # task-1 still appears from sbx-1
            filtered_task_ids = {t.task_id for t in analysis_filtered.task_summaries}
            self.assertIn("task-1", filtered_task_ids)
            self.assertIn("task-2", filtered_task_ids)
            # task-1 duration should be 100 (from sbx-1 only), not averaged with sbx-2's 200
            task1 = next(t for t in analysis_filtered.task_summaries if t.task_id == "task-1")
            self.assertEqual(task1.sandbox_id, "sbx-1")
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

    def test_restore_and_summary_tables_are_reported_per_sandbox(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_sandbox_summary_") as tmp:
            root = Path(tmp)
            path = root / "telemetry.jsonl"
            records = [
                {
                    "timestamp": "2026-03-23T01:00:00+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.duration_ms",
                    "value": 100.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "shared-task",
                        "component": "benchmark",
                        "agent_type": "iflow",
                        "llm_service_type": "iflow_trace_replay",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:01+08:00",
                    "kind": "metric",
                    "name": "benchmark.task.duration_ms",
                    "value": 200.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-2",
                        "task_id": "shared-task",
                        "component": "benchmark",
                        "agent_type": "iflow",
                        "llm_service_type": "iflow_trace_replay",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:00+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "shared-task",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "ckpt-job-1",
                        "checkpoint_scope": "full",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:01+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "shared-task",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "ckpt-job-1",
                        "checkpoint_scope": "full",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:02+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-2",
                        "sandbox_id": "sbx-2",
                        "task_id": "shared-task",
                        "checkpoint_id": "ckpt-2",
                        "job_id": "ckpt-job-2",
                        "checkpoint_scope": "full",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:03+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-2",
                        "sandbox_id": "sbx-2",
                        "task_id": "shared-task",
                        "checkpoint_id": "ckpt-2",
                        "job_id": "ckpt-job-2",
                        "checkpoint_scope": "full",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:04+08:00",
                    "kind": "event",
                    "name": "restore.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "shared-task",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:05+08:00",
                    "kind": "event",
                    "name": "restore.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "shared-task",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-1",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:06+08:00",
                    "kind": "event",
                    "name": "restore.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-2",
                        "sandbox_id": "sbx-2",
                        "task_id": "shared-task",
                        "checkpoint_id": "ckpt-2",
                        "job_id": "job-2",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:07+08:00",
                    "kind": "event",
                    "name": "restore.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-2",
                        "sandbox_id": "sbx-2",
                        "task_id": "shared-task",
                        "checkpoint_id": "ckpt-2",
                        "job_id": "job-2",
                        "status": "succeeded",
                    },
                },
            ]
            path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

            analysis = generate_report_bundle(path, output_dir=root / "report", window_size_seconds=15.0)

            self.assertEqual(len(analysis.task_summaries), 2)
            self.assertEqual({item.sandbox_id for item in analysis.task_summaries}, {"sbx-1", "sbx-2"})
            self.assertEqual({item.task_id for item in analysis.task_summaries}, {"shared-task"})
            self.assertIsNotNone(analysis.checkpoint_analysis)
            assert analysis.checkpoint_analysis is not None
            self.assertEqual(len(analysis.checkpoint_analysis.per_task), 2)
            self.assertEqual({item.sandbox_id for item in analysis.checkpoint_analysis.per_task}, {"sbx-1", "sbx-2"})
            self.assertIsNotNone(analysis.restore_analysis)
            assert analysis.restore_analysis is not None
            self.assertEqual(len(analysis.restore_analysis.per_task), 2)
            self.assertEqual({item.sandbox_id for item in analysis.restore_analysis.per_task}, {"sbx-1", "sbx-2"})

            summary_csv = (root / "report" / "task_summary.csv").read_text(encoding="utf-8")
            self.assertIn("sandbox_id,task_id,agent_type,llm_service_type", summary_csv)
            self.assertIn("sbx-1,shared-task", summary_csv)
            self.assertIn("sbx-2,shared-task", summary_csv)

            checkpoint_csv = (root / "report" / "checkpoint_per_task.csv").read_text(encoding="utf-8")
            self.assertIn("sandbox_id,task_id,total_count", checkpoint_csv)
            self.assertIn("sbx-1,shared-task,1", checkpoint_csv)
            self.assertIn("sbx-2,shared-task,1", checkpoint_csv)

            restore_csv = (root / "report" / "restore_per_task.csv").read_text(encoding="utf-8")
            self.assertIn("sandbox_id,task_id,total_count", restore_csv)
            self.assertIn("sbx-1,shared-task,1", restore_csv)
            self.assertIn("sbx-2,shared-task,1", restore_csv)

            html = (root / "report" / "report.html").read_text(encoding="utf-8")
            self.assertIn("Per-Sandbox Summary", html)
            self.assertIn("Checkpoint Per Sandbox", html)
            self.assertIn("Restore Per Sandbox", html)

    def test_checkpoint_restore_and_resource_analysis_are_reported(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_deep_report_") as tmp:
            root = Path(tmp)
            path = root / "telemetry.jsonl"
            records = [
                {
                    "timestamp": "2026-03-23T01:00:00+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-op-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                        "checkpoint_scope": "full",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:01+08:00",
                    "kind": "event",
                    "name": "checkpoint.process.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-proc-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:02+08:00",
                    "kind": "event",
                    "name": "checkpoint.process.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-proc-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:01.500000+08:00",
                    "kind": "event",
                    "name": "checkpoint.filesystem.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-fs-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:02.500000+08:00",
                    "kind": "event",
                    "name": "checkpoint.filesystem.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-fs-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:03+08:00",
                    "kind": "event",
                    "name": "checkpoint.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "ckpt-op-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                        "status": "succeeded",
                        "checkpoint_scope": "full",
                        "estimated_io_bytes": 1792,
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:00+08:00",
                    "kind": "event",
                    "name": "restore.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-op-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:00.100000+08:00",
                    "kind": "event",
                    "name": "restore.process.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-proc-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:00.400000+08:00",
                    "kind": "event",
                    "name": "restore.process.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-proc-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:00.200000+08:00",
                    "kind": "event",
                    "name": "restore.filesystem.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-fs-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:00.700000+08:00",
                    "kind": "event",
                    "name": "restore.filesystem.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-fs-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                        "status": "succeeded",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:01+08:00",
                    "kind": "event",
                    "name": "restore.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": "rst-op-1",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                        "status": "succeeded",
                        "mixed_sources": True,
                        "estimated_io_bytes": 1792,
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:03+08:00",
                    "kind": "metric",
                    "name": "checkpoint.process.size_bytes",
                    "value": 1024,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                        "checkpoint_scope": "full",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:03+08:00",
                    "kind": "metric",
                    "name": "checkpoint.filesystem.written_bytes",
                    "value": 768,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                        "checkpoint_scope": "full",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:03+08:00",
                    "kind": "metric",
                    "name": "checkpoint.estimated_io_bytes",
                    "value": 1792,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-ckpt-1",
                        "checkpoint_scope": "full",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:01+08:00",
                    "kind": "metric",
                    "name": "restore.source_gap.turns",
                    "value": 2,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                        "restore.mixed_sources": True,
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:01+08:00",
                    "kind": "metric",
                    "name": "restore.source_gap.ms",
                    "value": 1500,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                        "restore.mixed_sources": True,
                    },
                },
                {
                    "timestamp": "2026-03-23T01:01:01+08:00",
                    "kind": "metric",
                    "name": "restore.estimated_io_bytes",
                    "value": 1792,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-1",
                        "job_id": "job-rst-1",
                        "restore.mixed_sources": True,
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:20+08:00",
                    "kind": "metric",
                    "name": "llm.gate_wait_ms",
                    "value": 12.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "interceptor",
                        "request_id": "req-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:40+08:00",
                    "kind": "metric",
                    "name": "llm.gate_wait_ms",
                    "value": 20.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "interceptor",
                        "request_id": "req-2",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:25+08:00",
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
                    "timestamp": "2026-03-23T01:00:55+08:00",
                    "kind": "metric",
                    "name": "llm.agentcr_delay_ms",
                    "value": 50.0,
                    "attributes": {
                        "run_id": "run-a",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "component": "interceptor",
                        "request_id": "req-2",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:10+08:00",
                    "kind": "metric",
                    "name": "resource.host.cpu.usage_percent",
                    "value": 40,
                    "attributes": {"run_id": "run-a", "component": "monitoring"},
                },
                {
                    "timestamp": "2026-03-23T01:00:11+08:00",
                    "kind": "metric",
                    "name": "resource.host.memory.used_bytes",
                    "value": 1048576,
                    "attributes": {"run_id": "run-a", "component": "monitoring"},
                },
                {
                    "timestamp": "2026-03-23T01:00:12+08:00",
                    "kind": "metric",
                    "name": "resource.sandbox.cpu.usage_percent",
                    "value": 75,
                    "attributes": {
                        "run_id": "run-a",
                        "component": "monitoring",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                    },
                },
                {
                    "timestamp": "2026-03-23T01:00:13+08:00",
                    "kind": "metric",
                    "name": "resource.sandbox.memory.peak_bytes",
                    "value": 2097152,
                    "attributes": {
                        "run_id": "run-a",
                        "component": "monitoring",
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                    },
                },
            ]
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            analysis = generate_report_bundle(path, output_dir=root / "report", window_size_seconds=15.0)

            self.assertEqual(analysis.checkpoint_analysis.total_count, 1)
            self.assertEqual(analysis.checkpoint_analysis.success_count, 1)
            self.assertEqual(analysis.checkpoint_analysis.skip_count, 0)
            self.assertEqual(analysis.checkpoint_analysis.fail_count, 0)
            self.assertEqual(analysis.checkpoint_analysis.scope_counts["full"], 1)
            self.assertAlmostEqual(analysis.checkpoint_analysis.total_estimated_io_bytes, 1792.0)
            self.assertEqual(analysis.restore_analysis.total_count, 1)
            self.assertEqual(analysis.restore_analysis.success_count, 1)
            self.assertEqual(analysis.restore_analysis.skip_count, 0)
            self.assertEqual(analysis.restore_analysis.fail_count, 0)
            self.assertEqual(analysis.restore_analysis.mixed_source_count, 1)
            self.assertAlmostEqual(analysis.restore_analysis.mean_source_gap_turns, 2.0)
            self.assertIsNotNone(analysis.overhead_analysis)
            assert analysis.overhead_analysis is not None
            overhead_by_name = {item.metric_name: item for item in analysis.overhead_analysis.metrics}
            self.assertAlmostEqual(overhead_by_name["llm.gate_wait_ms"].mean_ms, 16.0)
            self.assertAlmostEqual(overhead_by_name["llm.gate_wait_ms"].p50_ms, 16.0)
            self.assertAlmostEqual(overhead_by_name["llm.agentcr_delay_ms"].mean_ms, 40.0)
            self.assertAlmostEqual(overhead_by_name["llm.agentcr_delay_ms"].p50_ms, 40.0)
            self.assertEqual(len(analysis.overhead_analysis.time_series["llm.gate_wait_ms"]), 2)
            self.assertEqual(len(analysis.overhead_analysis.time_series["llm.agentcr_delay_ms"]), 2)
            self.assertEqual(
                max(point.active_estimated_io_bytes for point in analysis.checkpoint_analysis.load_over_time),
                1792.0,
            )
            self.assertEqual(
                max(point.active_estimated_io_bytes for point in analysis.restore_analysis.load_over_time),
                1792.0,
            )
            self.assertEqual(len(analysis.checkpoint_analysis.latency_over_time["checkpoint.flow.duration_ms"]), 1)
            self.assertEqual(len(analysis.checkpoint_analysis.latency_over_time["checkpoint.process.duration_ms"]), 1)
            self.assertEqual(len(analysis.checkpoint_analysis.latency_over_time["checkpoint.filesystem.duration_ms"]), 1)
            self.assertEqual(len(analysis.restore_analysis.latency_over_time["restore.flow.duration_ms"]), 1)
            self.assertEqual(len(analysis.restore_analysis.latency_over_time["restore.process.duration_ms"]), 1)
            self.assertEqual(len(analysis.restore_analysis.latency_over_time["restore.filesystem.duration_ms"]), 1)
            self.assertTrue(analysis.resource_analysis.coverage["host_cpu"])
            self.assertTrue(analysis.resource_analysis.coverage["sandbox_memory"])

            html = (root / "report" / "report.html").read_text(encoding="utf-8")
            self.assertIn("Checkpoint Analysis", html)
            self.assertIn("Overhead Analysis", html)
            self.assertIn("Restore Analysis", html)
            self.assertLess(html.index("Checkpoint Analysis"), html.index("Overhead Analysis"))
            self.assertLess(html.index("Overhead Analysis"), html.index("Restore Analysis"))
            self.assertIn("15s windows", html)
            self.assertIn("Resource Usage", html)
            self.assertIn("Min (ms)", html)
            self.assertIn("Median (ms)", html)
            self.assertIn("P25 (ms)", html)
            self.assertIn("P50 (ms)", html)
            self.assertIn("llm.gate_wait", html)
            self.assertIn("llm.agentcr_delay", html)
            self.assertIn("Window-Aggregated Overhead Over Time", html)
            self.assertIn("Process Checkpoint Latency Over Time", html)
            self.assertIn("Filesystem Restore Latency Over Time", html)
            self.assertTrue((root / "report" / "checkpoint_load_jobs.svg").exists())
            self.assertTrue((root / "report" / "overhead_latency.svg").exists())
            self.assertTrue((root / "report" / "restore_load_jobs.svg").exists())
            self.assertTrue((root / "report" / "checkpoint_flow_latency.svg").exists())
            self.assertTrue((root / "report" / "checkpoint_process_latency.svg").exists())
            self.assertTrue((root / "report" / "checkpoint_filesystem_latency.svg").exists())
            self.assertTrue((root / "report" / "restore_flow_latency.svg").exists())
            self.assertTrue((root / "report" / "restore_process_latency.svg").exists())
            self.assertTrue((root / "report" / "restore_filesystem_latency.svg").exists())
            self.assertTrue((root / "report" / "resource_host_cpu.svg").exists())
            self.assertTrue((root / "report" / "resource_summary.csv").exists())
            self.assertTrue((root / "report" / "overhead_analysis.csv").exists())
            self.assertTrue((root / "report" / "overhead_timeseries.csv").exists())
            operation_csv = (root / "report" / "operation_summary.csv").read_text(encoding="utf-8")
            self.assertIn("p25_ms", operation_csv.splitlines()[0])
            checkpoint_load_svg = (root / "report" / "checkpoint_load_jobs.svg").read_text(encoding="utf-8")
            self.assertIn("legend-bg", checkpoint_load_svg)
            self.assertIn("active filesystem jobs", checkpoint_load_svg)
            overhead_svg = (root / "report" / "overhead_latency.svg").read_text(encoding="utf-8")
            self.assertIn("llm.gate_wait", overhead_svg)
            self.assertIn("llm.agentcr_delay", overhead_svg)
            restore_load_svg = (root / "report" / "restore_load_jobs.svg").read_text(encoding="utf-8")
            self.assertIn("active process jobs", restore_load_svg)


    def test_checkpoint_skip_and_fail_are_split(self) -> None:
        """Verify that succeeded, skipped, and failed checkpoints are counted separately."""
        with tempfile.TemporaryDirectory(prefix="agent_cr_telemetry_skip_") as tmp:
            root = Path(tmp)
            path = root / "telemetry.jsonl"
            base_ts = "2026-03-23T01:00:0"
            records = []
            # 3 checkpoint.flow operations: succeeded, skipped, failed
            for i, (op_id, status) in enumerate(
                [("op-ok", "succeeded"), ("op-skip", "skipped"), ("op-fail", "failed")]
            ):
                t_start = f"{base_ts}{i * 2}+08:00"
                t_finish = f"{base_ts}{i * 2 + 1}+08:00"
                records.append({
                    "timestamp": t_start,
                    "kind": "event",
                    "name": "checkpoint.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": op_id,
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": f"ckpt-{i}",
                        "job_id": f"job-{i}",
                        "checkpoint_scope": "full",
                    },
                })
                finish_attrs = {
                    "run_id": "run-a",
                    "op_id": op_id,
                    "sandbox_id": "sbx-1",
                    "task_id": "task-1",
                    "checkpoint_id": f"ckpt-{i}",
                    "job_id": f"job-{i}",
                    "status": status,
                    "checkpoint_scope": "full",
                    "estimated_io_bytes": 1024,
                }
                records.append({
                    "timestamp": t_finish,
                    "kind": "event",
                    "name": "checkpoint.flow.finish",
                    "attributes": finish_attrs,
                })
            # 3 restore.flow operations: succeeded, skipped, failed
            for i, (op_id, status) in enumerate(
                [("rst-ok", "succeeded"), ("rst-skip", "skipped"), ("rst-fail", "failed")]
            ):
                t_start = f"2026-03-23T01:01:0{i * 2}+08:00"
                t_finish = f"2026-03-23T01:01:0{i * 2 + 1}+08:00"
                records.append({
                    "timestamp": t_start,
                    "kind": "event",
                    "name": "restore.flow.start",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": op_id,
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-0",
                        "job_id": f"rst-job-{i}",
                    },
                })
                records.append({
                    "timestamp": t_finish,
                    "kind": "event",
                    "name": "restore.flow.finish",
                    "attributes": {
                        "run_id": "run-a",
                        "op_id": op_id,
                        "sandbox_id": "sbx-1",
                        "task_id": "task-1",
                        "checkpoint_id": "ckpt-0",
                        "job_id": f"rst-job-{i}",
                        "status": status,
                        "estimated_io_bytes": 512,
                    },
                })
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

            analysis = generate_report_bundle(path, output_dir=root / "report")

            ckpt = analysis.checkpoint_analysis
            self.assertIsNotNone(ckpt)
            self.assertEqual(ckpt.total_count, 3)
            self.assertEqual(ckpt.success_count, 1)
            self.assertEqual(ckpt.skip_count, 1)
            self.assertEqual(ckpt.fail_count, 1)
            self.assertAlmostEqual(ckpt.skip_rate, 1 / 3)
            self.assertAlmostEqual(ckpt.fail_rate, 1 / 3)
            # Per-task counts
            self.assertEqual(len(ckpt.per_task), 1)
            self.assertEqual(ckpt.per_task[0].success_count, 1)
            self.assertEqual(ckpt.per_task[0].skip_count, 1)
            self.assertEqual(ckpt.per_task[0].fail_count, 1)

            rst = analysis.restore_analysis
            self.assertIsNotNone(rst)
            self.assertEqual(rst.total_count, 3)
            self.assertEqual(rst.success_count, 1)
            self.assertEqual(rst.skip_count, 1)
            self.assertEqual(rst.fail_count, 1)
            self.assertAlmostEqual(rst.skip_rate, 1 / 3)
            self.assertAlmostEqual(rst.fail_rate, 1 / 3)
            # Per-task counts
            self.assertEqual(len(rst.per_task), 1)
            self.assertEqual(rst.per_task[0].success_count, 1)
            self.assertEqual(rst.per_task[0].skip_count, 1)
            self.assertEqual(rst.per_task[0].fail_count, 1)

            # HTML report includes "Skipped" labels
            html = (root / "report" / "report.html").read_text(encoding="utf-8")
            self.assertIn("Skipped", html)


if __name__ == "__main__":
    unittest.main()

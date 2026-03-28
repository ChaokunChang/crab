from __future__ import annotations

import unittest
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from agent_cr import CheckpointId, SandboxDescription, SandboxId
from agent_cr.models import CheckpointManifest
from benchmarks.real_host_scenario_base import RealHostScenarioHarness
from benchmarks.support import is_replay_llm_service_type
from integrations.agents import SandboxHandle
from integrations.llm_services.router import default_llm_service_type_for_agent


class MiniSWEBenchmarkHelperTests(unittest.TestCase):
    def test_replay_detection_includes_mini_swe_trace_replay(self) -> None:
        self.assertTrue(is_replay_llm_service_type("mini_swe_trace_replay"))

    def test_default_llm_service_type_for_mini_swe(self) -> None:
        self.assertEqual(default_llm_service_type_for_agent("mini_swe"), "mini_swe_trace_replay")

    def test_llm_service_checkpoint_metadata_prefers_task_status_cursor(self) -> None:
        sandbox_id = SandboxId("sbx-mini-meta")
        sandbox = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp/bundle"),
            status_port=8123,
            last_status={},
            llm_service_type="mini_swe_trace_replay",
        )
        sandbox.task_run = Mock()
        sandbox.task_run.poll_status.return_value = {"replay_trace_cursor": 7}

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness._sandbox_by_id = {sandbox_id: sandbox}
        harness._snapshot_llm_services = Mock()

        metadata = RealHostScenarioHarness._llm_service_checkpoint_metadata(harness, sandbox_id)

        self.assertEqual(metadata, {"benchmark_trace_cursor": 7})
        harness._snapshot_llm_services.assert_not_called()

    def test_restore_llm_service_state_records_restore_trace_cursor_for_task_run(self) -> None:
        sandbox_id = SandboxId("sbx-mini-restore-cursor")
        sandbox = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp/bundle"),
            status_port=8123,
            last_status={},
            llm_service_type="mini_swe_trace_replay",
        )
        sandbox.task_run = Mock()

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness._sandbox_by_id = {sandbox_id: sandbox}
        harness.llm_router_client = None
        harness.llm_server = None
        harness.storage = Mock()
        harness.storage.get_manifest.return_value = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-fs"),
            sandbox_id=sandbox_id,
            created_at=datetime.now(timezone.utc),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={"benchmark_trace_cursor": 7},
            integrity={},
        )
        manifest = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=CheckpointId("ckpt-current"),
            sandbox_id=sandbox_id,
            created_at=datetime.now(timezone.utc),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[],
            filesystem_artifacts=[],
            metadata={"filesystem_restore_checkpoint_id": "ckpt-fs"},
            integrity={},
        )

        RealHostScenarioHarness._restore_llm_service_state(harness, sandbox_id, manifest)

        sandbox.task_run.record_restore_trace_cursor.assert_called_once_with(7)

    def test_relaunch_sandbox_skips_llm_reset_when_preserving_mini_swe_task(self) -> None:
        sandbox_id = SandboxId("sbx-mini-relaunch")
        sandbox = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp/bundle"),
            status_port=8123,
            last_status={"state": "running"},
            llm_service_type="mini_swe_trace_replay",
            launch_source="compose",
        )
        sandbox.task_run = Mock()
        sandbox.task_run.survives_fault_relaunch.return_value = True
        sandbox.task_future = Future()

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.base_inspector = Mock()
        harness.runtime = Mock()
        harness.runtime.describe.return_value = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name="runc",
            status="running",
            metadata={"bundle_path": "/tmp/bundle"},
        )
        harness.runtime.launch.return_value = sandbox_id
        harness._sandbox_by_id = {sandbox_id: sandbox}
        harness._delete_runtime = Mock()
        harness._destroy_filesystem_dataset = Mock()
        harness._reset_llm_service_state = Mock()

        payload = RealHostScenarioHarness.relaunch_sandbox(harness, sandbox)

        self.assertEqual(payload, {"state": "running"})
        harness._reset_llm_service_state.assert_not_called()
        sandbox.task_run.request_stop.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from agent_cr import CheckpointId, JobStatus, SandboxId
from integrations.agents import SandboxHandle
from benchmarks.config import BenchmarkConfig
from benchmarks.real_host_scenario_base import RealHostScenarioHarness
from benchmarks.scenarios.common import choose_replay_chunk_targets, wait_for_auto_replay_checkpoint
from benchmarks.scenarios.fault import (
    parse_fault_options,
    run_auto as run_fault_auto,
    run_manual as run_fault_manual,
    run_replay_auto_sandbox,
    run_replay_manual_sandbox,
    summarize as summarize_fault,
)
from benchmarks.scenarios.spot import run_auto as run_spot_auto, run_manual as run_spot_manual
from benchmarks.support import BenchmarkTaskRecord


def _config(scenario: str, mode: str, **overrides: object) -> BenchmarkConfig:
    values: dict[str, object] = {
        "config_path": Path("/tmp/bench.yaml"),
        "scenario": scenario,
        "mode": mode,
        "provider": "openai",
        "agent": "simulated",
        "llm_service": "simulated",
        "task_dataset": None,
        "sandboxes": 2,
        "iterations": 2,
        "output": None,
        "log_level": "info",
        "transfer_delay_ms": 0.0,
        "work_dir_host_root": None,
        "scenario_options": {},
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


class _FakeStorage:
    def list_checkpoints(self, sandbox_id) -> list[str]:
        return [f"ckpt-{sandbox_id}"]


class _FakeTaskRun:
    def __init__(self, harness: "_BaseScenarioHarness", sandbox: SandboxHandle) -> None:
        self._harness = harness
        self._sandbox = sandbox

    def wait_for_progress(self, *, minimum_actions: int) -> dict[str, object]:
        sandbox_id = str(self._sandbox.sandbox_id)
        self._harness.wait_for_progress_calls.append(sandbox_id)
        if self._harness.progress_delay_s > 0:
            with self._harness._progress_lock:
                self._harness._active_progress += 1
                self._harness.max_concurrent_progress = max(
                    self._harness.max_concurrent_progress,
                    self._harness._active_progress,
                )
            try:
                time.sleep(self._harness.progress_delay_s)
            finally:
                with self._harness._progress_lock:
                    self._harness._active_progress -= 1
        payload = {"total_actions": max(int(self._harness._statuses[sandbox_id]["total_actions"]), minimum_actions)}
        self._harness._statuses[sandbox_id] = payload
        self._harness._progress_actions[sandbox_id] = int(payload["total_actions"])
        self._sandbox.last_status = payload
        return payload

    def wait_for_action_delta(self, *, delta: int) -> dict[str, object]:
        sandbox_id = str(self._sandbox.sandbox_id)
        self._harness.wait_for_action_delta_calls.append((sandbox_id, delta))
        payload = {"total_actions": int(self._sandbox.last_status["total_actions"]) + delta}
        self._harness._statuses[sandbox_id] = payload
        self._sandbox.last_status = payload
        return payload

    def poll_status(self) -> dict[str, object]:
        sandbox_id = str(self._sandbox.sandbox_id)
        if sandbox_id in self._harness._unreachable_sandboxes:
            raise AssertionError(f"poll_status should not be called for unreachable sandbox {sandbox_id}")
        return dict(self._harness._statuses[sandbox_id])


class _BaseScenarioHarness:
    def __init__(self) -> None:
        self.storage = _FakeStorage()
        self._next_port = 20000
        self._statuses: dict[str, dict[str, object]] = {}
        self._checkpoint_counts: dict[str, int] = {}
        self._progress_actions: dict[str, int] = {}
        self.launched: list[str] = []
        self.wait_for_progress_calls: list[str] = []
        self.wait_for_action_delta_calls: list[tuple[str, int]] = []
        self.checkpoint_if_due_calls: list[str] = []
        self.inject_fault_calls: list[str] = []
        self.notify_fault_calls: list[str] = []
        self.notify_preemption_calls: list[str] = []
        self.set_snapshot_metadata_calls: list[tuple[str, dict[str, object]]] = []
        self.clear_snapshot_metadata_calls: list[tuple[str, tuple[str, ...]]] = []
        self.progress_delay_s = 0.0
        self._progress_lock = threading.Lock()
        self._active_progress = 0
        self.max_concurrent_progress = 0
        self.launch_delay_s = 0.0
        self._launch_lock = threading.Lock()
        self._active_launches = 0
        self.max_concurrent_launches = 0
        self.fail_on_restore_sandbox: str | None = None
        self.fail_recovery_record_sandbox: str | None = None
        self._unreachable_sandboxes: set[str] = set()
        self.wait_for_task_completion_calls: list[str] = []
        self.verify_task_accuracy_calls: list[str] = []

    def _record_for_config(self, config: BenchmarkConfig, sandbox_index: int) -> BenchmarkTaskRecord:
        return BenchmarkTaskRecord(
            agent_type=config.agent,
            task_description=SimpleNamespace(prompt=""),
            task_config=SimpleNamespace(options={}),
            llm_service_type=config.llm_service,
            task_id=f"task-{sandbox_index}",
            trace_response_count=10 if config.llm_service == "iflow_trace_replay" else None,
        )

    def launch_task_record(self, sandbox_name: str, record: BenchmarkTaskRecord) -> SandboxHandle:
        if self.launch_delay_s > 0:
            with self._launch_lock:
                self._active_launches += 1
                self.max_concurrent_launches = max(self.max_concurrent_launches, self._active_launches)
            try:
                time.sleep(self.launch_delay_s)
            finally:
                with self._launch_lock:
                    self._active_launches -= 1
        sandbox_id = SandboxId(sandbox_name)
        handle = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp") / sandbox_name,
            status_port=self._next_port,
            last_status={"total_actions": 0},
            agent_type=record.agent_type,
            llm_service_type=record.llm_service_type,
            launch_metadata={"benchmark": {"task_id": record.task_id, "trace_response_count": record.trace_response_count}},
        )
        self._next_port += 1
        self.launched.append(sandbox_name)
        self._statuses[str(sandbox_id)] = {"total_actions": 0}
        self._checkpoint_counts[str(sandbox_id)] = 0
        handle.task_run = _FakeTaskRun(self, handle)
        return handle

    def checkpoint_if_due(self, sandbox: SandboxHandle):
        sandbox_id = str(sandbox.sandbox_id)
        self.checkpoint_if_due_calls.append(sandbox_id)
        self._checkpoint_counts[sandbox_id] = self._checkpoint_counts.get(sandbox_id, 0) + 1
        return SimpleNamespace(
            checkpoint_id=CheckpointId(f"{sandbox_id}-ckpt-{self._checkpoint_counts[sandbox_id]}"),
            status=JobStatus.SUCCEEDED,
            message="",
        )

    def checkpoint_manual(self, sandbox: SandboxHandle, leave_running: bool = True):
        _ = leave_running
        sandbox_id = str(sandbox.sandbox_id)
        self._checkpoint_counts[sandbox_id] = self._checkpoint_counts.get(sandbox_id, 0) + 1
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            checkpoint_id=CheckpointId(f"{sandbox_id}-manual-{self._checkpoint_counts[sandbox_id]}"),
            status=JobStatus.SUCCEEDED,
            message="",
            started_at=now,
            finished_at=now + timedelta(milliseconds=1),
        )

    def inject_fault(self, sandbox: SandboxHandle) -> None:
        self.inject_fault_calls.append(str(sandbox.sandbox_id))

    def restore_once(self, sandbox: SandboxHandle, checkpoint_id: CheckpointId):
        sandbox_id = str(sandbox.sandbox_id)
        if sandbox_id == self.fail_on_restore_sandbox:
            raise RuntimeError(f"restore failed for {sandbox_id}")
        restored_actions = self._progress_actions.get(sandbox_id, 6)
        self._statuses[sandbox_id] = {"total_actions": restored_actions}
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            started_at=now,
            finished_at=now + timedelta(milliseconds=1),
            checkpoint_id=checkpoint_id,
            status=SimpleNamespace(value="succeeded"),
            message=None,
        )

    def wait_for_recovery(self, sandbox: SandboxHandle, *, event_type: str, observed_after, timeout_s: float = 60.0):
        _ = observed_after
        _ = timeout_s
        sandbox_id = str(sandbox.sandbox_id)
        if sandbox_id == self.fail_on_restore_sandbox:
            raise RuntimeError(f"recovery wait failed for {sandbox_id}")
        if sandbox_id == self.fail_recovery_record_sandbox:
            self._unreachable_sandboxes.add(sandbox_id)
            return SimpleNamespace(status="failed")
        if event_type == "fault":
            restored_actions = self._progress_actions.get(sandbox_id, 6)
            self._statuses[sandbox_id] = {"total_actions": restored_actions}
        return SimpleNamespace(status="restored")

    def notify_fault(self, sandbox: SandboxHandle, *, reason: str = "fault") -> None:
        _ = reason
        self.notify_fault_calls.append(str(sandbox.sandbox_id))

    def notify_preemption(self, sandbox: SandboxHandle, *, grace_remaining_seconds: float) -> None:
        _ = grace_remaining_seconds
        sandbox_id = str(sandbox.sandbox_id)
        if sandbox_id == self.fail_on_restore_sandbox:
            raise RuntimeError(f"preemption notification failed for {sandbox_id}")
        self.notify_preemption_calls.append(sandbox_id)

    def set_snapshot_metadata(self, sandbox: SandboxHandle, **metadata: object) -> None:
        self.set_snapshot_metadata_calls.append((str(sandbox.sandbox_id), metadata))

    def clear_snapshot_metadata(self, sandbox: SandboxHandle, *keys: str) -> None:
        self.clear_snapshot_metadata_calls.append((str(sandbox.sandbox_id), keys))

    def wait_for_task_completion(self, sandbox: SandboxHandle, timeout_s: float | None = None) -> None:
        _ = timeout_s
        sandbox_id = str(sandbox.sandbox_id)
        self.wait_for_task_completion_calls.append(sandbox_id)
        trace_response_count = sandbox.launch_metadata.get("benchmark", {}).get("trace_response_count")
        if trace_response_count is None:
            return
        final_actions = int(trace_response_count)
        payload = {
            "state": "finished",
            "total_actions": final_actions,
            "replay_next_response_index": final_actions,
            "replay_is_complete": True,
        }
        self._statuses[sandbox_id] = payload
        sandbox.last_status = dict(payload)

    def verify_task_accuracy(self, sandbox: SandboxHandle, timeout_s: float | None = None) -> dict[str, object]:
        _ = (sandbox, timeout_s)
        self.verify_task_accuracy_calls.append(str(sandbox.sandbox_id))
        return {
            "verification_status": "passed",
            "verification_exit_code": 0,
            "verification_ms": 0.0,
            "verification_stdout": "ok\n",
            "verification_stderr": "",
            "verification_command": "bash /tests/run-tests.sh",
        }


class FaultScenarioTests(unittest.TestCase):
    def test_choose_replay_chunk_targets_uses_chunk_endpoints(self) -> None:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("fault-0"),
            bundle_dir=Path("/tmp/fault-0"),
            status_port=20000,
            last_status={"total_actions": 0},
            agent_type="iflow",
            llm_service_type="iflow_trace_replay",
            launch_metadata={"benchmark": {"task_id": "cross-entropy-method", "trace_response_count": 24}},
        )

        targets = choose_replay_chunk_targets(sandbox, 5)

        self.assertEqual(targets, [5, 10, 15, 20, 24])

    def test_manual_mode_runs_sandboxes_independently_and_sorts_rows(self) -> None:
        harness = _BaseScenarioHarness()
        harness.progress_delay_s = 0.05

        rows = run_fault_manual(_config("fault", "manual"), harness)

        self.assertEqual(harness.launched, ["fault-0", "fault-1"])
        self.assertGreaterEqual(harness.max_concurrent_progress, 2)
        self.assertEqual(
            [(int(row["iteration"]), row["sandbox_id"]) for row in rows],
            [(1, "fault-0"), (1, "fault-1"), (2, "fault-0"), (2, "fault-1")],
        )
        self.assertEqual(sorted(harness.inject_fault_calls), ["fault-0", "fault-0", "fault-1", "fault-1"])

    def test_auto_mode_uses_deterministic_per_sandbox_fault_selection(self) -> None:
        harness = _BaseScenarioHarness()

        rows = run_fault_auto(
            _config(
                "fault",
                "auto",
                iterations=3,
                scenario_options={"injection_rate": 0.0, "first_injection_iteration": 2},
            ),
            harness,
        )

        injected_rows = [(int(row["iteration"]), row["sandbox_id"]) for row in rows if int(row["event_injected"]) == 1]
        self.assertEqual(injected_rows, [(2, "fault-0"), (2, "fault-1")])
        self.assertEqual(harness.notify_fault_calls, ["fault-0", "fault-1"])

    def test_auto_mode_can_limit_launch_parallelism_below_sandbox_count(self) -> None:
        harness = _BaseScenarioHarness()
        harness.launch_delay_s = 0.05

        rows = run_fault_auto(
            _config(
                "fault",
                "auto",
                sandboxes=4,
                max_workers=2,
                iterations=1,
                scenario_options={"injection_rate": 0.0},
            ),
            harness,
        )

        self.assertEqual(len(rows), 4)
        self.assertEqual(harness.launched, ["fault-0", "fault-1", "fault-2", "fault-3"])
        self.assertLessEqual(harness.max_concurrent_launches, 2)

    def test_replay_mode_treats_completion_before_later_checkpoint_as_success(self) -> None:
        class _ReplayEarlyFinishTaskRun:
            def __init__(self, sandbox: SandboxHandle) -> None:
                self._sandbox = sandbox

            def wait_for_progress(self, *, minimum_actions: int) -> dict[str, object]:
                if minimum_actions <= 42:
                    payload = {
                        "state": "running",
                        "total_actions": 42,
                        "replay_next_response_index": 42,
                        "replay_is_complete": False,
                    }
                    self._sandbox.last_status = dict(payload)
                    return payload
                self._sandbox.last_status = {
                    "state": "finished",
                    "total_actions": 30,
                    "replay_next_response_index": 30,
                    "replay_is_complete": False,
                }
                raise RuntimeError(
                    "iflow replay task finished before reaching replay action count "
                    f"{minimum_actions}; last observed count was 30"
                )

            def wait_for_action_delta(self, *, delta: int) -> dict[str, object]:
                payload = {
                    "state": "running",
                    "total_actions": int(self._sandbox.last_status.get("total_actions", 0)) + delta,
                    "replay_next_response_index": int(self._sandbox.last_status.get("replay_next_response_index", 0)) + delta,
                    "replay_is_complete": False,
                }
                self._sandbox.last_status = dict(payload)
                return payload

            def poll_status(self) -> dict[str, object]:
                return dict(self._sandbox.last_status)

        harness = _BaseScenarioHarness()
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("fault-0"),
            bundle_dir=Path("/tmp/fault-0"),
            status_port=20000,
            last_status={"total_actions": 0},
            agent_type="iflow",
            llm_service_type="iflow_trace_replay",
            launch_metadata={"benchmark": {"task_id": "catch-me-if-you-can", "trace_response_count": 84}},
        )
        sandbox.task_run = _ReplayEarlyFinishTaskRun(sandbox)
        config = _config(
            "fault",
            "manual",
            sandboxes=1,
            iterations=2,
            agent="iflow",
            llm_service="iflow_trace_replay",
            scenario_options={"injection_rate": 1.0, "first_injection_iteration": 1},
        )

        row = run_replay_manual_sandbox(
            config,
            harness,
            sandbox_index=0,
            sandbox=sandbox,
            options=parse_fault_options(config),
        )

        self.assertEqual(row["task_error"], "")
        self.assertEqual(row["events_injected"], 1)
        self.assertEqual(row["recoveries_succeeded"], 1)
        self.assertEqual(row["success_ratio"], 1.0)

    def test_wait_for_auto_replay_checkpoint_returns_latest_checkpoint(self) -> None:
        class _Harness:
            def __init__(self) -> None:
                self.calls = 0

            def list_checkpoint_manifests(self, sandbox_id):
                self.calls += 1
                if self.calls < 2:
                    return []
                return [
                    SimpleNamespace(metadata={"benchmark_replay_action_count": 4}),
                ]

        sandbox = SandboxHandle(
            sandbox_id=SandboxId("fault-0"),
            bundle_dir=Path("/tmp/fault-0"),
            status_port=20000,
            last_status={"total_actions": 4, "replay_next_response_index": 4},
        )
        sandbox.task_run = SimpleNamespace(poll_status=lambda: dict(sandbox.last_status))

        manifest, checkpoint_actions = wait_for_auto_replay_checkpoint(
            _Harness(),
            sandbox,
            minimum_actions=4,
            trace_response_count=10,
        )

        self.assertIsNotNone(manifest)
        self.assertEqual(checkpoint_actions, 4)

    def test_list_checkpoint_manifests_skips_manifest_deleted_during_enumeration(self) -> None:
        class _Storage:
            def list_checkpoints(self, sandbox_id):
                _ = sandbox_id
                return [CheckpointId("ckpt-1"), CheckpointId("ckpt-2")]

            def get_manifest(self, sandbox_id, checkpoint_id):
                _ = sandbox_id
                if checkpoint_id == CheckpointId("ckpt-1"):
                    return SimpleNamespace(checkpoint_id=checkpoint_id, metadata={"benchmark_replay_action_count": 1})
                raise FileNotFoundError("manifest pruned")

        harness = RealHostScenarioHarness.__new__(RealHostScenarioHarness)
        harness.storage = _Storage()

        manifests = harness.list_checkpoint_manifests(SandboxId("fault-0"))

        self.assertEqual([manifest.checkpoint_id for manifest in manifests], [CheckpointId("ckpt-1")])

    def test_replay_auto_mode_waits_for_completion_and_runs_final_verification(self) -> None:
        harness = _BaseScenarioHarness()
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("fault-0"),
            bundle_dir=Path("/tmp/fault-0"),
            status_port=20000,
            last_status={"total_actions": 0},
            agent_type="iflow",
            llm_service_type="iflow_trace_replay",
            launch_metadata={"benchmark": {"task_id": "amuse-install", "trace_response_count": 10}},
        )
        sandbox.task_run = _FakeTaskRun(harness, sandbox)
        harness._statuses[str(sandbox.sandbox_id)] = {"total_actions": 0}
        harness._progress_actions[str(sandbox.sandbox_id)] = 0
        config = _config(
            "fault",
            "auto",
            sandboxes=1,
            iterations=2,
            agent="iflow",
            llm_service="iflow_trace_replay",
            scenario_options={"injection_rate": 0.0, "first_injection_iteration": 2},
        )

        with mock.patch(
            "benchmarks.scenarios.fault.wait_for_auto_replay_checkpoint",
            return_value=(SimpleNamespace(metadata={"benchmark_replay_action_count": 2}), 2),
        ):
            row = run_replay_auto_sandbox(
                config,
                harness,
                sandbox_index=0,
                sandbox=sandbox,
                options=parse_fault_options(config),
            )

        self.assertEqual(row["task_error"], "")
        self.assertEqual(row["success_ratio"], 1.0)
        self.assertEqual(row["verification_status"], "passed")
        self.assertEqual(row["verification_stdout"], "ok\n")
        self.assertEqual(row["replay_final_index"], 10)
        self.assertTrue(bool(row["replay_is_complete"]))
        self.assertEqual(harness.wait_for_task_completion_calls, ["fault-0"])
        self.assertEqual(harness.verify_task_accuracy_calls, ["fault-0"])

    def test_replay_auto_mode_uses_chunk_checkpoint_for_forced_fault(self) -> None:
        class _ReplayChunkTaskRun:
            def __init__(self, sandbox: SandboxHandle) -> None:
                self._sandbox = sandbox

            def wait_for_progress(self, *, minimum_actions: int) -> dict[str, object]:
                if minimum_actions <= 5:
                    payload = {
                        "state": "running",
                        "total_actions": 5,
                        "replay_next_response_index": 5,
                        "replay_is_complete": False,
                    }
                elif minimum_actions <= 10:
                    payload = {
                        "state": "running",
                        "total_actions": 24,
                        "replay_next_response_index": 24,
                        "replay_is_complete": True,
                    }
                else:
                    payload = {
                        "state": "finished",
                        "total_actions": 24,
                        "replay_next_response_index": 24,
                        "replay_is_complete": True,
                    }
                self._sandbox.last_status = dict(payload)
                return payload

            def poll_status(self) -> dict[str, object]:
                return dict(self._sandbox.last_status)

        harness = _BaseScenarioHarness()
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("fault-0"),
            bundle_dir=Path("/tmp/fault-0"),
            status_port=20000,
            last_status={"total_actions": 0},
            agent_type="iflow",
            llm_service_type="iflow_trace_replay",
            launch_metadata={"benchmark": {"task_id": "cross-entropy-method", "trace_response_count": 24}},
        )
        sandbox.task_run = _ReplayChunkTaskRun(sandbox)
        harness._statuses[str(sandbox.sandbox_id)] = {"total_actions": 0}
        config = _config(
            "fault",
            "auto",
            sandboxes=1,
            iterations=5,
            agent="iflow",
            llm_service="iflow_trace_replay",
            scenario_options={"injection_rate": 0.0, "first_forced_event_chunk": 2},
        )

        with mock.patch(
            "benchmarks.scenarios.fault.wait_for_auto_replay_checkpoint",
            return_value=(SimpleNamespace(metadata={"benchmark_replay_action_count": 10}), 10),
        ):
            row = run_replay_auto_sandbox(
                config,
                harness,
                sandbox_index=0,
                sandbox=sandbox,
                options=parse_fault_options(config),
            )

        self.assertEqual(row["task_error"], "")
        self.assertEqual(row["events_injected"], 1)
        self.assertEqual(row["recoveries_succeeded"], 1)
        self.assertEqual(row["chunks_planned"], 5)
        self.assertEqual(row["chunks_completed"], 3)
        self.assertEqual(harness.inject_fault_calls, ["fault-0"])
        self.assertEqual(harness.notify_fault_calls, ["fault-0"])

    def test_replay_auto_summary_uses_aggregate_metrics(self) -> None:
        rows = [
            {
                "iterations_planned": 5,
                "checkpoint_ms_avg": 0.0,
                "restore_ms_avg": 0.0,
                "recovery_ms_avg": 12.0,
                "readiness_ms_avg": 3.0,
                "end_to_end_recovery_ms_avg": 15.0,
                "lost_actions_avg": 0.5,
                "success_ratio": 1.0,
            }
        ]

        summary = summarize_fault(_config("fault", "auto", agent="iflow", llm_service="iflow_trace_replay"), rows)

        self.assertEqual(summary["recovery_ms"], 12.0)
        self.assertEqual(summary["success_ratio"], 1.0)


class SpotScenarioTests(unittest.TestCase):
    def test_manual_mode_sets_and_clears_preemption_metadata(self) -> None:
        harness = _BaseScenarioHarness()

        rows = run_spot_manual(
            _config(
                "spot",
                "manual",
                sandboxes=1,
                scenario_options={"grace_period_seconds": 30.0},
            ),
            harness,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(len(harness.set_snapshot_metadata_calls), 2)
        self.assertEqual(len(harness.clear_snapshot_metadata_calls), 2)
        self.assertEqual([row["event_type"] for row in rows], ["preemption", "preemption"])

    def test_auto_mode_uses_deterministic_preemption_selection(self) -> None:
        harness = _BaseScenarioHarness()

        rows = run_spot_auto(
            _config(
                "spot",
                "auto",
                iterations=3,
                scenario_options={
                    "injection_rate": 0.0,
                    "first_injection_iteration": 2,
                    "grace_period_seconds": 15.0,
                },
            ),
            harness,
        )

        injected_rows = [(int(row["iteration"]), row["sandbox_id"]) for row in rows if int(row["event_injected"]) == 1]
        self.assertEqual(injected_rows, [(2, "spot-0"), (2, "spot-1")])
        self.assertEqual(harness.notify_preemption_calls, ["spot-0", "spot-1"])


if __name__ == "__main__":
    unittest.main()

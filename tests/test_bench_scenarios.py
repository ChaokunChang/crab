from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

from agent_cr import CheckpointId, SandboxId
from benchmarks.real_host_scenario_base import BenchmarkTaskRecord
from benchmarks.bench_fault_tolerance import run_fault_tolerance_benchmark
from benchmarks.bench_spot_agent import run_spot_agent_benchmark
from benchmarks.real_host_scenario_base import SandboxHandle


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
        self.fail_on_restore_sandbox: str | None = None
        self.fail_recovery_record_sandbox: str | None = None
        self._unreachable_sandboxes: set[str] = set()

    def launch_sandbox(self, sandbox_name: str) -> SandboxHandle:
        sandbox_id = SandboxId(sandbox_name)
        handle = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp") / sandbox_name,
            status_port=self._next_port,
            last_status={"total_actions": 0},
        )
        self._next_port += 1
        self.launched.append(sandbox_name)
        self._statuses[str(sandbox_id)] = {"total_actions": 0}
        self._checkpoint_counts[str(sandbox_id)] = 0
        handle.task_run = _FakeTaskRun(self, handle)
        return handle

    def load_dataset(self, path):
        raise AssertionError(f"dataset should not be loaded in this test: {path}")

    def select_task_record(
        self,
        dataset,
        *,
        sandbox_index: int,
        default_agent_type: str,
        default_task_description,
        default_task_config,
    ) -> BenchmarkTaskRecord:
        _ = dataset
        _ = sandbox_index
        return BenchmarkTaskRecord(
            agent_type=default_agent_type,
            task_description=default_task_description,
            task_config=default_task_config,
        )

    def launch_sandbox_and_task(self, sandbox_name: str, *, agent_type, task_description, task_config) -> SandboxHandle:
        _ = (agent_type, task_description, task_config)
        return self.launch_sandbox(sandbox_name)

    def launch_task_record(self, sandbox_name: str, record: BenchmarkTaskRecord) -> SandboxHandle:
        _ = record
        return self.launch_sandbox(sandbox_name)

    def checkpoint_if_due(self, sandbox: SandboxHandle):
        sandbox_id = str(sandbox.sandbox_id)
        self.checkpoint_if_due_calls.append(sandbox_id)
        self._checkpoint_counts[sandbox_id] += 1
        return SimpleNamespace(checkpoint_id=CheckpointId(f"{sandbox_id}-ckpt-{self._checkpoint_counts[sandbox_id]}"))

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


class FaultToleranceBenchmarkTests(unittest.TestCase):
    def _args(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "sandboxes": 2,
            "iters": 2,
            "auto_cr": False,
            "fault_rate": 0.5,
            "first_fault_iteration": 0,
            "agent_type": "simulated",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_manual_mode_runs_sandboxes_independently_and_sorts_rows(self) -> None:
        harness = _BaseScenarioHarness()
        harness.progress_delay_s = 0.05

        rows = run_fault_tolerance_benchmark(self._args(), harness)

        self.assertEqual(harness.launched, ["fault-0", "fault-1"])
        self.assertGreaterEqual(harness.max_concurrent_progress, 2)
        self.assertEqual(
            [(int(row["iter"]), row["sandbox_id"]) for row in rows],
            [(1, "fault-0"), (1, "fault-1"), (2, "fault-0"), (2, "fault-1")],
        )
        self.assertEqual(sorted(harness.inject_fault_calls), ["fault-0", "fault-0", "fault-1", "fault-1"])

    def test_auto_mode_uses_deterministic_per_sandbox_fault_selection(self) -> None:
        harness = _BaseScenarioHarness()

        rows = run_fault_tolerance_benchmark(
            self._args(auto_cr=True, iters=3, fault_rate=0.0, first_fault_iteration=2),
            harness,
        )

        injected_rows = [(int(row["iter"]), row["sandbox_id"]) for row in rows if int(row["event_injected"]) == 1]
        self.assertEqual(injected_rows, [(2, "fault-0")])
        self.assertEqual(harness.notify_fault_calls, ["fault-0"])

    def test_manual_mode_propagates_worker_failures(self) -> None:
        harness = _BaseScenarioHarness()
        harness.fail_on_restore_sandbox = "fault-1"

        with self.assertRaisesRegex(RuntimeError, "restore failed for fault-1"):
            run_fault_tolerance_benchmark(self._args(iters=1), harness)


class SpotAgentBenchmarkTests(unittest.TestCase):
    def _args(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "sandboxes": 2,
            "iters": 2,
            "grace_period_seconds": 60.0,
            "auto_cr": False,
            "preemption_rate": 0.5,
            "first_preempt_iteration": 0,
            "agent_type": "simulated",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_manual_mode_runs_sandboxes_independently_and_sorts_rows(self) -> None:
        harness = _BaseScenarioHarness()
        harness.progress_delay_s = 0.05

        rows = run_spot_agent_benchmark(self._args(), harness)

        self.assertEqual(harness.launched, ["spot-0", "spot-1"])
        self.assertGreaterEqual(harness.max_concurrent_progress, 2)
        self.assertEqual(
            [(int(row["iter"]), row["sandbox_id"]) for row in rows],
            [(1, "spot-0"), (1, "spot-1"), (2, "spot-0"), (2, "spot-1")],
        )
        self.assertEqual(len(harness.set_snapshot_metadata_calls), 4)
        self.assertEqual(len(harness.clear_snapshot_metadata_calls), 4)

    def test_auto_mode_uses_deterministic_per_sandbox_preemption_selection(self) -> None:
        harness = _BaseScenarioHarness()

        rows = run_spot_agent_benchmark(
            self._args(auto_cr=True, iters=3, preemption_rate=0.0, first_preempt_iteration=2),
            harness,
        )

        injected_rows = [(int(row["iter"]), row["sandbox_id"]) for row in rows if int(row["event_injected"]) == 1]
        self.assertEqual(injected_rows, [(2, "spot-0")])
        self.assertEqual(harness.notify_preemption_calls, ["spot-0"])
        self.assertEqual(harness.wait_for_progress_calls, ["spot-0", "spot-1"])
        self.assertEqual(
            harness.wait_for_action_delta_calls,
            [("spot-0", 1), ("spot-0", 1), ("spot-1", 1), ("spot-1", 1)],
        )

    def test_auto_mode_stops_after_failed_recovery_record_without_polling_status(self) -> None:
        harness = _BaseScenarioHarness()
        harness.fail_recovery_record_sandbox = "spot-0"

        rows = run_spot_agent_benchmark(
            self._args(sandboxes=1, auto_cr=True, iters=3, preemption_rate=0.0, first_preempt_iteration=1),
            harness,
        )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["recovery_status"], "failed")
        self.assertEqual(rows[0]["event_injected"], 1)

    def test_manual_mode_propagates_worker_failures(self) -> None:
        harness = _BaseScenarioHarness()
        harness.fail_on_restore_sandbox = "spot-1"

        with self.assertRaisesRegex(RuntimeError, "restore failed for spot-1"):
            run_spot_agent_benchmark(self._args(iters=1), harness)


if __name__ == "__main__":
    unittest.main()

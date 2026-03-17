from __future__ import annotations

import argparse
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

from agent_cr import CheckpointId, SandboxId
from integrations.agents import SandboxHandle
from benchmarks.bench_tree_search import run_tree_search_benchmark
from benchmarks.support import BenchmarkTaskRecord, TreeSearchCheckpointRecord


class _FakeStorage:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_all_checkpoints(self, sandbox_id) -> None:
        self.deleted.append(str(sandbox_id))


class _FakeSystem:
    def __init__(self) -> None:
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self) -> None:
        self.stop_calls += 1


class _FakeTaskRun:
    def __init__(self, harness: "_FakeHarness", sandbox: SandboxHandle) -> None:
        self._harness = harness
        self._sandbox = sandbox

    def wait_for_action_delta(self, *, delta: int) -> dict[str, object]:
        sandbox_id = str(self._sandbox.sandbox_id)
        if sandbox_id.startswith("tree-source-") and self._harness.source_action_delay_s > 0:
            with self._harness._source_action_lock:
                self._harness._active_source_actions += 1
                self._harness.max_concurrent_source_actions = max(
                    self._harness.max_concurrent_source_actions,
                    self._harness._active_source_actions,
                )
            try:
                time.sleep(self._harness.source_action_delay_s)
            finally:
                with self._harness._source_action_lock:
                    self._harness._active_source_actions -= 1
        self._harness.action_delta_calls.append((sandbox_id, delta))
        current = int(self._harness._statuses[sandbox_id]["total_actions"])
        payload = {"total_actions": current + delta}
        self._harness._statuses[sandbox_id] = payload
        self._sandbox.last_status = payload
        completion_target = self._harness._task_completion_targets.get(sandbox_id)
        task_future = self._sandbox.task_future
        if (
            completion_target is not None
            and task_future is not None
            and not task_future.done()
            and int(payload["total_actions"]) >= completion_target
        ):
            task_future.set_result(None)
        return payload

    def poll_status(self) -> dict[str, object]:
        return dict(self._harness._statuses[str(self._sandbox.sandbox_id)])

    def request_stop(self) -> None:
        sandbox_id = str(self._sandbox.sandbox_id)
        current = int(self._harness._statuses[sandbox_id]["total_actions"])
        self._harness.stop_requests.append(sandbox_id)
        self._harness.events.append(("request_stop", sandbox_id, current))


class _FakeHarness:
    def __init__(self) -> None:
        self.storage = _FakeStorage()
        self.system = _FakeSystem()
        self._next_port = 19000
        self._handles: dict[str, SandboxHandle] = {}
        self._statuses: dict[str, dict[str, object]] = {}
        self._snapshot_steps: dict[str, list[int]] = {}
        self._checkpoint_steps: dict[str, int] = {}
        self._checkpoint_to_step: dict[str, int] = {}
        self._task_completion_targets: dict[str, int | None] = {}
        self._source_launches: list[str] = []
        self.task_launches: list[str] = []
        self.stop_requests: list[str] = []
        self.action_delta_calls: list[tuple[str, int]] = []
        self.manual_checkpoint_calls: list[tuple[str, bool]] = []
        self.wait_for_tree_search_calls: list[tuple[str, int]] = []
        self.deactivated: list[str] = []
        self.destroyed: list[str] = []
        self.events: list[tuple[str, str, int]] = []
        self.restore_delay_s = 0.0
        self._restore_lock = threading.Lock()
        self._active_restores = 0
        self.max_concurrent_restores = 0
        self.source_action_delay_s = 0.0
        self._source_action_lock = threading.Lock()
        self._active_source_actions = 0
        self.max_concurrent_source_actions = 0
        self.source_task_completion_delta: int | None = None
        self.fork_task_completion_delta: int | None = None

    def _completion_delta_for(self, sandbox_id: str) -> int | None:
        if sandbox_id.startswith("tree-fork-"):
            return self.fork_task_completion_delta
        return self.source_task_completion_delta

    def _install_task(self, handle: SandboxHandle) -> None:
        sandbox_id = str(handle.sandbox_id)
        baseline = int(self._statuses[sandbox_id]["total_actions"])
        completion_delta = self._completion_delta_for(sandbox_id)
        self._task_completion_targets[sandbox_id] = (
            None if completion_delta is None else baseline + completion_delta
        )
        handle.task_run = _FakeTaskRun(self, handle)
        handle.task_future = Future()

    def launch_sandbox_and_task(
        self,
        sandbox_name: str,
        *,
        agent_type,
        llm_service_type=None,
        task_description,
        task_config,
    ) -> SandboxHandle:
        _ = (agent_type, llm_service_type, task_description, task_config)
        sandbox_id = SandboxId(sandbox_name)
        handle = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp") / sandbox_name,
            status_port=self._next_port,
            last_status={"total_actions": 0},
            status_host="10.250.0.2",
            agent_type=agent_type,
            task_description=task_description,
            task_config=task_config,
        )
        self._next_port += 1
        self._source_launches.append(sandbox_name)
        self._handles[sandbox_name] = handle
        self._statuses[str(sandbox_id)] = {"total_actions": 0}
        self._snapshot_steps[str(sandbox_id)] = []
        self._install_task(handle)
        return handle

    def load_dataset(self, path):
        raise AssertionError(f"dataset should not be loaded in this test: {path}")

    def select_task_record(
        self,
        dataset,
        *,
        sandbox_index: int,
        default_agent_type: str,
        default_llm_service_type: str | None,
        default_task_description,
        default_task_config,
    ) -> BenchmarkTaskRecord:
        _ = (dataset, sandbox_index, default_llm_service_type)
        return BenchmarkTaskRecord(
            agent_type=default_agent_type,
            task_description=default_task_description,
            task_config=default_task_config,
        )

    def launch_task_record(self, sandbox_name: str, record: BenchmarkTaskRecord) -> SandboxHandle:
        _ = record
        return self.launch_sandbox_and_task(
            sandbox_name,
            agent_type="simulated",
            task_description=SimpleNamespace(prompt=""),
            task_config=SimpleNamespace(),
        )

    def drain_request_state_changes(self) -> int:
        return 0

    def set_snapshot_metadata(self, sandbox: SandboxHandle, **metadata: object) -> None:
        step = int(metadata["tree_search_step"])
        sandbox_id = str(sandbox.sandbox_id)
        self._snapshot_steps[sandbox_id].append(step)
        self._checkpoint_steps[sandbox_id] = step

    def checkpoint_manual(self, sandbox: SandboxHandle, leave_running: bool = False):
        sandbox_id = str(sandbox.sandbox_id)
        step = self._checkpoint_steps[sandbox_id]
        self.manual_checkpoint_calls.append((sandbox_id, leave_running))
        checkpoint_id = CheckpointId(f"manual-{sandbox_id}-{step}")
        self._checkpoint_to_step[str(checkpoint_id)] = step
        return SimpleNamespace(checkpoint_id=checkpoint_id)

    def wait_for_tree_search_checkpoints(
        self,
        sandbox_id: SandboxId,
        *,
        initial_steps: int,
        timeout_s: float = 45.0,
    ) -> dict[int, TreeSearchCheckpointRecord]:
        _ = timeout_s
        sandbox_id_text = str(sandbox_id)
        self.wait_for_tree_search_calls.append((sandbox_id_text, initial_steps))
        return {
            step: TreeSearchCheckpointRecord(
                CheckpointId(f"auto-{sandbox_id_text}-{step}"),
                replay_actions=step,
            )
            for step in range(1, initial_steps + 1)
        }

    def deactivate_sandbox_runtime(self, sandbox: SandboxHandle) -> None:
        sandbox_id = str(sandbox.sandbox_id)
        self.deactivated.append(sandbox_id)
        current = int(self._statuses[sandbox_id]["total_actions"])
        self.events.append(("deactivate", sandbox_id, current))
        if sandbox.task_future is not None and not sandbox.task_future.done():
            sandbox.task_future.set_result(None)

    def clone_tree_search_checkpoint_to_fork(
        self,
        source: SandboxHandle,
        checkpoint_id: CheckpointId,
        fork_name: str,
    ) -> SandboxHandle:
        _ = source
        step = self._checkpoint_to_step.get(str(checkpoint_id))
        if step is None:
            step = int(str(checkpoint_id).rsplit("-", 1)[1])
            self._checkpoint_to_step[str(checkpoint_id)] = step
        self.events.append(("clone", fork_name, step))
        fork = SandboxHandle(
            sandbox_id=SandboxId(fork_name),
            bundle_dir=Path("/tmp") / fork_name,
            status_port=self._next_port,
            last_status={"total_actions": 0},
            status_host="10.250.0.99",
            agent_type=source.agent_type,
            task_description=source.task_description,
            task_config=source.task_config,
        )
        self._next_port += 1
        self._statuses[str(fork.sandbox_id)] = {"total_actions": 0}
        self._checkpoint_steps[str(fork.sandbox_id)] = step
        self._handles[str(fork.sandbox_id)] = fork
        fork.task_run = _FakeTaskRun(self, fork)
        fork.task_future = None
        return fork

    def launch_task(
        self,
        agent_type,
        task_description,
        task_config,
        sandbox_id: str,
    ):
        _ = (agent_type, task_description, task_config)
        sandbox_id_text = str(sandbox_id)
        sandbox = self._handles[sandbox_id_text]
        self.task_launches.append(sandbox_id_text)
        current = int(self._statuses[sandbox_id_text]["total_actions"])
        self.events.append(("launch_task", sandbox_id_text, current))
        self._install_task(sandbox)
        return sandbox.task_run

    def restore_once(self, sandbox: SandboxHandle, checkpoint_id: CheckpointId):
        step = self._checkpoint_to_step[str(checkpoint_id)]
        with self._restore_lock:
            self._active_restores += 1
            self.max_concurrent_restores = max(self.max_concurrent_restores, self._active_restores)
        try:
            self.events.append(("restore", str(sandbox.sandbox_id), step))
            if self.restore_delay_s > 0:
                time.sleep(self.restore_delay_s)
            self._statuses[str(sandbox.sandbox_id)] = {"total_actions": max(0, step - 1)}
        finally:
            with self._restore_lock:
                self._active_restores -= 1
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            status=SimpleNamespace(value="succeeded"),
            started_at=now,
            finished_at=now + timedelta(milliseconds=1),
            message=None,
        )

    def destroy_sandbox_dataset(self, sandbox: SandboxHandle) -> None:
        self.destroyed.append(str(sandbox.sandbox_id))

    def inject_fault(self, sandbox: SandboxHandle) -> None:
        raise AssertionError(f"inject_fault should not be called for {sandbox.sandbox_id}")

    def notify_fault(self, sandbox: SandboxHandle, *, reason: str = "fault") -> None:
        _ = reason
        raise AssertionError(f"notify_fault should not be called for {sandbox.sandbox_id}")


class TreeSearchBenchmarkTests(unittest.TestCase):
    @staticmethod
    def _event_index(events: list[tuple[str, str, int]], kind: str, sandbox_id: str) -> int:
        return next(index for index, event in enumerate(events) if event[0] == kind and event[1] == sandbox_id)

    def _args(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "sandboxes": 1,
            "initial_steps": 4,
            "replay_points": 2,
            "fork_steps": 2,
            "auto_cr": False,
            "replay_mode": "sequential",
            "agent_type": "simulated",
            "llm_service_type": "simulated",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_manual_mode_runs_sources_independently(self) -> None:
        harness = _FakeHarness()
        harness.source_action_delay_s = 0.05

        rows = run_tree_search_benchmark(self._args(sandboxes=2, initial_steps=3, replay_points=1, fork_steps=1), harness)

        self.assertEqual(harness._source_launches, ["tree-source-0", "tree-source-1"])
        self.assertEqual(harness.wait_for_tree_search_calls, [])
        self.assertEqual(harness._snapshot_steps["tree-source-0"], [1, 2, 3])
        self.assertEqual(harness._snapshot_steps["tree-source-1"], [1, 2, 3])
        self.assertEqual(sorted(harness.manual_checkpoint_calls), [("tree-source-0", True)] * 3 + [("tree-source-1", True)] * 3)
        self.assertGreaterEqual(harness.max_concurrent_source_actions, 2)
        self.assertEqual(sorted(row["source_index"] for row in rows), [0, 1])
        self.assertEqual([row["replay_actions"] for row in rows], [1, 1])
        self.assertEqual(sorted(harness.task_launches), ["tree-fork-0-1", "tree-fork-1-1"])
        fork_progress_calls = sorted(call for call in harness.action_delta_calls if call[0].startswith("tree-fork-"))
        self.assertEqual(fork_progress_calls, [("tree-fork-0-1", 1), ("tree-fork-1-1", 1)])

    def test_auto_mode_runs_one_rollout_wave_and_stops_system_once(self) -> None:
        harness = _FakeHarness()
        harness.source_action_delay_s = 0.05

        rows = run_tree_search_benchmark(self._args(sandboxes=2, auto_cr=True, initial_steps=3, replay_points=1, fork_steps=1), harness)

        self.assertEqual(harness.manual_checkpoint_calls, [])
        self.assertEqual(harness.system.start_calls, 1)
        self.assertEqual(harness.system.stop_calls, 1)
        self.assertGreaterEqual(harness.max_concurrent_source_actions, 2)
        self.assertEqual(
            sorted(harness.wait_for_tree_search_calls),
            [("tree-source-0", 3), ("tree-source-1", 3)],
        )
        self.assertEqual(sorted(row["source_index"] for row in rows), [0, 1])
        self.assertEqual([row["replay_actions"] for row in rows], [1, 1])

    def test_concurrent_mode_restores_forks_in_parallel(self) -> None:
        harness = _FakeHarness()
        harness.restore_delay_s = 0.05

        rows = run_tree_search_benchmark(self._args(replay_mode="concurrent"), harness)

        clone_events = [event for event in harness.events if event[0] == "clone"]
        self.assertEqual([event[1] for event in clone_events], ["tree-fork-0-1", "tree-fork-0-2"])
        self.assertIn(("launch_task", "tree-fork-0-1", 0), harness.events)
        self.assertIn(("launch_task", "tree-fork-0-2", 1), harness.events)
        self.assertIn("tree-fork-0-1", harness.stop_requests)
        self.assertIn("tree-fork-0-2", harness.stop_requests)
        self.assertGreaterEqual(harness.max_concurrent_restores, 2)
        self.assertEqual([row["replay_step"] for row in rows], [1, 2])
        self.assertEqual([row["source_index"] for row in rows], [0, 0])

    def test_manual_mode_waits_for_source_task_stop_before_first_clone(self) -> None:
        harness = _FakeHarness()

        run_tree_search_benchmark(self._args(sandboxes=1, initial_steps=2, replay_points=1, fork_steps=1), harness)

        self.assertIn("tree-source-0", harness.stop_requests)
        self.assertLess(
            self._event_index(harness.events, "deactivate", "tree-source-0"),
            self._event_index(harness.events, "clone", "tree-fork-0-1"),
        )

    def test_auto_mode_waits_for_source_task_stop_before_first_clone(self) -> None:
        harness = _FakeHarness()

        run_tree_search_benchmark(self._args(sandboxes=1, auto_cr=True, initial_steps=2, replay_points=1, fork_steps=1), harness)

        self.assertIn("tree-source-0", harness.stop_requests)
        self.assertLess(
            self._event_index(harness.events, "deactivate", "tree-source-0"),
            self._event_index(harness.events, "clone", "tree-fork-0-1"),
        )

    def test_fork_task_can_finish_before_budget_without_manual_stop(self) -> None:
        harness = _FakeHarness()
        harness.fork_task_completion_delta = 1

        run_tree_search_benchmark(self._args(sandboxes=1, replay_points=1, fork_steps=3), harness)

        self.assertEqual(harness.task_launches, ["tree-fork-0-1"])
        self.assertNotIn("tree-fork-0-1", harness.stop_requests)

    def test_fork_task_is_manually_stopped_when_budget_is_exhausted(self) -> None:
        harness = _FakeHarness()

        run_tree_search_benchmark(self._args(sandboxes=1, replay_points=1, fork_steps=1), harness)

        self.assertIn("tree-fork-0-1", harness.task_launches)
        self.assertIn("tree-fork-0-1", harness.stop_requests)
        self.assertLess(
            self._event_index(harness.events, "request_stop", "tree-fork-0-1"),
            self._event_index(harness.events, "deactivate", "tree-fork-0-1"),
        )


if __name__ == "__main__":
    unittest.main()

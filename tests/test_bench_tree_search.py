from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from agent_cr import CheckpointId, SandboxId
from benchmarks.bench_tree_search import run_tree_search_benchmark
from benchmarks.real_host_scenario_base import SandboxHandle, TreeSearchCheckpointRecord


class _FakeStorage:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    def delete_all_checkpoints(self, sandbox_id) -> None:
        self.deleted.append(str(sandbox_id))


class _FakeHarness:
    def __init__(self) -> None:
        self.storage = _FakeStorage()
        self._next_port = 19000
        self._statuses: dict[str, dict[str, object]] = {}
        self._snapshot_steps: dict[str, list[int]] = {}
        self._checkpoint_steps: dict[str, int] = {}
        self.action_delta_calls: list[tuple[str, int]] = []
        self.checkpoint_if_due_calls: list[str] = []
        self.wait_for_tree_search_calls: list[tuple[str, int]] = []
        self.deactivated: list[str] = []
        self.destroyed: list[str] = []
        self.events: list[tuple[str, int]] = []
        self.auto_index = {
            1: TreeSearchCheckpointRecord(CheckpointId("auto-1"), replay_actions=1),
            2: TreeSearchCheckpointRecord(CheckpointId("auto-2"), replay_actions=2),
            3: TreeSearchCheckpointRecord(CheckpointId("auto-3"), replay_actions=3),
            4: TreeSearchCheckpointRecord(CheckpointId("auto-4"), replay_actions=4),
        }
        self._checkpoint_to_step = {str(record.checkpoint_id): step for step, record in self.auto_index.items()}

    def launch_sandbox(self, sandbox_name: str) -> SandboxHandle:
        sandbox_id = SandboxId(sandbox_name)
        handle = SandboxHandle(
            sandbox_id=sandbox_id,
            bundle_dir=Path("/tmp") / sandbox_name,
            status_port=self._next_port,
            last_status={"total_actions": 0},
        )
        self._next_port += 1
        self._statuses[str(sandbox_id)] = {"total_actions": 0}
        self._snapshot_steps[str(sandbox_id)] = []
        return handle

    def wait_for_action_delta(self, sandbox: SandboxHandle, *, delta: int) -> dict[str, object]:
        sandbox_id = str(sandbox.sandbox_id)
        self.action_delta_calls.append((sandbox_id, delta))
        current = int(self._statuses[sandbox_id]["total_actions"])
        payload = {"total_actions": current + delta}
        self._statuses[sandbox_id] = payload
        sandbox.last_status = payload
        return payload

    def drain_request_state_changes(self) -> int:
        return 0

    def set_snapshot_metadata(self, sandbox: SandboxHandle, **metadata: object) -> None:
        step = int(metadata["tree_search_step"])
        sandbox_id = str(sandbox.sandbox_id)
        self._snapshot_steps[sandbox_id].append(step)
        self._checkpoint_steps[sandbox_id] = step

    def checkpoint_if_due(self, sandbox: SandboxHandle):
        sandbox_id = str(sandbox.sandbox_id)
        step = self._checkpoint_steps[sandbox_id]
        self.checkpoint_if_due_calls.append(sandbox_id)
        checkpoint_id = CheckpointId(f"manual-{step}")
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
        self.wait_for_tree_search_calls.append((str(sandbox_id), initial_steps))
        return {step: self.auto_index[step] for step in range(1, initial_steps + 1)}

    def deactivate_sandbox_runtime(self, sandbox: SandboxHandle) -> None:
        self.deactivated.append(str(sandbox.sandbox_id))

    def clone_checkpoint_to_fork(
        self,
        source: SandboxHandle,
        checkpoint_id: CheckpointId,
        fork_name: str,
    ) -> SandboxHandle:
        _ = source
        step = self._checkpoint_to_step[str(checkpoint_id)]
        self.events.append(("clone", step))
        fork = SandboxHandle(
            sandbox_id=SandboxId(fork_name),
            bundle_dir=Path("/tmp") / fork_name,
            status_port=self._next_port,
            last_status={"total_actions": 0},
        )
        self._next_port += 1
        self._statuses[str(fork.sandbox_id)] = {"total_actions": 0}
        self._checkpoint_steps[str(fork.sandbox_id)] = step
        return fork

    def restore_once(self, sandbox: SandboxHandle, checkpoint_id: CheckpointId):
        step = self._checkpoint_to_step[str(checkpoint_id)]
        self.events.append(("restore", step))
        self._statuses[str(sandbox.sandbox_id)] = {"total_actions": max(0, step - 1)}
        now = datetime.now(timezone.utc)
        return SimpleNamespace(
            status=SimpleNamespace(value="succeeded"),
            started_at=now,
            finished_at=now + timedelta(milliseconds=1),
            message=None,
        )

    def poll_status(self, sandbox: SandboxHandle) -> dict[str, object]:
        return dict(self._statuses[str(sandbox.sandbox_id)])

    def destroy_sandbox_dataset(self, sandbox: SandboxHandle) -> None:
        self.destroyed.append(str(sandbox.sandbox_id))

    def inject_fault(self, sandbox: SandboxHandle) -> None:
        raise AssertionError(f"inject_fault should not be called for {sandbox.sandbox_id}")

    def notify_fault(self, sandbox: SandboxHandle, *, reason: str = "fault") -> None:
        _ = reason
        raise AssertionError(f"notify_fault should not be called for {sandbox.sandbox_id}")


class TreeSearchBenchmarkTests(unittest.TestCase):
    def _args(self, **overrides: object) -> argparse.Namespace:
        values: dict[str, object] = {
            "iters": 1,
            "initial_steps": 4,
            "replay_points": 2,
            "fork_steps": 2,
            "auto_cr": False,
            "replay_mode": "sequential",
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_manual_mode_checkpoints_every_step_and_waits_for_replay_progress(self) -> None:
        harness = _FakeHarness()

        rows = run_tree_search_benchmark(self._args(auto_cr=False), harness)

        self.assertEqual(harness.wait_for_tree_search_calls, [])
        self.assertEqual(harness._snapshot_steps["tree-source-0"], [1, 2, 3, 4])
        self.assertEqual(harness.checkpoint_if_due_calls, ["tree-source-0"] * 4)
        self.assertEqual([row["replay_actions"] for row in rows], [1, 2])
        fork_progress_calls = [call for call in harness.action_delta_calls if call[0].startswith("tree-fork-")]
        self.assertEqual(fork_progress_calls, [("tree-fork-0-1", 1), ("tree-fork-0-1", 1), ("tree-fork-0-2", 1), ("tree-fork-0-2", 1)])

    def test_auto_mode_waits_for_tree_search_checkpoints_without_sleep_or_faults(self) -> None:
        harness = _FakeHarness()

        with patch("benchmarks.bench_tree_search.time.sleep", side_effect=AssertionError("sleep not expected")):
            rows = run_tree_search_benchmark(self._args(auto_cr=True), harness)

        self.assertEqual(harness.checkpoint_if_due_calls, [])
        self.assertEqual(harness.wait_for_tree_search_calls, [("tree-source-0", 4)])
        self.assertEqual([row["replay_actions"] for row in rows], [1, 2])
        fork_progress_calls = [call for call in harness.action_delta_calls if call[0].startswith("tree-fork-")]
        self.assertEqual(fork_progress_calls, [("tree-fork-0-1", 1), ("tree-fork-0-1", 1), ("tree-fork-0-2", 1), ("tree-fork-0-2", 1)])

    def test_concurrent_mode_prepares_all_forks_before_serial_restore(self) -> None:
        harness = _FakeHarness()

        run_tree_search_benchmark(self._args(replay_mode="concurrent"), harness)

        self.assertEqual(
            harness.events,
            [("clone", 1), ("clone", 2), ("restore", 1), ("restore", 2)],
        )


if __name__ == "__main__":
    unittest.main()

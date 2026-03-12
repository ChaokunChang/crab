from __future__ import annotations

import unittest

from agent_cr import (
    CRScheduler,
    FaultToleranceCheckpointingPolicy,
    InMemorySandboxInspector,
    InMemorySchedulerStateStore,
    SandboxDescription,
    SandboxId,
    SandboxSnapshot,
    SchedulerConfig,
)
from agent_cr.models import utc_now


class RecordingSandboxManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, SandboxId]] = []
        self._items: dict[SandboxId, SandboxDescription] = {}
        self.fail_pause_for: set[SandboxId] = set()

    def add(self, sandbox_id: SandboxId) -> None:
        self._items[sandbox_id] = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name="runc",
            status="running",
        )

    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        _ = (runtime_name, metadata)
        raise NotImplementedError

    def stop(self, sandbox_id: SandboxId) -> None:
        self.calls.append(("stop", sandbox_id))
        self._items[sandbox_id] = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name="runc",
            status="stopped",
        )

    def pause(self, sandbox_id: SandboxId) -> None:
        self.calls.append(("pause", sandbox_id))
        if sandbox_id in self.fail_pause_for:
            raise RuntimeError("container not running")
        current = self._items[sandbox_id]
        self._items[sandbox_id] = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name=current.runtime_name,
            status="paused",
            metadata=current.metadata,
        )

    def resume(self, sandbox_id: SandboxId) -> None:
        self.calls.append(("resume", sandbox_id))
        current = self._items[sandbox_id]
        self._items[sandbox_id] = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name=current.runtime_name,
            status="running",
            metadata=current.metadata,
        )

    def delete(self, sandbox_id: SandboxId) -> None:
        self.calls.append(("delete", sandbox_id))
        self._items.pop(sandbox_id, None)

    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        return self._items[sandbox_id]


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.inspector = InMemorySandboxInspector()
        self.sandbox_manager = RecordingSandboxManager()
        self.sandbox_id = SandboxId("sbx-1")
        self.sandbox_manager.add(self.sandbox_id)
        self.scheduler = CRScheduler(
            SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            ),
            self.inspector,
            self.sandbox_manager,
            InMemorySchedulerStateStore(),
        )

    def test_query_resumes_sandbox_when_checkpoint_not_needed(self) -> None:
        self.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=self.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )

        decision = self.scheduler.query_checkpoint(self.sandbox_id)

        self.assertFalse(decision.should_checkpoint)
        self.assertEqual(self.sandbox_manager.calls, [("pause", self.sandbox_id), ("resume", self.sandbox_id)])
        self.assertEqual(self.sandbox_manager.describe(self.sandbox_id).status, "running")

    def test_query_returns_process_only_scope_and_keeps_sandbox_paused(self) -> None:
        self.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=self.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )

        decision = self.scheduler.query_checkpoint(self.sandbox_id)

        self.assertTrue(decision.should_checkpoint)
        self.assertTrue(decision.checkpoint_process)
        self.assertFalse(decision.checkpoint_filesystem)
        self.assertFalse(decision.leave_running)
        self.assertEqual(self.sandbox_manager.calls, [("pause", self.sandbox_id)])
        self.assertEqual(self.sandbox_manager.describe(self.sandbox_id).status, "paused")

    def test_query_promotes_filesystem_change_to_full_checkpoint(self) -> None:
        self.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=self.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=False,
                filesystem_changed=True,
                observed_at=utc_now(),
            )
        )

        decision = self.scheduler.query_checkpoint(self.sandbox_id)

        self.assertTrue(decision.should_checkpoint)
        self.assertTrue(decision.checkpoint_process)
        self.assertTrue(decision.checkpoint_filesystem)
        self.assertFalse(decision.leave_running)
        self.assertEqual(self.sandbox_manager.calls, [("pause", self.sandbox_id)])
        self.assertEqual(self.sandbox_manager.describe(self.sandbox_id).status, "paused")

    def test_evaluate_hydrates_last_checkpoint_from_state_store(self) -> None:
        observed_at = utc_now()
        last_checkpoint_at = observed_at.replace(microsecond=0)
        scheduler = CRScheduler(
            SchedulerConfig(
                min_checkpoint_interval_seconds=60.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            ),
            self.inspector,
            self.sandbox_manager,
            InMemorySchedulerStateStore(),
        )
        scheduler.mark_checkpoint_complete(self.sandbox_id, last_checkpoint_at)

        decision = scheduler.evaluate(
            SandboxSnapshot(
                sandbox_id=self.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=False,
                observed_at=observed_at,
                last_checkpoint_at=None,
            )
        )

        self.assertFalse(decision.should_checkpoint)
        self.assertEqual(decision.reason, "minimum_interval_not_elapsed")
        self.assertEqual(decision.policy_name, "default-checkpointing")
        self.assertFalse(decision.leave_running)

    def test_query_returns_both_scopes_when_both_dimensions_changed(self) -> None:
        self.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=self.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=True,
                observed_at=utc_now(),
            )
        )

        decision = self.scheduler.query_checkpoint(self.sandbox_id)

        self.assertTrue(decision.should_checkpoint)
        self.assertTrue(decision.checkpoint_process)
        self.assertTrue(decision.checkpoint_filesystem)
        self.assertFalse(decision.leave_running)

    def test_query_fault_tolerance_policy_sets_leave_running(self) -> None:
        scheduler = CRScheduler(
            SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            ),
            self.inspector,
            self.sandbox_manager,
            InMemorySchedulerStateStore(),
            None,
            FaultToleranceCheckpointingPolicy(SchedulerConfig()),
        )
        self.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=self.sandbox_id,
                runtime_name="runc",
                is_running=True,
                process_changed=True,
                filesystem_changed=True,
                observed_at=utc_now(),
            )
        )

        decision = scheduler.query_checkpoint(self.sandbox_id)

        self.assertTrue(decision.should_checkpoint)
        self.assertTrue(decision.leave_running)

    def test_query_handles_pause_failure_when_snapshot_is_not_running(self) -> None:
        self.sandbox_manager.fail_pause_for.add(self.sandbox_id)
        self.inspector.upsert_snapshot(
            SandboxSnapshot(
                sandbox_id=self.sandbox_id,
                runtime_name="runc",
                is_running=False,
                process_changed=False,
                filesystem_changed=False,
                observed_at=utc_now(),
            )
        )

        decision = self.scheduler.query_checkpoint(self.sandbox_id)

        self.assertFalse(decision.should_checkpoint)
        self.assertEqual(decision.reason, "sandbox_not_running")
        self.assertEqual(self.sandbox_manager.calls, [("pause", self.sandbox_id)])


if __name__ == "__main__":
    unittest.main()

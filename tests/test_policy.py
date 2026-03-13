from __future__ import annotations

import unittest
from datetime import timedelta

from agent_cr import (
    FaultToleranceCheckpointingPolicy,
    SandboxId,
    SchedulerConfig,
    SpotPreemptionCheckpointingPolicy,
    TreeSearchCheckpointingPolicy,
)
from agent_cr.models import SandboxSnapshot, utc_now
from agent_cr.scheduler import CheckpointingPolicy


class PolicyTests(unittest.TestCase):
    def test_checkpoint_when_no_previous_checkpoint(self) -> None:
        policy = CheckpointingPolicy(SchedulerConfig())
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=True,
            filesystem_changed=False,
            observed_at=utc_now(),
            last_checkpoint_at=None,
        )
        decision = policy.evaluate(snapshot)
        self.assertTrue(decision.should_checkpoint)
        self.assertEqual(decision.reason, "no_previous_checkpoint")
        self.assertEqual(decision.policy_name, "default-checkpointing")
        self.assertFalse(decision.leave_running)

    def test_no_checkpoint_without_change_signal(self) -> None:
        policy = CheckpointingPolicy(
            SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=1000.0,
                require_change_signal=True,
            )
        )
        now = utc_now()
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=False,
            filesystem_changed=False,
            observed_at=now,
            last_checkpoint_at=now - timedelta(seconds=120),
        )
        decision = policy.evaluate(snapshot)
        self.assertFalse(decision.should_checkpoint)
        self.assertEqual(decision.reason, "no_change_signal")
        self.assertFalse(decision.leave_running)

    def test_force_interval_overrides_min_interval(self) -> None:
        policy = CheckpointingPolicy(
            SchedulerConfig(
                min_checkpoint_interval_seconds=60.0,
                force_checkpoint_after_seconds=10.0,
                require_change_signal=False,
            )
        )
        now = utc_now()
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=False,
            filesystem_changed=False,
            observed_at=now,
            last_checkpoint_at=now - timedelta(seconds=12),
        )
        decision = policy.evaluate(snapshot)
        self.assertTrue(decision.should_checkpoint)
        self.assertEqual(decision.reason, "force_interval_elapsed")
        self.assertFalse(decision.leave_running)

    def test_minimum_interval_defers_checkpoint(self) -> None:
        policy = CheckpointingPolicy(
            SchedulerConfig(
                min_checkpoint_interval_seconds=60.0,
                force_checkpoint_after_seconds=1000.0,
                require_change_signal=True,
            )
        )
        now = utc_now()
        last_checkpoint_at = now - timedelta(seconds=12)
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=True,
            filesystem_changed=False,
            observed_at=now,
            last_checkpoint_at=last_checkpoint_at,
        )
        decision = policy.evaluate(snapshot)
        self.assertFalse(decision.should_checkpoint)
        self.assertEqual(decision.reason, "minimum_interval_not_elapsed")
        self.assertFalse(decision.leave_running)

    def test_prefers_checkpoint_during_llm_request_window(self) -> None:
        policy = CheckpointingPolicy(
            SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=1000.0,
                require_change_signal=True,
                prefer_checkpoint_during_llm_request=True,
            )
        )
        now = utc_now()
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=True,
            filesystem_changed=False,
            observed_at=now,
            last_checkpoint_at=now - timedelta(seconds=12),
            metadata={"llm_request_in_flight": True},
        )
        decision = policy.evaluate(snapshot)
        self.assertTrue(decision.should_checkpoint)
        self.assertEqual(decision.reason, "llm_request_window_available")
        self.assertFalse(decision.leave_running)

    def test_prefers_llm_request_window_for_first_checkpoint(self) -> None:
        policy = CheckpointingPolicy(
            SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=1000.0,
                require_change_signal=True,
                prefer_checkpoint_during_llm_request=True,
            )
        )
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=True,
            filesystem_changed=False,
            observed_at=utc_now(),
            last_checkpoint_at=None,
            metadata={"llm_request_in_flight": True},
        )
        decision = policy.evaluate(snapshot)
        self.assertTrue(decision.should_checkpoint)
        self.assertEqual(decision.reason, "llm_request_window_available")
        self.assertFalse(decision.leave_running)

    def test_requires_llm_request_when_configured(self) -> None:
        policy = CheckpointingPolicy(
            SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=1000.0,
                require_change_signal=True,
                require_llm_request_for_checkpoint=True,
            )
        )
        now = utc_now()
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=True,
            filesystem_changed=False,
            observed_at=now,
            last_checkpoint_at=now - timedelta(seconds=12),
            metadata={"llm_request_in_flight": False},
        )
        decision = policy.evaluate(snapshot)
        self.assertFalse(decision.should_checkpoint)
        self.assertEqual(decision.reason, "llm_request_required")
        self.assertFalse(decision.leave_running)

    def test_fault_tolerance_policy_keeps_sandbox_running(self) -> None:
        policy = FaultToleranceCheckpointingPolicy(
            SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            )
        )
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=True,
            filesystem_changed=True,
            observed_at=utc_now(),
        )

        decision = policy.evaluate(snapshot)

        self.assertTrue(decision.should_checkpoint)
        self.assertTrue(decision.leave_running)
        self.assertEqual(decision.policy_name, "fault-tolerance")

    def test_fault_tolerance_policy_allows_request_window_checkpoint(self) -> None:
        policy = FaultToleranceCheckpointingPolicy(
            SchedulerConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
                prefer_checkpoint_during_llm_request=True,
            )
        )
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=True,
            filesystem_changed=True,
            observed_at=utc_now(),
            metadata={"llm_request_in_flight": True},
        )

        decision = policy.evaluate(snapshot)

        self.assertTrue(decision.should_checkpoint)
        self.assertEqual(decision.reason, "llm_request_window_available")
        self.assertTrue(decision.leave_running)
        self.assertEqual(decision.policy_name, "fault-tolerance")

    def test_spot_policy_requires_preemption_notice(self) -> None:
        policy = SpotPreemptionCheckpointingPolicy(SchedulerConfig())
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=True,
            filesystem_changed=True,
            observed_at=utc_now(),
        )

        decision = policy.evaluate(snapshot)

        self.assertFalse(decision.should_checkpoint)
        self.assertEqual(decision.reason, "awaiting_preemption_notice")

    def test_spot_policy_checkpoints_on_preemption_notice(self) -> None:
        policy = SpotPreemptionCheckpointingPolicy(SchedulerConfig())
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=False,
            filesystem_changed=False,
            observed_at=utc_now(),
            metadata={
                "preemption_notice": True,
                "preemption_grace_remaining_seconds": 42.0,
            },
        )

        decision = policy.evaluate(snapshot)

        self.assertTrue(decision.should_checkpoint)
        self.assertFalse(decision.leave_running)
        self.assertEqual(decision.reason, "preemption_notice_received")

    def test_tree_search_policy_checkpoints_each_step(self) -> None:
        policy = TreeSearchCheckpointingPolicy()
        snapshot = SandboxSnapshot(
            sandbox_id=SandboxId("sbx-1"),
            runtime_name="docker",
            is_running=True,
            process_changed=False,
            filesystem_changed=False,
            observed_at=utc_now(),
            metadata={"tree_search_step": 7},
        )

        decision = policy.evaluate(snapshot)

        self.assertTrue(decision.should_checkpoint)
        self.assertTrue(decision.leave_running)
        self.assertEqual(decision.metadata["tree_search_step"], 7)


if __name__ == "__main__":
    unittest.main()

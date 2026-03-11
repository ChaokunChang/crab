from __future__ import annotations

import unittest
from datetime import timedelta

from agent_cr import SandboxId, SchedulerConfig
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


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import wait as concurrent_wait
from dataclasses import dataclass, field
from typing import Any

from agent_cr import CheckpointId, CheckpointManifest, SandboxId
from agent_cr.models import utc_now
from benchmarks.real_host_scenario_base import _SpeculativeSandboxController


@dataclass
class _FakeSandbox:
    sandbox_id: SandboxId
    task_run: Any = None


@dataclass
class _FakeRestoreStatus:
    value: str = "succeeded"


@dataclass
class _FakeRestoreResult:
    status: _FakeRestoreStatus = field(default_factory=_FakeRestoreStatus)
    message: str = ""


class _FakeStorage:
    """Minimal CheckpointManager-shaped stub.

    Drives ``_SpeculativeSandboxController.kick_prefork`` and ``ensure_fork``
    without touching real storage: maintains a checkpoint id list and a
    manifest map keyed on (sandbox_id, checkpoint_id).
    """

    def __init__(self) -> None:
        self._checkpoints: dict[SandboxId, list[CheckpointId]] = {}
        self._manifests: dict[tuple[SandboxId, CheckpointId], CheckpointManifest] = {}

    def append_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        process_artifact_count: int = 1,
    ) -> None:
        self._checkpoints.setdefault(sandbox_id, []).append(checkpoint_id)
        self._manifests[(sandbox_id, checkpoint_id)] = CheckpointManifest(
            schema_version="v1",
            checkpoint_id=checkpoint_id,
            sandbox_id=sandbox_id,
            created_at=utc_now(),
            runtime_name="runc",
            runtime_version=None,
            process_artifacts=[
                # process_artifacts is a tuple-of-references in the model;
                # any non-empty iterable suffices for the controller checks.
                _FakeArtifactRef()
                for _ in range(process_artifact_count)
            ],
            filesystem_artifacts=[],
            metadata={},
        )

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        return list(self._checkpoints.get(sandbox_id, []))

    def get_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        try:
            return self._manifests[(sandbox_id, checkpoint_id)]
        except KeyError as exc:
            raise FileNotFoundError(checkpoint_id) from exc


@dataclass
class _FakeArtifactRef:
    """Sentinel — only truthiness is read by the controller."""

    name: str = "stub"


class _FakeHarness:
    """Stands in for ``RealHostScenarioHarness`` in controller tests.

    Records every clone/restore/destroy call so tests can assert on the
    prefork sequence without booting runc + ZFS.
    """

    def __init__(self, *, fork_restore_succeeds: bool = True, restore_delay_s: float = 0.0) -> None:
        self.storage = _FakeStorage()
        self._fork_restore_succeeds = fork_restore_succeeds
        self._restore_delay_s = restore_delay_s
        self.lock = threading.Lock()
        self.cloned: list[tuple[SandboxId, CheckpointId, str]] = []
        self.restored: list[tuple[SandboxId, CheckpointId]] = []
        self.destroyed: list[SandboxId] = []
        self._next_id_counter = 0

    def clone_checkpoint_to_fork(
        self,
        source: _FakeSandbox,
        checkpoint_id: CheckpointId,
        fork_name: str,
    ) -> _FakeSandbox:
        with self.lock:
            self._next_id_counter += 1
            fork_id = SandboxId(f"{fork_name}-{self._next_id_counter}")
            self.cloned.append((source.sandbox_id, checkpoint_id, fork_name))
        return _FakeSandbox(sandbox_id=fork_id)

    def restore_once(
        self,
        sandbox: _FakeSandbox,
        checkpoint_id: CheckpointId,
    ) -> _FakeRestoreResult:
        if self._restore_delay_s > 0:
            time.sleep(self._restore_delay_s)
        with self.lock:
            self.restored.append((sandbox.sandbox_id, checkpoint_id))
        if not self._fork_restore_succeeds:
            return _FakeRestoreResult(status=_FakeRestoreStatus(value="failed"))
        return _FakeRestoreResult()

    def destroy_sandbox_dataset(self, sandbox: _FakeSandbox) -> None:
        with self.lock:
            self.destroyed.append(sandbox.sandbox_id)


class SpeculativePreforkTests(unittest.TestCase):
    def _make_controller(
        self,
        harness: _FakeHarness,
        *,
        background_prefork_enabled: bool = True,
        prefork_min_interval_seconds: float = 0.0,
        prefork_wait_timeout_seconds: float = 5.0,
    ) -> _SpeculativeSandboxController:
        active = _FakeSandbox(sandbox_id=SandboxId("active"))
        return _SpeculativeSandboxController(
            harness,  # type: ignore[arg-type]
            active,  # type: ignore[arg-type]
            background_prefork_enabled=background_prefork_enabled,
            prefork_min_interval_seconds=prefork_min_interval_seconds,
            prefork_wait_timeout_seconds=prefork_wait_timeout_seconds,
        )

    def _wait_for_prefork(
        self,
        controller: _SpeculativeSandboxController,
        timeout: float = 5.0,
    ) -> None:
        future = controller._prefork_future
        if future is None:
            return
        concurrent_wait([future], timeout=timeout)

    def test_prefork_disabled_is_noop(self) -> None:
        harness = _FakeHarness()
        harness.storage.append_checkpoint(SandboxId("active"), CheckpointId("ck-1"))
        controller = self._make_controller(harness, background_prefork_enabled=False)
        try:
            controller.kick_prefork()
            # No executor created when disabled; no work scheduled.
            self.assertIsNone(controller._prefork_executor)
            self.assertEqual(harness.cloned, [])
            self.assertEqual(harness.restored, [])
        finally:
            controller.shutdown()

    def test_prefork_warms_cache_with_latest_checkpoint(self) -> None:
        harness = _FakeHarness()
        harness.storage.append_checkpoint(SandboxId("active"), CheckpointId("ck-1"))
        controller = self._make_controller(harness)
        try:
            controller.kick_prefork()
            self._wait_for_prefork(controller)
            with controller._lock:
                cached = controller._cached_fork
                cached_id = controller._cached_fork_checkpoint_id
            self.assertIsNotNone(cached)
            self.assertEqual(cached_id, CheckpointId("ck-1"))
            self.assertEqual(len(harness.cloned), 1)
            self.assertEqual(harness.cloned[0][1], CheckpointId("ck-1"))
            self.assertEqual(len(harness.restored), 1)
        finally:
            controller.shutdown()

    def test_ensure_fork_returns_warmed_prefork_without_new_clone(self) -> None:
        harness = _FakeHarness()
        sbx = SandboxId("active")
        harness.storage.append_checkpoint(sbx, CheckpointId("ck-1"))
        controller = self._make_controller(harness)
        try:
            controller.kick_prefork()
            self._wait_for_prefork(controller)
            outcome = controller.ensure_fork()
            assert outcome is not None
            fork, created, reused = outcome
            self.assertFalse(created)
            self.assertTrue(reused)
            # Only the prefork's clone happened; ensure_fork did not clone again.
            self.assertEqual(len(harness.cloned), 1)
        finally:
            controller.shutdown()

    def test_ensure_fork_discards_stale_prefork_for_newer_checkpoint(self) -> None:
        harness = _FakeHarness()
        sbx = SandboxId("active")
        harness.storage.append_checkpoint(sbx, CheckpointId("ck-1"))
        controller = self._make_controller(harness)
        try:
            controller.kick_prefork()
            self._wait_for_prefork(controller)
            # New checkpoint lands AFTER the prefork commit.
            harness.storage.append_checkpoint(sbx, CheckpointId("ck-2"))
            outcome = controller.ensure_fork()
            assert outcome is not None
            _fork, created, reused = outcome
            # Stale prefork dropped; synchronous path created a fresh one.
            self.assertTrue(created)
            self.assertFalse(reused)
            self.assertGreaterEqual(len(harness.cloned), 2)
            # The first cloned fork (stale prefork) was destroyed.
            self.assertGreaterEqual(len(harness.destroyed), 1)
        finally:
            controller.shutdown()

    def test_kick_prefork_is_deduped_while_in_flight(self) -> None:
        harness = _FakeHarness(restore_delay_s=0.05)
        sbx = SandboxId("active")
        harness.storage.append_checkpoint(sbx, CheckpointId("ck-1"))
        controller = self._make_controller(harness)
        try:
            controller.kick_prefork()
            # Second kick during the in-flight window must not schedule a
            # second prefork — that would duplicate clone + restore work.
            controller.kick_prefork()
            self._wait_for_prefork(controller)
            self.assertEqual(len(harness.cloned), 1)
        finally:
            controller.shutdown()

    def test_kick_prefork_skips_when_cache_already_warm(self) -> None:
        harness = _FakeHarness()
        sbx = SandboxId("active")
        harness.storage.append_checkpoint(sbx, CheckpointId("ck-1"))
        controller = self._make_controller(harness)
        try:
            controller.kick_prefork()
            self._wait_for_prefork(controller)
            controller.kick_prefork()  # cache hot; should no-op
            self._wait_for_prefork(controller)
            self.assertEqual(len(harness.cloned), 1)
        finally:
            controller.shutdown()

    def test_invalidate_during_prefork_destroys_completed_fork(self) -> None:
        harness = _FakeHarness(restore_delay_s=0.1)
        sbx = SandboxId("active")
        harness.storage.append_checkpoint(sbx, CheckpointId("ck-1"))
        controller = self._make_controller(harness)
        try:
            controller.kick_prefork()
            # Bump epoch while prefork is mid-restore so the prefork's
            # commit gate sees a stale epoch and discards the fork.
            time.sleep(0.01)
            controller.invalidate_fork()
            self._wait_for_prefork(controller)
            with controller._lock:
                self.assertIsNone(controller._cached_fork)
            self.assertGreaterEqual(len(harness.destroyed), 1)
        finally:
            controller.shutdown()

    def test_global_throttle_limits_concurrent_preforks_across_sandboxes(self) -> None:
        # Throttle = 1 means at most one prefork in flight across both
        # controllers; the second sandbox's kick must skip rather than
        # queue. Use restore_delay_s so the first prefork is still busy
        # when the second kicks.
        # Reset the class-level semaphore state from any earlier test.
        _SpeculativeSandboxController._global_prefork_semaphore = None
        _SpeculativeSandboxController._global_prefork_max = 0
        harness_a = _FakeHarness(restore_delay_s=0.2)
        harness_b = _FakeHarness(restore_delay_s=0.2)
        harness_a.storage.append_checkpoint(SandboxId("active-a"), CheckpointId("ck-a1"))
        harness_b.storage.append_checkpoint(SandboxId("active-b"), CheckpointId("ck-b1"))
        active_a = _FakeSandbox(sandbox_id=SandboxId("active-a"))
        active_b = _FakeSandbox(sandbox_id=SandboxId("active-b"))
        controller_a = _SpeculativeSandboxController(
            harness_a, active_a,
            background_prefork_enabled=True,
            prefork_max_concurrent_global=1,
        )
        controller_b = _SpeculativeSandboxController(
            harness_b, active_b,
            background_prefork_enabled=True,
            prefork_max_concurrent_global=1,
        )
        try:
            controller_a.kick_prefork()  # acquires the only global slot
            controller_b.kick_prefork()  # must skip (throttle)
            self._wait_for_prefork(controller_a)
            self._wait_for_prefork(controller_b)
            # Only A produced a clone; B was throttled.
            self.assertEqual(len(harness_a.cloned), 1)
            self.assertEqual(len(harness_b.cloned), 0)
            self.assertEqual(controller_b._prefork_skipped_throttled, 1)
            # After A finishes, B's next kick can proceed.
            controller_b.kick_prefork()
            self._wait_for_prefork(controller_b)
            self.assertEqual(len(harness_b.cloned), 1)
        finally:
            controller_a.shutdown()
            controller_b.shutdown()

    def test_global_throttle_disabled_when_max_concurrent_zero(self) -> None:
        _SpeculativeSandboxController._global_prefork_semaphore = None
        _SpeculativeSandboxController._global_prefork_max = 0
        harness = _FakeHarness()
        harness.storage.append_checkpoint(SandboxId("active"), CheckpointId("ck-1"))
        controller = self._make_controller(harness)
        try:
            controller.kick_prefork()
            self._wait_for_prefork(controller)
            self.assertEqual(controller._prefork_skipped_throttled, 0)
            self.assertEqual(len(harness.cloned), 1)
        finally:
            controller.shutdown()

    def test_kick_prefork_skips_fs_only_checkpoint(self) -> None:
        harness = _FakeHarness()
        sbx = SandboxId("active")
        harness.storage.append_checkpoint(
            sbx, CheckpointId("ck-fs"), process_artifact_count=0
        )
        controller = self._make_controller(harness)
        try:
            controller.kick_prefork()
            # No process artifacts → controller refuses to fork (matches
            # ensure_fork's fs-only safeguard).
            self.assertEqual(harness.cloned, [])
            self.assertIsNone(controller._prefork_future)
        finally:
            controller.shutdown()


if __name__ == "__main__":
    unittest.main()

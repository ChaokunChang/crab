from __future__ import annotations

import unittest

from agent_cr import ArtifactKind, ArtifactReference, CheckpointId, CheckpointManifest, SandboxId
from agent_cr.models import utc_now
from benchmarks.bench_tree_search import choose_replay_steps
from benchmarks.real_host_scenario_base import (
    compute_summary,
    resolve_checkpoint_copy_plan,
    select_injected_indices,
    total_actions,
)


class BenchmarkHelperTests(unittest.TestCase):
    def test_choose_replay_steps_is_deterministic(self) -> None:
        self.assertEqual(choose_replay_steps(6, 2), [1, 3])
        self.assertEqual(choose_replay_steps(4, 10), [1, 2, 3])

    def test_compute_summary_averages_metrics(self) -> None:
        rows = [
            {"checkpoint_ms": 10.0, "restore_ms": 20.0},
            {"checkpoint_ms": 30.0, "restore_ms": 40.0},
        ]
        self.assertEqual(
            compute_summary(rows, ["checkpoint_ms", "restore_ms"]),
            {"checkpoint_ms": 20.0, "restore_ms": 30.0},
        )

    def test_total_actions_reads_payload(self) -> None:
        self.assertEqual(total_actions({"total_actions": 7}), 7)

    def test_select_injected_indices_honors_first_forced_iteration(self) -> None:
        import random

        rng = random.Random(0)
        self.assertEqual(
            select_injected_indices(3, iteration=1, rate=1.0, first_forced_iteration=3, rng=rng),
            [],
        )
        self.assertEqual(
            select_injected_indices(3, iteration=2, rate=1.0, first_forced_iteration=3, rng=rng),
            [],
        )
        forced = select_injected_indices(3, iteration=3, rate=0.0, first_forced_iteration=3, rng=rng)
        self.assertEqual(forced, [0])

    def test_resolve_checkpoint_copy_plan_only_includes_required_ancestors(self) -> None:
        sid = SandboxId("sbx")
        checkpoint_order = [CheckpointId("ckpt-1"), CheckpointId("ckpt-2"), CheckpointId("ckpt-3")]
        process_ref = ArtifactReference(
            kind=ArtifactKind.PROCESS,
            name="process_checkpoint.json",
            relative_path="artifacts/sbx/ckpt-1/process/process_checkpoint.json",
            size_bytes=1,
            sha256="0" * 64,
            metadata={},
        )
        filesystem_ref = ArtifactReference(
            kind=ArtifactKind.FILESYSTEM,
            name="filesystem_checkpoint.json",
            relative_path="artifacts/sbx/ckpt-1/filesystem/filesystem_checkpoint.json",
            size_bytes=1,
            sha256="1" * 64,
            metadata={},
        )
        manifests = {
            CheckpointId("ckpt-1"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-1"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[process_ref],
                filesystem_artifacts=[filesystem_ref],
                metadata={},
            ).with_integrity(),
            CheckpointId("ckpt-2"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-2"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity(),
            CheckpointId("ckpt-3"): CheckpointManifest(
                schema_version="v1",
                checkpoint_id=CheckpointId("ckpt-3"),
                sandbox_id=sid,
                created_at=utc_now(),
                runtime_name="runc",
                runtime_version=None,
                process_artifacts=[process_ref],
                filesystem_artifacts=[],
                metadata={},
            ).with_integrity(),
        }

        self.assertEqual(
            resolve_checkpoint_copy_plan(checkpoint_order, manifests, CheckpointId("ckpt-3")),
            [
                (CheckpointId("ckpt-1"), False, True),
                (CheckpointId("ckpt-3"), True, False),
            ],
        )


if __name__ == "__main__":
    unittest.main()

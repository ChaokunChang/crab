from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from crab.ids import CheckpointId, JobId, SandboxId
from crab.models import CheckpointJob, RuntimeCapabilities
from crab.runtime.in_memory import InMemoryRuntime
from crab.workers.process import AdapterProcessCWorker


class _IncrementalStubRuntime(InMemoryRuntime):
    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_process_checkpoint=True,
            supports_filesystem_checkpoint=True,
            supports_incremental_process=True,
        )

    def pre_dump_location(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> str | None:
        return f"/fake/{sandbox_id}/{checkpoint_id}/pre_dump"

    def process_checkpoint_location(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> str | None:
        return f"/fake/{sandbox_id}/{checkpoint_id}/process"


class WorkerIncrementalDispatchTests(unittest.TestCase):
    def _payload(self, runtime, job, checkpoint_id):
        worker = AdapterProcessCWorker(runtime)
        result = worker.checkpoint(job, checkpoint_id)
        return json.loads(result.artifacts[0].data)

    def _job(self, **kwargs) -> CheckpointJob:
        defaults = dict(
            job_id=JobId("j-1"),
            sandbox_id=SandboxId("sbx-1"),
            requested_at=datetime.now(timezone.utc),
        )
        defaults.update(kwargs)
        return CheckpointJob(**defaults)

    def test_full_path_records_no_pre_dump_artifacts(self) -> None:
        payload = self._payload(InMemoryRuntime(), self._job(), CheckpointId("c-1"))
        self.assertEqual(payload["process_kind"], "full")
        self.assertNotIn("pre_dump_status", payload)
        self.assertNotIn("parent_checkpoint_id", payload)
        self.assertNotIn("pre_dump_location", payload)

    def test_incremental_runtime_capable_runs_two_phase(self) -> None:
        runtime = _IncrementalStubRuntime()
        job = self._job(
            is_incremental_process=True,
            parent_process_checkpoint_id=CheckpointId("c-prev"),
            produce_pre_dump=True,
        )
        payload = self._payload(runtime, job, CheckpointId("c-2"))
        self.assertEqual(payload["process_kind"], "incremental")
        self.assertEqual(payload["parent_checkpoint_id"], "c-prev")
        self.assertEqual(payload["pre_dump_location"], "/fake/sbx-1/c-2/pre_dump")
        # Pre-dump command carries the previous checkpoint id as parent.
        self.assertIn("--parent=c-prev", payload["pre_dump_status"]["command"])
        # Final dump command's parent is the sibling pre_dump (this ckpt id).
        self.assertIn("--parent=c-2", payload["status"]["command"])

    def test_chain_anchor_runs_pair_without_parent(self) -> None:
        # Anchors (chain root, chain reset) produce a pre_dump with no
        # parent so subsequent incrementals can chain off them. Manifest
        # kind stays "full" because the anchor itself doesn't chain.
        runtime = _IncrementalStubRuntime()
        job = self._job(produce_pre_dump=True)
        payload = self._payload(runtime, job, CheckpointId("c-anchor"))
        self.assertEqual(payload["process_kind"], "full")
        self.assertNotIn("parent_checkpoint_id", payload)
        self.assertEqual(payload["pre_dump_location"], "/fake/sbx-1/c-anchor/pre_dump")
        self.assertIn("pre_dump_status", payload)
        # Pre-dump has no --parent flag (anchor dumps full memory).
        self.assertNotIn("--parent", " ".join(payload["pre_dump_status"]["command"]))
        # Final dump still chains off our just-taken pre_dump (same ckpt id).
        self.assertIn("--parent=c-anchor", payload["status"]["command"])

    def test_incremental_falls_back_to_full_when_runtime_lacks_capability(self) -> None:
        # InMemoryRuntime advertises supports_incremental_process=False, so the
        # worker must silently fall back to a single full dump rather than
        # invoking pre_dump_process. Logged warning is captured separately.
        job = self._job(
            is_incremental_process=True,
            parent_process_checkpoint_id=CheckpointId("c-prev"),
            produce_pre_dump=True,
        )
        payload = self._payload(InMemoryRuntime(), job, CheckpointId("c-2"))
        self.assertEqual(payload["process_kind"], "full")
        self.assertNotIn("pre_dump_status", payload)


if __name__ == "__main__":
    unittest.main()

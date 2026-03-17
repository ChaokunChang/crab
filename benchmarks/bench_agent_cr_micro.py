#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_cr import (
    ArtifactKind,
    ArtifactPayload,
    CheckpointId,
    CheckpointJob,
    JobId,
    SandboxId,
    SandboxSnapshot,
    StorageConfig,
    build_default_system,
)
from agent_cr.models import utc_now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent-CR interface microbench")
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--storage-iters", type=int, default=200)
    p.add_argument("--executor-jobs", type=int, default=64)
    p.add_argument("--runtime", choices=["docker", "runc"], default="docker")
    p.add_argument("--out", default="")
    return p.parse_args()


def bench_scheduler_eval(system, sandbox_id: SandboxId, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        snapshot = SandboxSnapshot(
            sandbox_id=sandbox_id,
            runtime_name="dry-run",
            is_running=True,
            process_changed=True,
            filesystem_changed=True,
            observed_at=utc_now(),
            last_checkpoint_at=None,
        )
        system.scheduler.evaluate(snapshot)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def bench_storage(system, sandbox_id: SandboxId, iters: int) -> float:
    t0 = time.perf_counter()
    for i in range(iters):
        ckpt = CheckpointId.new(prefix="bench")
        payload = ArtifactPayload(
            kind=ArtifactKind.METADATA,
            name=f"meta-{i}.txt",
            data=f"artifact-{i}".encode("utf-8"),
        )
        ref = system.storage.put_artifact(sandbox_id, ckpt, payload)
        _ = system.storage.get_artifact(sandbox_id, ckpt, ref)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def bench_executor(system, sandbox_id: SandboxId, jobs: int) -> float:
    reqs = [
        CheckpointJob(
            job_id=JobId.new(prefix="benchcp"),
            sandbox_id=sandbox_id,
            requested_at=utc_now(),
            reason="bench",
        )
        for _ in range(jobs)
    ]
    t0 = time.perf_counter()
    _ = system.executor.run_checkpoints(reqs)
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory(prefix="agent_cr_micro_") as tmp:
        system = build_default_system(
            storage_root=tmp,
            runtime=args.runtime,
            storage_config=StorageConfig(root_dir=Path(tmp)),
        )
        sandbox_id = SandboxId("bench-sandbox")
        try:
            eval_ms = bench_scheduler_eval(system, sandbox_id, args.iters)
            storage_ms = bench_storage(system, sandbox_id, args.storage_iters)
            executor_ms = bench_executor(system, sandbox_id, args.executor_jobs)

            rows = [
                {
                    "metric": "scheduler_eval_ms_total",
                    "value": eval_ms,
                    "iters": args.iters,
                },
                {
                    "metric": "storage_put_get_ms_total",
                    "value": storage_ms,
                    "iters": args.storage_iters,
                },
                {
                    "metric": "executor_run_checkpoints_ms_total",
                    "value": executor_ms,
                    "iters": args.executor_jobs,
                },
            ]

            if args.out:
                with open(args.out, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=["metric", "value", "iters"])
                    w.writeheader()
                    for row in rows:
                        w.writerow(row)

            for row in rows:
                print(f"{row['metric']}: {row['value']:.3f} (n={row['iters']})")
        finally:
            system.executor.shutdown()


if __name__ == "__main__":
    main()

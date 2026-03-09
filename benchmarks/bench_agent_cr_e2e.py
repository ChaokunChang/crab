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
    JobId,
    PolicyConfig,
    RestoreJob,
    SandboxId,
    SandboxSnapshot,
    StorageConfig,
    build_default_system,
)
from agent_cr.models import utc_now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Agent-CR simulated end-to-end benchmark")
    p.add_argument("--sandboxes", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--runtime", choices=["docker", "runc"], default="docker")
    p.add_argument("--out", default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with tempfile.TemporaryDirectory(prefix="agent_cr_e2e_") as tmp:
        system = build_default_system(
            storage_root=tmp,
            runtime=args.runtime,
            storage_config=StorageConfig(root_dir=Path(tmp)),
            policy_config=PolicyConfig(
                min_checkpoint_interval_seconds=0.0,
                force_checkpoint_after_seconds=0.0,
                require_change_signal=True,
            ),
        )

        sandboxes = [SandboxId(f"sandbox-{i}") for i in range(args.sandboxes)]
        for sid in sandboxes:
            system.inspector.upsert_snapshot(
                SandboxSnapshot(
                    sandbox_id=sid,
                    runtime_name=args.runtime,
                    is_running=True,
                    process_changed=True,
                    filesystem_changed=True,
                    observed_at=utc_now(),
                )
            )

        rows = []
        for i in range(args.iters):
            t0 = time.perf_counter()
            checkpoint_jobs = []
            for sid in sandboxes:
                maybe_job = system.scheduler.poll_and_schedule(sid)
                if maybe_job is not None:
                    checkpoint_jobs.append(maybe_job)
            ckpt_results = system.executor.run_checkpoints(checkpoint_jobs)
            t1 = time.perf_counter()

            restore_jobs = [
                RestoreJob(
                    job_id=JobId.new(prefix="restore"),
                    sandbox_id=r.sandbox_id,
                    checkpoint_id=r.checkpoint_id,
                    requested_at=utc_now(),
                    reason="bench_restore",
                )
                for r in ckpt_results
            ]
            restore_results = system.executor.run_restores(restore_jobs)
            t2 = time.perf_counter()

            rows.append(
                {
                    "iter": i,
                    "sandboxes": args.sandboxes,
                    "checkpoints": len(ckpt_results),
                    "restores": len(restore_results),
                    "checkpoint_batch_ms": (t1 - t0) * 1000.0,
                    "restore_batch_ms": (t2 - t1) * 1000.0,
                    "success_ratio": (
                        sum(1 for x in restore_results if x.status.value == "succeeded")
                        / max(1, len(restore_results))
                    ),
                }
            )

            for sid in sandboxes:
                obs = utc_now()
                system.inspector.upsert_snapshot(
                    SandboxSnapshot(
                        sandbox_id=sid,
                        runtime_name=args.runtime,
                        is_running=True,
                        process_changed=True,
                        filesystem_changed=(i % 2 == 0),
                        observed_at=obs,
                        last_checkpoint_at=obs,
                    )
                )

        if args.out:
            with open(args.out, "w", newline="") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "iter",
                        "sandboxes",
                        "checkpoints",
                        "restores",
                        "checkpoint_batch_ms",
                        "restore_batch_ms",
                        "success_ratio",
                    ],
                )
                w.writeheader()
                for row in rows:
                    w.writerow(row)

        avg_ckpt = sum(r["checkpoint_batch_ms"] for r in rows) / len(rows)
        avg_restore = sum(r["restore_batch_ms"] for r in rows) / len(rows)
        avg_success = sum(r["success_ratio"] for r in rows) / len(rows)
        print(f"checkpoint_batch_ms_avg: {avg_ckpt:.3f}")
        print(f"restore_batch_ms_avg:    {avg_restore:.3f}")
        print(f"restore_success_ratio_avg: {avg_success:.3f}")

        system.executor.shutdown()


if __name__ == "__main__":
    main()

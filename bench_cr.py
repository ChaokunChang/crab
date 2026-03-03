#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

def sh(cmd, check=True, capture=True, text=True, sudo=False, cwd=None):
    if sudo and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    try:
        p = subprocess.run(
            cmd,
            check=check,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=text,
            cwd=cwd,
        )
        return p
    except UnicodeDecodeError:
        # Retry in binary mode if decoding fails
        p = subprocess.run(
            cmd,
            check=check,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            text=False,
            cwd=cwd,
        )
        return p
    except subprocess.CalledProcessError as e:
        # Print useful diagnostics
        print("\n[command failed]")
        print("cmd:", " ".join(e.cmd) if isinstance(e.cmd, list) else e.cmd)
        print("returncode:", e.returncode)
        if e.stdout:
            print("\n--- stdout ---\n", e.stdout)
        if e.stderr:
            print("\n--- stderr ---\n", e.stderr)
        raise

def now_ns():
    return time.perf_counter_ns()

def docker_inspect_counter(workdir):
    state = Path(workdir) / "state.json"
    if not state.exists():
        return None
    try:
        return json.loads(state.read_text()).get("counter")
    except Exception:
        return None

def dir_size_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

def ensure_clean(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def benchmark_docker_checkpoint(args, out_rows):
    """
    Uses:
      docker run --name ...
      docker checkpoint create <ctr> <ckpt>
      docker start --checkpoint <ckpt> <ctr>
    """
    method = "docker-checkpoint"
    for i in range(args.iters):
        ctr = f"{args.name}-docker-{i}"
        ckpt = f"ckpt{i}"
        workdir = Path(args.work_root) / f"work-docker-{i}"
        ensure_clean(workdir)

        # run container
        sh([
            "docker", "run", "-d", "--rm=false", "--name", ctr, 
            "--network", "host", 
            "--security-opt", "seccomp=unconfined",
            "--security-opt", "apparmor=unconfined",
            "--cap-add=SYS_PTRACE",
            "--cap-add=CHECKPOINT_RESTORE",
            "-e", f"MEM_MB={args.mem_mb}",
            "-v", f"{workdir.absolute()}:/work",
            args.image
        ])

        time.sleep(args.warmup_s)

        c_before = docker_inspect_counter(workdir)

        # checkpoint
        t0 = now_ns()
        sh(["docker", "checkpoint", "create", "--leave-running=false", ctr, ckpt])
        t1 = now_ns()

        # checkpoint files location (Docker-managed)
        # typically: /var/lib/docker/containers/<id>/checkpoints/<ckpt>/
        # we can locate via docker inspect
        insp = sh(["docker", "inspect", ctr]).stdout
        j = json.loads(insp)[0]
        cid = j["Id"]
        ckpt_path = Path(args.docker_root) / "containers" / cid / "checkpoints" / ckpt
        ckpt_size = dir_size_bytes(ckpt_path) if ckpt_path.exists() else None

        # restore
        t2 = now_ns()
        sh(["docker", "start", "--checkpoint", ckpt, ctr])
        t3 = now_ns()

        time.sleep(args.post_restore_s)
        c_after = docker_inspect_counter(workdir)

        # cleanup
        sh(["docker", "rm", "-f", ctr], check=False)

        out_rows.append({
            "iter": i,
            "method": method,
            "checkpoint_ms": (t1 - t0) / 1e6,
            "restore_ms": (t3 - t2) / 1e6,
            "ckpt_size_bytes": ckpt_size,
            "counter_before": c_before,
            "counter_after": c_after,
            "counter_continues": (c_before is not None and c_after is not None and c_after > c_before),
        })

def benchmark_runc_criu(args, out_rows):
    """
    Runs a container with runc, then uses CRIU via runc checkpoint/restore:
      runc run -d <id>
      runc checkpoint <id> --image-path <dir> --work-path <dir>
      runc restore -d <id> --image-path <dir> --work-path <dir>
    Notes:
      - requires root (sudo).
      - uses an OCI bundle generated from Docker image via `docker export`.
    """
    method = "runc-criu"

    base = Path(args.work_root) / "runc-bundle"
    ensure_clean(base)

    # Create OCI bundle rootfs from docker image
    # 1) create temp container, export filesystem
    tmp_ctr = f"{args.name}-tmp-export"
    sh(["docker", "rm", "-f", tmp_ctr], check=False)
    sh(["docker", "create", 
        "--rm=false", 
        "--name", tmp_ctr, 
        "--network", "host", 
        "--security-opt", "seccomp=unconfined",
        "--security-opt", "apparmor=unconfined",
        "--cap-add=SYS_PTRACE",
        "--cap-add=CHECKPOINT_RESTORE",
        args.image])
    rootfs = base / "rootfs"
    ensure_clean(rootfs)
    # export tar -> rootfs
    # stream extract with tar (avoid huge memory); use subprocess piping
    p1 = subprocess.Popen(["docker", "export", tmp_ctr], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["tar", "-C", str(rootfs), "-xf", "-"], stdin=p1.stdout)
    p1.stdout.close()
    rc2 = p2.wait()
    rc1 = p1.wait()
    if rc1 != 0:
        raise RuntimeError("docker export failed")
    if rc2 != 0:
        raise RuntimeError("tar extraction failed")
    sh(["docker", "rm", "-f", tmp_ctr], check=False)

    # Create config.json using runc spec, then edit command/env/mounts
    # runc spec writes to ./config.json in cwd.
    # Run without sudo so config remains editable by this process.
    sh(["runc", "spec"], check=True, capture=False, cwd=base)

    config_path = base / "config.json"
    cfg = json.loads(config_path.read_text())

    # Set process args for workload
    cfg["process"]["args"] = ["python", "/app/agent_workload.py"]
    cfg["process"]["env"] = cfg["process"].get("env", [])
    # ensure OUT_DIR and MEM_MB
    cfg["process"]["env"] = [e for e in cfg["process"]["env"] if not e.startswith("OUT_DIR=") and not e.startswith("MEM_MB=")]
    cfg["process"]["env"] += [f"OUT_DIR=/work", f"MEM_MB={args.mem_mb}"]

    # Bind mount a host workdir into /work
    # We'll create per-iter workdir and update mount source each time.
    # Keep a placeholder mount entry we can edit.
    mounts = cfg.get("mounts", [])
    mounts = [m for m in mounts if m.get("destination") != "/work"]
    mounts.append({
        "destination": "/work",
        "type": "bind",
        "source": "/REPLACE_ME_WORKDIR",
        "options": ["rbind", "rw"]
    })
    cfg["mounts"] = mounts

    # Ensure python exists in rootfs (it does, from image)
    config_path.write_text(json.dumps(cfg, indent=2))

    for i in range(args.iters):
        cid = f"{args.name}-runc-{i}"
        workdir = Path(args.work_root) / f"work-runc-{i}"
        ensure_clean(workdir)

        # patch config.json mount source for this iter
        cfg_i = json.loads(config_path.read_text())
        for m in cfg_i["mounts"]:
            if m.get("destination") == "/work":
                m["source"] = str(workdir.absolute())
        (base / "config.json").write_text(json.dumps(cfg_i, indent=2))

        # run detached
        sh(["runc", "run", "-d", cid], sudo=True, check=True, capture=True, cwd=base)
        time.sleep(args.warmup_s)

        c_before = docker_inspect_counter(workdir)

        ckpt_dir = Path(args.work_root) / f"ckpt-runc-{i}"
        work_path = Path(args.work_root) / f"ckpt-work-{i}"
        ensure_clean(ckpt_dir)
        ensure_clean(work_path)

        # checkpoint
        t0 = now_ns()
        sh([
            "runc", "checkpoint", cid,
            "--image-path", str(ckpt_dir),
            "--work-path", str(work_path),
        ], sudo=True, check=True, capture=True, cwd=base)
        t1 = now_ns()
        ckpt_size = dir_size_bytes(ckpt_dir)

        # restore (same id)
        t2 = now_ns()
        sh([
            "runc", "restore", "-d", cid,
            "--image-path", str(ckpt_dir),
            "--work-path", str(work_path),
        ], sudo=True, check=True, capture=True, cwd=base)
        t3 = now_ns()

        time.sleep(args.post_restore_s)
        c_after = docker_inspect_counter(workdir)

        # cleanup
        sh(["runc", "delete", "-f", cid], sudo=True, check=False, capture=True, cwd=base)

        out_rows.append({
            "iter": i,
            "method": method,
            "checkpoint_ms": (t1 - t0) / 1e6,
            "restore_ms": (t3 - t2) / 1e6,
            "ckpt_size_bytes": ckpt_size,
            "counter_before": c_before,
            "counter_after": c_after,
            "counter_continues": (c_before is not None and c_after is not None and c_after > c_before),
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="agent-sandbox-bench:latest")
    ap.add_argument("--name", default="asb")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--mem-mb", type=int, default=128)
    ap.add_argument("--warmup-s", type=float, default=2.0)
    ap.add_argument("--post-restore-s", type=float, default=1.0)
    ap.add_argument("--out", default="results.csv")

    # adjust if your docker root differs
    ap.add_argument("--docker-root", default="/var/lib/docker")

    # where we store host-side work dirs / runc bundle / ckpts
    ap.add_argument("--work-root", default="./bench_out")

    ap.add_argument("--run-docker", action="store_true")
    ap.add_argument("--run-runc", action="store_true")

    args = ap.parse_args()
    Path(args.work_root).mkdir(parents=True, exist_ok=True)

    rows = []

    if not args.run_docker and not args.run_runc:
        # default: run both
        args.run_docker = True
        args.run_runc = True

    if args.run_docker:
        benchmark_docker_checkpoint(args, rows)

    if args.run_runc:
        # runc+criu almost always needs root; keep it explicit in README usage.
        benchmark_runc_criu(args, rows)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "iter","method","checkpoint_ms","restore_ms","ckpt_size_bytes",
            "counter_before","counter_after","counter_continues"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # print a tiny summary
    def avg(key, method):
        xs = [float(r[key]) for r in rows if r["method"] == method and r[key] is not None]
        return sum(xs)/len(xs) if xs else None

    for m in sorted(set(r["method"] for r in rows)):
        print(f"\n== {m} ==")
        print(f"checkpoint_ms_avg: {avg('checkpoint_ms', m)}")
        print(f"restore_ms_avg:    {avg('restore_ms', m)}")

    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()

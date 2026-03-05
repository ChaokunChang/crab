#!/usr/bin/env python3
import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


def sh(cmd, check=True, capture=True, text=True, sudo=False, cwd=None, timeout=None, null_stdio=False):
    if sudo and os.geteuid() != 0:
        cmd = ["sudo"] + cmd
    if null_stdio:
        # For detached workloads, avoid inheriting caller TTY fds.
        stdin = subprocess.DEVNULL
        stdout = subprocess.PIPE if capture else subprocess.DEVNULL
        stderr = subprocess.PIPE if capture else subprocess.DEVNULL
    else:
        stdin = None
        stdout = subprocess.PIPE if capture else None
        stderr = subprocess.PIPE if capture else None
    try:
        p = subprocess.run(
            cmd,
            check=check,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            text=text,
            cwd=cwd,
            timeout=timeout,
        )
        return p
    except UnicodeDecodeError:
        # Retry in binary mode if decoding fails
        print("decoding fails, retry in binary mode...")
        p = subprocess.run(
            cmd,
            check=check,
            stdin=stdin,
            stdout=stdout,
            stderr=stderr,
            text=False,
            cwd=cwd,
            timeout=timeout,
        )
        return p
    except subprocess.TimeoutExpired as e:
        print("\n[command timed out]")
        print("cmd:", " ".join(e.cmd) if isinstance(e.cmd, list) else e.cmd)
        print("timeout_s:", e.timeout)
        if e.stdout:
            print("\n--- stdout ---\n", e.stdout)
        if e.stderr:
            print("\n--- stderr ---\n", e.stderr)
        raise
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


def ensure_removed(path: Path):
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def bridge_checkpoint_to_daemon(src_ckpt_dir: Path, dst_ckpt_dir: Path, mode: str):
    if not src_ckpt_dir.exists():
        raise RuntimeError(
            f"source checkpoint dir does not exist: {src_ckpt_dir}")
    dst_ckpt_dir.parent.mkdir(parents=True, exist_ok=True)
    ensure_removed(dst_ckpt_dir)
    if mode == "copy":
        shutil.copytree(src_ckpt_dir, dst_ckpt_dir)
        return
    if mode == "symlink":
        os.symlink(src_ckpt_dir, dst_ckpt_dir, target_is_directory=True)
        return
    if mode == "hardlink":
        shutil.copytree(src_ckpt_dir, dst_ckpt_dir, copy_function=os.link)
        return
    raise ValueError(f"unknown checkpoint bridge mode: {mode}")


def pick_python_in_rootfs(rootfs: Path) -> str:
    candidates = [
        "usr/bin/python3",
        "usr/local/bin/python3",
        "bin/python3",
        "usr/bin/python",
        "usr/local/bin/python",
        "bin/python",
    ]
    for rel in candidates:
        if (rootfs / rel).exists():
            return "/" + rel
    # Best-effort fallback if image layout is unusual.
    return "python3"


def benchmark_docker_checkpoint(args, out_rows):
    """
    Uses:
      docker run --name ...
      docker checkpoint create <ctr> <ckpt>
      docker start --checkpoint <ckpt> <ctr>
    """
    method = "docker-checkpoint"
    run_id = int(time.time() * 1000)
    use_custom_checkpoint_dir = args.use_custom_checkpoint_dir
    for i in range(args.iters):
        while True:
            ctr = f"{args.name}-docker-{run_id}-{i}"
            ckpt = f"ckpt-docker-{run_id}-{i}"
            workdir = Path(args.work_root) / f"work-docker-{i}"
            ckpt_dir = (Path(args.work_root) / ckpt).absolute()
            ensure_clean(workdir)
            if use_custom_checkpoint_dir:
                ensure_clean(ckpt_dir)
            else:
                # If we've fallen back to daemon-managed checkpoints, clean any stale host-side dir.
                if ckpt_dir.exists():
                    shutil.rmtree(ckpt_dir)

            # Make reruns robust if prior execution crashed mid-iteration.
            sh(["docker", "rm", "-f", ctr], check=False)

            fallback_to_daemon_ckpt_dir = False

            try:
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
                checkpoint_cmd = [
                    "docker", "checkpoint", "create",
                    "--leave-running=false",
                ]
                if use_custom_checkpoint_dir:
                    checkpoint_cmd += ["--checkpoint-dir", str(ckpt_dir)]
                checkpoint_cmd += [ctr, ckpt]
                p_ckpt = sh(checkpoint_cmd, check=False)
                t1 = now_ns()
                if p_ckpt.returncode != 0:
                    stderr = p_ckpt.stderr or ""
                    if use_custom_checkpoint_dir and "custom checkpointdir is not supported" in stderr:
                        fallback_to_daemon_ckpt_dir = True
                    else:
                        print("\n[command failed]")
                        print("cmd:", " ".join(checkpoint_cmd))
                        print("returncode:", p_ckpt.returncode)
                        if p_ckpt.stdout:
                            print("\n--- stdout ---\n", p_ckpt.stdout)
                        if p_ckpt.stderr:
                            print("\n--- stderr ---\n", p_ckpt.stderr)
                        raise RuntimeError(
                            f"docker checkpoint failed for {ctr} (checkpoint {ckpt})")

                if fallback_to_daemon_ckpt_dir:
                    continue

                # Docker-managed checkpoint path (daemon storage)
                insp = sh(["docker", "inspect", ctr]).stdout
                j = json.loads(insp)[0]
                cid = j["Id"]
                # If we pass --checkpoint-dir, Docker stores files there; otherwise use daemon storage.
                ckpt_path_custom = Path(ckpt_dir) / ckpt
                ckpt_path_daemon = Path(
                    args.docker_root) / "containers" / cid / "checkpoints" / ckpt
                restore_with_custom_ckpt_dir = use_custom_checkpoint_dir

                # Optional bridge mode: write checkpoint to custom dir, then restore via daemon path.
                if (
                    use_custom_checkpoint_dir
                    and args.docker_custom_ckpt_restore == "daemon"
                ):
                    try:
                        bridge_checkpoint_to_daemon(
                            ckpt_path_custom,
                            ckpt_path_daemon,
                            args.docker_custom_ckpt_bridge,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            "failed to bridge custom checkpoint to daemon-managed dir "
                            f"({args.docker_custom_ckpt_bridge}): {e}"
                        ) from e
                    restore_with_custom_ckpt_dir = False

                if ckpt_path_custom.exists():
                    ckpt_size = dir_size_bytes(ckpt_path_custom)
                elif ckpt_path_daemon.exists():
                    ckpt_size = dir_size_bytes(ckpt_path_daemon)
                else:
                    ckpt_size = None

                # restore
                t2 = now_ns()
                restore_retries = 0
                restore_ms_success_only = None
                for attempt in range(args.docker_start_retries + 1):
                    t_attempt0 = now_ns()
                    start_cmd = [
                        "docker", "start",
                        "--checkpoint", ckpt,
                    ]
                    if restore_with_custom_ckpt_dir:
                        start_cmd += ["--checkpoint-dir", str(ckpt_dir)]
                    start_cmd += [ctr]
                    p = sh(start_cmd, check=False)
                    t_attempt1 = now_ns()
                    if p.returncode == 0:
                        restore_retries = attempt
                        restore_ms_success_only = (
                            t_attempt1 - t_attempt0) / 1e6
                        break
                    stderr = p.stderr or ""
                    if restore_with_custom_ckpt_dir and "custom checkpointdir is not supported" in stderr:
                        if args.docker_custom_ckpt_restore == "custom":
                            print("\n[command failed]")
                            print("cmd:", " ".join(start_cmd))
                            print("returncode:", p.returncode)
                            if p.stdout:
                                print("\n--- stdout ---\n", p.stdout)
                            if p.stderr:
                                print("\n--- stderr ---\n", p.stderr)
                            raise RuntimeError(
                                "docker daemon/runtime does not support restoring from --checkpoint-dir "
                                "(set --docker-custom-ckpt-restore auto/daemon to bridge into daemon-managed storage)"
                            )
                        if args.docker_custom_ckpt_restore == "auto":
                            try:
                                bridge_checkpoint_to_daemon(
                                    ckpt_path_custom,
                                    ckpt_path_daemon,
                                    args.docker_custom_ckpt_bridge,
                                )
                            except Exception as e:
                                raise RuntimeError(
                                    "docker restore with --checkpoint-dir is unsupported and bridge failed: "
                                    f"{e}"
                                ) from e
                            restore_with_custom_ckpt_dir = False
                            print(
                                "[docker] custom checkpoint-dir restore is unsupported; "
                                f"bridged checkpoint to daemon dir via {args.docker_custom_ckpt_bridge}",
                                flush=True,
                            )
                            continue
                        fallback_to_daemon_ckpt_dir = True
                        break
                    transient = (
                        "failed to upload checkpoint to containerd" in stderr
                        and "already exists" in stderr
                    )
                    if transient and attempt < args.docker_start_retries:
                        wait_s = args.docker_retry_delay_s * (attempt + 1)
                        print(
                            f"[docker restore retry {attempt + 1}/{args.docker_start_retries}] "
                            f"transient containerd content conflict, retrying in {wait_s:.1f}s",
                            flush=True,
                        )
                        time.sleep(wait_s)
                        continue
                    print("\n[command failed]")
                    print("cmd:", " ".join(start_cmd))
                    print("returncode:", p.returncode)
                    if p.stdout:
                        print("\n--- stdout ---\n", p.stdout)
                    if p.stderr:
                        print("\n--- stderr ---\n", p.stderr)
                    raise RuntimeError(
                        f"docker restore failed for {ctr} (checkpoint {ckpt})")
                t3 = now_ns()
                if fallback_to_daemon_ckpt_dir:
                    continue
                if restore_ms_success_only is None:
                    raise RuntimeError(
                        f"docker restore succeeded but no success timing recorded for {ctr}")

                time.sleep(args.post_restore_s)
                c_after = docker_inspect_counter(workdir)

                out_rows.append({
                    "iter": i,
                    "method": method,
                    "checkpoint_ms": (t1 - t0) / 1e6,
                    "restore_ms": (t3 - t2) / 1e6,
                    "restore_ms_success_only": restore_ms_success_only,
                    "restore_retries": restore_retries,
                    "ckpt_size_bytes": ckpt_size,
                    "counter_before": c_before,
                    "counter_after": c_after,
                    "counter_continues": (c_before is not None and c_after is not None and c_after > c_before),
                })
            finally:
                # cleanup
                sh(["docker", "rm", "-f", ctr], check=False)

            if fallback_to_daemon_ckpt_dir and use_custom_checkpoint_dir:
                use_custom_checkpoint_dir = False
                print(
                    "[docker] custom checkpoint-dir is not supported by this daemon/runtime; "
                    "falling back to daemon-managed checkpoints",
                    flush=True,
                )
                continue

            break

        if args.docker_iter_settle_s > 0 and i < args.iters - 1:
            print(
                f"[docker settle] sleeping {args.docker_iter_settle_s:.1f}s before next iteration",
                flush=True,
            )
            time.sleep(args.docker_iter_settle_s)


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
    p1 = subprocess.Popen(["docker", "export", tmp_ctr],
                          stdout=subprocess.PIPE)
    p2 = subprocess.Popen(
        ["tar", "-C", str(rootfs), "-xf", "-"], stdin=p1.stdout)
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
    py_bin = pick_python_in_rootfs(rootfs)
    cfg["process"]["args"] = [py_bin, "/app/agent_workload.py"]
    # Detached runc run/restore requires no TTY.
    cfg["process"]["terminal"] = False
    cfg["process"]["env"] = cfg["process"].get("env", [])
    # ensure OUT_DIR and MEM_MB
    cfg["process"]["env"] = [e for e in cfg["process"]["env"]
                             if not e.startswith("OUT_DIR=") and not e.startswith("MEM_MB=")]
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
        print(f"[runc {i+1}/{args.iters}] preparing bundle", flush=True)
        cid = f"{args.name}-runc-{i}"
        workdir = Path(args.work_root) / f"work-runc-{i}"
        ensure_clean(workdir)

        # patch config.json mount source for this iter
        cfg_i = json.loads(config_path.read_text())
        for m in cfg_i["mounts"]:
            if m.get("destination") == "/work":
                m["source"] = str(workdir.absolute())
        (base / "config.json").write_text(json.dumps(cfg_i, indent=2))

        # Previous failed/aborted runs can leave the same ID behind.
        # Make reruns idempotent by deleting any stale container first.
        sh(
            ["runc", "delete", "-f", cid],
            sudo=True,
            check=False,
            capture=True,
            cwd=base,
            timeout=args.cmd_timeout_s,
        )

        # run detached
        print(f"[runc {i+1}/{args.iters}] run -d ({py_bin})", flush=True)
        sh(
            ["runc", "run", "-d", cid],
            sudo=True,
            check=True,
            # In detached mode, capturing can block in communicate()
            # if descendants keep inherited stdio open.
            capture=False,
            null_stdio=True,
            cwd=base,
            timeout=args.cmd_timeout_s,
        )
        time.sleep(args.warmup_s)
        state_j = json.loads(sh(
            ["runc", "state", cid],
            sudo=True,
            check=True,
            capture=True,
            cwd=base,
            timeout=args.cmd_timeout_s,
        ).stdout)
        if state_j.get("status") != "running":
            sh(["runc", "delete", "-f", cid], sudo=True,
               check=False, capture=True, cwd=base)
            raise RuntimeError(
                f"runc container {cid} is '{state_j.get('status')}' right after launch; "
                f"workload exited early (command was {py_bin} /app/agent_workload.py)."
            )

        c_before = docker_inspect_counter(workdir)

        ckpt_dir = (Path(args.work_root) / f"ckpt-runc-{i}").absolute()
        work_path = (Path(args.work_root) / f"ckpt-work-{i}").absolute()
        ensure_clean(ckpt_dir)
        ensure_clean(work_path)

        # checkpoint
        print(f"[runc {i+1}/{args.iters}] checkpoint", flush=True)
        t0 = now_ns()
        sh([
            "runc", "checkpoint", cid,
            "--image-path", str(ckpt_dir),
            "--work-path", str(work_path),
        ], sudo=True, check=True, capture=True, cwd=base, timeout=args.cmd_timeout_s)
        t1 = now_ns()
        ckpt_size = dir_size_bytes(ckpt_dir)

        # After checkpoint, container metadata with this ID still exists
        # (usually as "stopped"). Remove it before restore.
        sh(
            ["runc", "delete", "-f", cid],
            sudo=True,
            check=False,
            capture=True,
            cwd=base,
            timeout=args.cmd_timeout_s,
        )

        # restore (same id)
        print(f"[runc {i+1}/{args.iters}] restore", flush=True)
        t2 = now_ns()
        sh([
            "runc", "restore", "-d",
            "--image-path", str(ckpt_dir),
            "--work-path", str(work_path),
            cid,
        ], sudo=True, check=True, capture=True, null_stdio=True, cwd=base, timeout=args.cmd_timeout_s)
        t3 = now_ns()

        time.sleep(args.post_restore_s)
        c_after = docker_inspect_counter(workdir)

        # cleanup
        sh(["runc", "delete", "-f", cid], sudo=True,
           check=False, capture=True, cwd=base)

        out_rows.append({
            "iter": i,
            "method": method,
            "checkpoint_ms": (t1 - t0) / 1e6,
            "restore_ms": (t3 - t2) / 1e6,
            "restore_ms_success_only": (t3 - t2) / 1e6,
            "restore_retries": 0,
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
    ap.add_argument("--cmd-timeout-s", type=float, default=120.0)
    ap.add_argument("--docker-start-retries", type=int, default=2)
    ap.add_argument("--docker-retry-delay-s", type=float, default=1.0)
    ap.add_argument("--docker-iter-settle-s", type=float, default=0.0)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--use-custom-checkpoint-dir", action="store_true")
    ap.add_argument(
        "--docker-custom-ckpt-restore",
        choices=["auto", "daemon", "custom"],
        default="auto",
        help=(
            "When checkpoint is created in --checkpoint-dir: "
            "'custom' restores from that dir directly; "
            "'daemon' bridges to Docker daemon checkpoint dir then restores without --checkpoint-dir; "
            "'auto' tries custom restore first, then bridges if unsupported."
        ),
    )
    ap.add_argument(
        "--docker-custom-ckpt-bridge",
        choices=["copy", "symlink", "hardlink"],
        default="copy",
        help="How to place custom checkpoint under daemon-managed dir for restore.",
    )

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
            "iter", "method", "checkpoint_ms", "restore_ms", "restore_ms_success_only", "restore_retries", "ckpt_size_bytes",
            "counter_before", "counter_after", "counter_continues"
        ])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # print a tiny summary
    def avg(key, method):
        xs = [float(r[key]) for r in rows if r["method"]
              == method and r[key] is not None]
        return sum(xs)/len(xs) if xs else None

    for m in sorted(set(r["method"] for r in rows)):
        print(f"\n== {m} ==")
        print(f"checkpoint_ms_avg: {avg('checkpoint_ms', m)}")
        print(f"restore_ms_avg:    {avg('restore_ms', m)}")
        print(
            f"restore_ms_success_only_avg: {avg('restore_ms_success_only', m)}")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

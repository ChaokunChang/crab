#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# ----------------------------
# Helpers
# ----------------------------

ALL_CAPS = [
    "CAP_AUDIT_CONTROL",
    "CAP_AUDIT_READ",
    "CAP_AUDIT_WRITE",
    "CAP_BLOCK_SUSPEND",
    "CAP_BPF",
    "CAP_CHECKPOINT_RESTORE",
    "CAP_CHOWN",
    "CAP_DAC_OVERRIDE",
    "CAP_DAC_READ_SEARCH",
    "CAP_FOWNER",
    "CAP_FSETID",
    "CAP_IPC_LOCK",
    "CAP_IPC_OWNER",
    "CAP_KILL",
    "CAP_LEASE",
    "CAP_LINUX_IMMUTABLE",
    "CAP_MAC_ADMIN",
    "CAP_MAC_OVERRIDE",
    "CAP_MKNOD",
    "CAP_NET_ADMIN",
    "CAP_NET_BIND_SERVICE",
    "CAP_NET_BROADCAST",
    "CAP_NET_RAW",
    "CAP_PERFMON",
    "CAP_SETFCAP",
    "CAP_SETGID",
    "CAP_SETPCAP",
    "CAP_SETUID",
    "CAP_SYSLOG",
    "CAP_SYS_ADMIN",
    "CAP_SYS_BOOT",
    "CAP_SYS_CHROOT",
    "CAP_SYS_MODULE",
    "CAP_SYS_NICE",
    "CAP_SYS_PACCT",
    "CAP_SYS_PTRACE",
    "CAP_SYS_RAWIO",
    "CAP_SYS_RESOURCE",
    "CAP_SYS_TIME",
    "CAP_SYS_TTY_CONFIG",
    "CAP_WAKE_ALARM",
]
MIB = 1024 * 1024
ZERO_BLOCK = b"\0" * MIB

def sh(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    capture: bool = True,
    input_text: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    kwargs = {
        "cwd": str(cwd) if cwd else None,
        "text": True,
        "stdin": subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        "stdout": subprocess.PIPE if capture else None,
        "stderr": subprocess.PIPE if capture else None,
        "timeout": timeout,
    }
    try:
        p = subprocess.run(cmd, check=False, input=input_text, **kwargs)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(
            f"command timed out after {timeout}s: {' '.join(cmd)}\n"
            f"stdout:\n{(e.stdout or '') if isinstance(e.stdout, str) else ''}\n"
            f"stderr:\n{(e.stderr or '') if isinstance(e.stderr, str) else ''}"
        ) from e
    if check and p.returncode != 0:
        raise RuntimeError(
            f"command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{p.stdout or ''}\n"
            f"stderr:\n{p.stderr or ''}"
        )
    return p


def shell_pipe_export_to_tar(cid: str, rootfs: Path, docker_bin: str = "docker") -> None:
    p1 = subprocess.Popen([docker_bin, "export", cid], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(
        ["tar", "-xpf", "-", "-C", str(rootfs)],
        stdin=p1.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    assert p1.stdout is not None
    p1.stdout.close()
    _, err2 = p2.communicate()
    rc1 = p1.wait()
    if rc1 != 0 or p2.returncode != 0:
        raise RuntimeError(
            f"docker export/tar failed rc1={rc1} rc2={p2.returncode} stderr={err2.decode(errors='ignore')}"
        )


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def mkdirp(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")


# ----------------------------
# Data model
# ----------------------------

@dataclass
class BenchConfig:
    base_dir: Path
    state_root: Path
    pool_name: str
    file_vdev: Path | None
    device: str | None
    pool_size: str
    image: str
    containers: int
    iterations: int
    workload: str
    custom_cmd: str | None
    files_per_iter: int
    snapshot_prefix: str
    keep: bool
    docker_bin: str
    runc_bin: str
    zpool_bin: str
    zfs_bin: str
    max_workers: int
    pause_before_snapshot: bool
    snapshot_while_writing: bool
    writer_containers: int
    writer_mode: str
    live_writer_cmd: str | None
    writer_chunk_mb: int
    writer_files_per_loop: int
    writer_slots: int
    writer_warmup_ms: int
    restore_ratio: float
    restore_workers: int
    host_writer_count: int
    host_writer_mode: str
    host_writer_dir: Path | None
    host_writer_total_mb: int
    host_writer_files_per_loop: int
    host_writer_slots: int


@dataclass(frozen=True)
class CommandSpec:
    command: str
    profile: str
    estimated_write_bytes: int | None = None


@dataclass
class ContainerSpec:
    idx: int
    name: str
    dataset: str
    bundle_dir: Path
    rootfs_dir: Path


# ----------------------------
# Benchmark class
# ----------------------------

class ZfsRuncBenchmark:
    def __init__(self, cfg: BenchConfig):
        self.cfg = cfg
        self.results: list[dict] = []
        self._container_profiles: dict[str, str] = {}
        self._host_writer_root_cache: Path | None = None
        self._lock = threading.Lock()

    def record(self, **kwargs) -> None:
        container_name = kwargs.get("container")
        if container_name is not None and "container_profile" not in kwargs:
            kwargs["container_profile"] = self._container_profiles.get(str(container_name))
        with self._lock:
            self.results.append(kwargs)

    def run(self) -> None:
        self._prepare_dirs()
        self._create_pool()
        effective_workers = min(self.cfg.max_workers, self.cfg.containers)
        if self.cfg.containers < 2 and self.cfg.max_workers > 1:
            print(
                "warning: --max-workers only caps thread fan-out; use --containers > 1 "
                "to measure concurrent snapshots across multiple containers",
                file=sys.stderr,
            )
        elif effective_workers < self.cfg.max_workers:
            print(
                f"note: effective worker count is {effective_workers} because only "
                f"{self.cfg.containers} containers were requested",
                file=sys.stderr,
            )
        containers: list[ContainerSpec] = []
        try:
            containers = self._prepare_containers()
            self._concurrent_initial_snapshot(containers)

            for it in range(1, self.cfg.iterations + 1):
                if (
                    self.cfg.snapshot_while_writing
                    or self.cfg.restore_ratio > 0.0
                    or self.cfg.host_writer_count > 0
                ):
                    self._run_mixed_round(containers, it)
                else:
                    self._run_workloads(containers, it)
                    self._concurrent_snapshot_round(containers, it)

            self._write_results()
        finally:
            if not self.cfg.keep:
                self._cleanup_containers(containers)
                self._cleanup_host_writer_root()
                self._destroy_pool()

    # -------- storage --------

    def _prepare_dirs(self) -> None:
        mkdirp(self.cfg.base_dir)
        mkdirp(self.cfg.state_root)

    def _cleanup_host_writer_root(self) -> None:
        root = self._host_writer_root_cache
        if root is None:
            return
        if root.exists():
            shutil.rmtree(root, ignore_errors=True)

    def _host_writer_root(self) -> Path:
        if self._host_writer_root_cache is not None:
            return self._host_writer_root_cache
        if self.cfg.host_writer_dir is not None:
            root = self.cfg.host_writer_dir
        elif self.cfg.file_vdev is not None:
            root = self.cfg.file_vdev.parent / f"{self.cfg.pool_name}.host-io"
        else:
            root = self.cfg.base_dir / "host-io"
        mkdirp(root)
        self._host_writer_root_cache = root
        return root

    def _create_pool(self) -> None:
        try:
            sh([self.cfg.zpool_bin, "destroy", "-f", self.cfg.pool_name], check=False)
        except Exception:
            pass

        if self.cfg.file_vdev:
            mkdirp(self.cfg.file_vdev.parent)
            sh(["truncate", "-s", self.cfg.pool_size, str(self.cfg.file_vdev)])
            sh([self.cfg.zpool_bin, "create", "-f", self.cfg.pool_name, str(self.cfg.file_vdev)])
        elif self.cfg.device:
            sh([self.cfg.zpool_bin, "create", "-f", self.cfg.pool_name, self.cfg.device])
        else:
            raise ValueError("must provide either --file-vdev or --device")

        sh([self.cfg.zfs_bin, "create", "-o", "mountpoint=none", f"{self.cfg.pool_name}/containers"])

    def _destroy_pool(self) -> None:
        sh([self.cfg.zpool_bin, "destroy", "-f", self.cfg.pool_name], check=False)
        if self.cfg.file_vdev and self.cfg.file_vdev.exists():
            self.cfg.file_vdev.unlink(missing_ok=True)

    # -------- container prep --------

    def _prepare_containers(self) -> list[ContainerSpec]:
        out: list[ContainerSpec] = []
        for i in range(self.cfg.containers):
            name = f"bench-{i}-{uuid.uuid4().hex[:8]}"
            bundle_dir = self.cfg.base_dir / "bundles" / name
            rootfs_dir = bundle_dir / "rootfs"
            dataset = f"{self.cfg.pool_name}/containers/{name}"
            mkdirp(bundle_dir)
            sh([self.cfg.zfs_bin, "create", "-o", f"mountpoint={rootfs_dir}", dataset])

            self._materialize_rootfs(self.cfg.image, rootfs_dir)
            self._make_runc_bundle(bundle_dir, name)
            self._runc_create_start(name, bundle_dir)

            out.append(
                ContainerSpec(
                    idx=i,
                    name=name,
                    dataset=dataset,
                    bundle_dir=bundle_dir,
                    rootfs_dir=rootfs_dir,
                )
            )
            self._container_profiles[name] = self._benchmark_profile_name(out[-1])
        return out

    def _workload_needs_network(self) -> bool:
        return self.cfg.workload in {"apt", "pip"}

    def _sync_host_network_files(self, rootfs_dir: Path) -> None:
        etc_dir = rootfs_dir / "etc"
        mkdirp(etc_dir)
        for name in ("resolv.conf", "hosts"):
            src = Path("/etc") / name
            dst = etc_dir / name
            if not src.exists():
                continue
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            shutil.copy2(src, dst)

    def _materialize_rootfs(self, image: str, rootfs_dir: Path) -> None:
        cid = sh([self.cfg.docker_bin, "create", image], capture=True).stdout.strip()
        try:
            shell_pipe_export_to_tar(cid, rootfs_dir, docker_bin=self.cfg.docker_bin)
        finally:
            sh([self.cfg.docker_bin, "rm", "-f", cid], check=False)
        if self._workload_needs_network():
            self._sync_host_network_files(rootfs_dir)

    def _make_runc_bundle(self, bundle_dir: Path, name: str) -> None:
        sh([self.cfg.runc_bin, "spec"], cwd=bundle_dir)
        cfg_path = bundle_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

        cfg["root"]["path"] = "rootfs"
        cfg["root"]["readonly"] = False
        cfg["hostname"] = name
        cfg["process"]["terminal"] = False
        cfg["process"]["cwd"] = "/"
        cfg["process"]["args"] = [
            "/bin/sh",
            "-lc",
            "trap 'exit 0' TERM INT; while :; do sleep 3600; done",
        ]

        env = cfg["process"].get("env", [])
        env = [x for x in env if not x.startswith("PATH=")]
        env.append("PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        if self.cfg.workload == "apt":
            env.append("DEBIAN_FRONTEND=noninteractive")
        cfg["process"]["env"] = env
        cfg["process"]["capabilities"] = {
            name: list(ALL_CAPS)
            for name in ("ambient", "bounding", "effective", "inheritable", "permitted")
        }
        cfg["process"]["noNewPrivileges"] = False
        if self._workload_needs_network():
            cfg["linux"]["namespaces"] = [
                ns for ns in cfg["linux"].get("namespaces", [])
                if ns.get("type") != "network"
            ]

        write_json(cfg_path, cfg)

    def _runc_create_start(self, name: str, bundle_dir: Path) -> None:
        # Do not capture stdio here: the stopped container init inherits these
        # file descriptors, which keeps subprocess.run(..., stdout/stderr=PIPE)
        # waiting forever for EOF even after `runc create` itself exits.
        sh(
            [self.cfg.runc_bin, "--root", str(self.cfg.state_root), "create", "--bundle", str(bundle_dir), name],
            capture=False,
            timeout=30,
        )
        try:
            sh(
                [self.cfg.runc_bin, "--root", str(self.cfg.state_root), "start", name],
                capture=False,
                timeout=30,
            )
        except Exception:
            sh([self.cfg.runc_bin, "--root", str(self.cfg.state_root), "delete", "-f", name], check=False)
            raise

        for _ in range(20):
            p = sh(
                [self.cfg.runc_bin, "--root", str(self.cfg.state_root), "exec", name, "/bin/sh", "-lc", "true"],
                check=False,
            )
            if p.returncode == 0:
                return
            time.sleep(0.2)
        raise RuntimeError(f"container did not become exec-ready: {name}")

    # -------- workloads --------

    def _build_workload_cmd(self, it: int) -> str:
        raise NotImplementedError("use _build_workload_spec(...)")

    def _benchmark_profile_name(self, c: ContainerSpec) -> str:
        profiles = ("tree", "logs", "build", "cache")
        return profiles[c.idx % len(profiles)]

    def _smallfile_command(self, *, base: str, file_count: int) -> CommandSpec:
        return CommandSpec(
            command=(
                "set -eu; "
                f"base={shlex.quote(base)}; "
                "mkdir -p \"$base\"; "
                "host=${HOSTNAME:-container}; "
                "i=0; "
                f"while [ \"$i\" -lt {file_count} ]; do "
                "d=\"$base/$((i % 64))\"; "
                "mkdir -p \"$d\"; "
                "printf '%s %s %s\\n' \"$i\" \"$host\" \"xxxxxxxxxxxxxxxxxxxxxxxx\" > \"$d/f_$i.txt\"; "
                "i=$((i + 1)); "
                "done; "
                "find \"$base\" -type f | head -n 128 | while read -r f; do echo tail >> \"$f\"; done"
            ),
            profile="smallfiles",
            estimated_write_bytes=max(1, file_count) * 128,
        )

    def _benchmark_like_spec(
        self,
        c: ContainerSpec,
        *,
        iteration: int,
        loop_index: int,
        file_count: int,
        live: bool,
    ) -> CommandSpec:
        profile = self._benchmark_profile_name(c)
        hot_count = max(8, min(256, max(1, file_count // 8)))
        temp_blob_kb = 64 if live else 256
        base = f"/var/tmp/bench/benchmark_like/{profile}"
        prefix = "live" if live else "iter"
        command_prefix = (
            "set -eu; "
            f"base={shlex.quote(base)}; "
            "host=${HOSTNAME:-container}; "
            f"iter={iteration}; "
            f"loop={loop_index}; "
            f"count={file_count}; "
            f"hot={hot_count}; "
            f"blob_kb={temp_blob_kb}; "
        )
        profile_commands = {
            "tree": (
                "mkdir -p \"$base/tree\" \"$base/rename\" \"$base/logs\" \"$base/tmp\"; "
                f"root=\"$base/tree/{prefix}_${{iter}}\"; "
                "mkdir -p \"$root\"; "
                "i=0; "
                "while [ \"$i\" -lt \"$count\" ]; do "
                "d=\"$root/pkg_$((i % 32))/mod_$((i % 16))\"; "
                "mkdir -p \"$d\"; "
                "printf 'host=%s iter=%s loop=%s i=%s\\n' \"$host\" \"$iter\" \"$loop\" \"$i\" > \"$d/f_$i.txt\"; "
                "i=$((i + 1)); "
                "done; "
                "i=0; "
                "while [ \"$i\" -lt \"$hot\" ]; do "
                "src=\"$root/pkg_$((i % 32))/mod_$((i % 16))/f_$i.txt\"; "
                "dst=\"$base/rename/r_${iter}_${loop}_$i.txt\"; "
                "if [ -f \"$src\" ]; then mv \"$src\" \"$dst\"; fi; "
                "i=$((i + 1)); "
                "done; "
                "find \"$base\" -type f | head -n 96 | while read -r f; do printf 'tail %s %s %s\\n' \"$host\" \"$iter\" \"$loop\" >> \"$f\"; done; "
                "i=0; "
                "while [ \"$i\" -lt \"$hot\" ]; do "
                "tmp=\"$base/tmp/t_${iter}_${loop}_$i.bin\"; "
                "dd if=/dev/zero of=\"$tmp\" bs=1024 count=\"$blob_kb\" conv=fsync status=none; "
                "rm -f \"$tmp\"; "
                "i=$((i + 1)); "
                "done"
            ),
            "logs": (
                "mkdir -p \"$base/current\" \"$base/archive\" \"$base/index\" \"$base/tmp\"; "
                "i=0; "
                "while [ \"$i\" -lt \"$count\" ]; do "
                "f=\"$base/current/log_$((i % 24)).txt\"; "
                "printf 'host=%s iter=%s loop=%s line=%s %s\\n' \"$host\" \"$iter\" \"$loop\" \"$i\" \"xxxxxxxxxxxxxxxxxxxxxxxx\" >> \"$f\"; "
                "i=$((i + 1)); "
                "done; "
                "i=0; "
                "while [ \"$i\" -lt \"$hot\" ]; do "
                "src=\"$base/current/log_$((i % 24)).txt\"; "
                "dst=\"$base/archive/log_${iter}_${loop}_$i.txt\"; "
                "if [ -f \"$src\" ]; then cp \"$src\" \"$dst\"; : > \"$src\"; fi; "
                "idx=\"$base/tmp/index_$i.tmp\"; "
                "printf 'iter=%s loop=%s file=%s\\n' \"$iter\" \"$loop\" \"$dst\" > \"$idx\"; "
                "mv \"$idx\" \"$base/index/manifest_$i.txt\"; "
                "i=$((i + 1)); "
                "done; "
                "find \"$base/current\" -type f | head -n 64 | while read -r f; do printf 'rotate-tail %s %s\\n' \"$iter\" \"$loop\" >> \"$f\"; done"
            ),
            "build": (
                "mkdir -p \"$base/src\" \"$base/obj\" \"$base/pkg\" \"$base/tmp\"; "
                "i=0; "
                "while [ \"$i\" -lt \"$count\" ]; do "
                "src=\"$base/src/unit_$i.c\"; "
                "obj_tmp=\"$base/tmp/unit_$i.o.tmp\"; "
                "obj=\"$base/obj/unit_$i.o\"; "
                "printf 'int unit_%s(void){return %s + %s;}\\n' \"$i\" \"$i\" \"$loop\" > \"$src\"; "
                "printf 'object host=%s iter=%s loop=%s unit=%s\\n' \"$host\" \"$iter\" \"$loop\" \"$i\" > \"$obj_tmp\"; "
                "mv \"$obj_tmp\" \"$obj\"; "
                "i=$((i + 1)); "
                "done; "
                "i=0; "
                "while [ \"$i\" -lt \"$hot\" ]; do "
                "blob_tmp=\"$base/tmp/blob_$i.bin.tmp\"; "
                "blob=\"$base/pkg/blob_${iter}_${loop}_$i.bin\"; "
                "dd if=/dev/zero of=\"$blob_tmp\" bs=1024 count=\"$blob_kb\" conv=fsync status=none; "
                "mv \"$blob_tmp\" \"$blob\"; "
                "i=$((i + 1)); "
                "done; "
                "find \"$base/obj\" -type f | head -n 64 | while read -r f; do printf 'relink %s %s\\n' \"$iter\" \"$loop\" >> \"$f\"; done"
            ),
            "cache": (
                "mkdir -p \"$base/index\" \"$base/objects\" \"$base/manifests\" \"$base/tmp\"; "
                "i=0; "
                "while [ \"$i\" -lt \"$count\" ]; do "
                "d=\"$base/objects/$((i % 64))/$((i % 32))\"; "
                "mkdir -p \"$d\"; "
                "obj=\"$d/cache_${iter}_${loop}_$i.dat\"; "
                "printf 'cache host=%s iter=%s loop=%s obj=%s\\n' \"$host\" \"$iter\" \"$loop\" \"$i\" > \"$obj\"; "
                "idx_tmp=\"$base/tmp/idx_$i.tmp\"; "
                "printf '%s %s %s %s\\n' \"$host\" \"$iter\" \"$loop\" \"$obj\" > \"$idx_tmp\"; "
                "mv \"$idx_tmp\" \"$base/index/entry_$i.txt\"; "
                "i=$((i + 1)); "
                "done; "
                "i=0; "
                "while [ \"$i\" -lt \"$hot\" ]; do "
                "manifest_tmp=\"$base/tmp/manifest_$i.tmp\"; "
                "manifest=\"$base/manifests/m_${iter}_${loop}_$i.json\"; "
                "printf '{\"iter\":%s,\"loop\":%s,\"slot\":%s}\\n' \"$iter\" \"$loop\" \"$i\" > \"$manifest_tmp\"; "
                "mv \"$manifest_tmp\" \"$manifest\"; "
                "old=\"$base/index/entry_$((i % 16)).txt\"; "
                "if [ -f \"$old\" ]; then rm -f \"$old\"; fi; "
                "i=$((i + 1)); "
                "done; "
                "find \"$base/manifests\" -type f | head -n 64 | while read -r f; do printf 'touch %s %s\\n' \"$iter\" \"$loop\" >> \"$f\"; done"
            ),
        }
        estimated_write_bytes = max(1, file_count) * 256 + hot_count * temp_blob_kb * 1024
        return CommandSpec(
            command=command_prefix + profile_commands[profile],
            profile=f"benchmark_{profile}",
            estimated_write_bytes=estimated_write_bytes,
        )

    def _build_workload_spec(self, c: ContainerSpec, it: int) -> CommandSpec:
        if self.cfg.custom_cmd:
            return CommandSpec(command=self.cfg.custom_cmd, profile="custom")

        if self.cfg.workload == "apt":
            return CommandSpec(
                command="apt-get -o APT::Sandbox::User=root update && apt-get -o APT::Sandbox::User=root install -y libbpf-dev",
                profile="apt",
            )

        if self.cfg.workload == "pip":
            return CommandSpec(
                command="python3 -m pip install --no-input --disable-pip-version-check requests",
                profile="pip",
            )

        if self.cfg.workload == "benchmark":
            return self._benchmark_like_spec(
                c,
                iteration=it,
                loop_index=0,
                file_count=max(1, self.cfg.files_per_iter),
                live=False,
            )

        return self._smallfile_command(
            base=f"/var/tmp/bench/iter_{it}",
            file_count=max(1, self.cfg.files_per_iter),
        )

    def _exec_in_container(self, name: str, cmd: str) -> subprocess.CompletedProcess[str]:
        return sh(
            [
                self.cfg.runc_bin,
                "--root",
                str(self.cfg.state_root),
                "exec",
                name,
                "/bin/sh",
                "-lc",
                cmd,
            ],
            capture=True,
        )

    def _run_workloads(self, containers: list[ContainerSpec], it: int) -> None:
        with ThreadPoolExecutor(max_workers=min(self.cfg.max_workers, len(containers))) as ex:
            futs = {
                ex.submit(self._run_one_workload, c, it): c
                for c in containers
            }
            for fut in as_completed(futs):
                fut.result()

    def _run_one_workload(self, c: ContainerSpec, it: int) -> None:
        spec = self._build_workload_spec(c, it)
        t0 = now_ms()
        p = self._exec_in_container(c.name, spec.command)
        t1 = now_ms()
        self.record(
            phase="workload",
            profile=spec.profile,
            container=c.name,
            dataset=c.dataset,
            iteration=it,
            duration_ms=round(t1 - t0, 3),
            estimated_write_bytes=spec.estimated_write_bytes,
            rc=p.returncode,
            stdout=(p.stdout or "")[:4000],
            stderr=(p.stderr or "")[:4000],
        )

    def _build_live_write_spec(self, c: ContainerSpec, iteration: int, *, loop_index: int) -> CommandSpec:
        if self.cfg.live_writer_cmd:
            return CommandSpec(command=self.cfg.live_writer_cmd, profile="live_custom")

        if self.cfg.writer_mode == "workload":
            return self._build_workload_spec(c, iteration)

        if self.cfg.writer_mode == "benchmark":
            return self._benchmark_like_spec(
                c,
                iteration=iteration,
                loop_index=loop_index,
                file_count=max(1, self.cfg.writer_files_per_loop),
                live=True,
            )

        if self.cfg.writer_mode == "smallfiles":
            files_per_loop = max(1, self.cfg.writer_files_per_loop)
            slots = max(1, self.cfg.writer_slots)
            slot = loop_index % slots
            return CommandSpec(
                command=(
                    "set -eu; "
                    f"base=/var/tmp/bench/live_iter_{iteration}/slot_{slot}; "
                    "mkdir -p \"$base\"; "
                    "host=${HOSTNAME:-container}; "
                    "i=0; "
                    f"while [ \"$i\" -lt {files_per_loop} ]; do "
                    "d=\"$base/$((i % 64))\"; "
                    "mkdir -p \"$d\"; "
                    "printf '%s %s %s\\n' \"$i\" \"$host\" \"xxxxxxxxxxxxxxxxxxxxxxxx\" > \"$d/f_$i.txt\"; "
                    "i=$((i + 1)); "
                    "done; "
                    "find \"$base\" -type f | head -n 128 | while read -r f; do echo tail >> \"$f\"; done"
                ),
                profile="smallfiles",
                estimated_write_bytes=files_per_loop * 128,
            )

        chunk_mb = max(1, self.cfg.writer_chunk_mb)
        slots = max(1, self.cfg.writer_slots)
        slot = loop_index % slots
        return CommandSpec(
            command=(
                "set -eu; "
                f"base=/var/tmp/bench/live_iter_{iteration}; "
                "mkdir -p \"$base\"; "
                f"dd if=/dev/zero of=\"$base/slot_{slot}.bin\" "
                f"bs=1M count={chunk_mb} conv=fsync status=none"
            ),
            profile="dd",
            estimated_write_bytes=chunk_mb * 1024 * 1024,
        )

    def _host_benchmark_profile_name(self, writer_index: int) -> str:
        profiles = ("checkpoint", "logs", "artifacts", "workspace")
        return profiles[writer_index % len(profiles)]

    def _reset_dir(self, path: Path) -> None:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

    def _write_binary_atomic(self, path: Path, total_bytes: int) -> None:
        mkdirp(path.parent)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        remaining = max(0, total_bytes)
        with tmp_path.open("wb", buffering=0) as handle:
            while remaining > 0:
                chunk_size = min(remaining, len(ZERO_BLOCK))
                handle.write(ZERO_BLOCK[:chunk_size])
                remaining -= chunk_size
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _write_text_atomic(self, path: Path, text: str) -> None:
        mkdirp(path.parent)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    def _append_text_sync(self, path: Path, text: str) -> None:
        mkdirp(path.parent)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def _split_total_bytes(self, total_bytes: int, part_count: int) -> list[int]:
        count = max(1, part_count)
        total = max(count, total_bytes)
        base = total // count
        remainder = total % count
        return [base + (1 if i < remainder else 0) for i in range(count)]

    def _run_host_process_like_write(
        self,
        *,
        root: Path,
        writer_index: int,
        iteration: int,
        loop_index: int,
        profile: str,
    ) -> tuple[str, int]:
        slot = loop_index % max(1, self.cfg.host_writer_slots)
        slot_dir = root / profile / f"writer_{writer_index}" / f"slot_{slot}"
        self._reset_dir(slot_dir)
        total_bytes = max(1, self.cfg.host_writer_total_mb) * MIB
        image_count = max(4, self.cfg.host_writer_files_per_loop)
        image_sizes = self._split_total_bytes(total_bytes, image_count)
        for idx, size in enumerate(image_sizes):
            self._write_binary_atomic(slot_dir / f"pages-{idx:05d}.img", size)
        self._write_text_atomic(
            slot_dir / "inventory.img",
            json.dumps(
                {
                    "iteration": iteration,
                    "loop_index": loop_index,
                    "writer_index": writer_index,
                    "image_count": image_count,
                    "total_bytes": total_bytes,
                },
                sort_keys=True,
            ),
        )
        self._write_text_atomic(
            slot_dir / "stats-dump",
            f"iter={iteration} loop={loop_index} writer={writer_index} total_bytes={total_bytes}\n",
        )
        self._write_text_atomic(
            slot_dir / "pstree.img",
            f"root-task-{writer_index}\nchild-{iteration}\nchild-{loop_index}\n",
        )
        return profile, total_bytes

    def _run_host_smallfiles_write(
        self,
        *,
        root: Path,
        writer_index: int,
        iteration: int,
        loop_index: int,
        profile: str,
    ) -> tuple[str, int]:
        slot = loop_index % max(1, self.cfg.host_writer_slots)
        slot_dir = root / profile / f"writer_{writer_index}" / f"slot_{slot}"
        self._reset_dir(slot_dir)
        file_count = max(1, self.cfg.host_writer_files_per_loop)
        estimated_bytes = 0
        for idx in range(file_count):
            leaf = slot_dir / f"d_{idx % 64}" / f"f_{idx:05d}.txt"
            payload = (
                f"writer={writer_index} iter={iteration} loop={loop_index} file={idx} "
                f"{'x' * 96}\n"
            )
            self._write_text_atomic(leaf, payload)
            estimated_bytes += len(payload.encode("utf-8"))
        for idx in range(min(96, file_count)):
            leaf = slot_dir / f"d_{idx % 64}" / f"f_{idx:05d}.txt"
            self._append_text_sync(leaf, f"tail iter={iteration} loop={loop_index} file={idx}\n")
            estimated_bytes += len(f"tail iter={iteration} loop={loop_index} file={idx}\n".encode("utf-8"))
        return profile, estimated_bytes

    def _run_host_benchmark_write(
        self,
        *,
        root: Path,
        writer_index: int,
        iteration: int,
        loop_index: int,
    ) -> tuple[str, int]:
        profile = self._host_benchmark_profile_name(writer_index)
        benchmark_root = root / "benchmark"
        if profile == "checkpoint":
            return self._run_host_process_like_write(
                root=benchmark_root,
                writer_index=writer_index,
                iteration=iteration,
                loop_index=loop_index,
                profile="host_benchmark_checkpoint",
            )

        slot = loop_index % max(1, self.cfg.host_writer_slots)
        slot_dir = benchmark_root / profile / f"writer_{writer_index}" / f"slot_{slot}"
        self._reset_dir(slot_dir)
        file_count = max(8, self.cfg.host_writer_files_per_loop)
        total_bytes = max(1, self.cfg.host_writer_total_mb) * MIB
        blob_bytes = max(MIB, total_bytes // max(1, file_count // 4))
        estimated_bytes = 0

        if profile == "logs":
            log_count = max(8, min(24, file_count // 2))
            for idx in range(log_count):
                log_path = slot_dir / "current" / f"log_{idx:03d}.txt"
                lines = [
                    f"writer={writer_index} iter={iteration} loop={loop_index} log={idx} line={line}\n"
                    for line in range(max(8, file_count // max(1, log_count)))
                ]
                payload = "".join(lines)
                self._append_text_sync(log_path, payload)
                estimated_bytes += len(payload.encode("utf-8"))
            for idx in range(min(log_count, 8)):
                archive_path = slot_dir / "archive" / f"log_{idx:03d}.bin"
                self._write_binary_atomic(archive_path, blob_bytes)
                estimated_bytes += blob_bytes
                self._write_text_atomic(
                    slot_dir / "index" / f"manifest_{idx:03d}.txt",
                    f"writer={writer_index} iter={iteration} loop={loop_index} archive={archive_path.name}\n",
                )
        elif profile == "artifacts":
            object_count = max(16, file_count)
            for idx in range(object_count):
                obj_path = slot_dir / "objects" / f"artifact_{idx:05d}.bin"
                size = max(4096, blob_bytes // 4)
                self._write_binary_atomic(obj_path, size)
                estimated_bytes += size
                self._write_text_atomic(
                    slot_dir / "manifests" / f"artifact_{idx:05d}.json",
                    json.dumps(
                        {
                            "writer": writer_index,
                            "iteration": iteration,
                            "loop": loop_index,
                            "artifact": obj_path.name,
                            "bytes": size,
                        },
                        sort_keys=True,
                    ),
                )
        else:
            object_count = max(16, file_count)
            for idx in range(object_count):
                src_path = slot_dir / "workspace" / "src" / f"unit_{idx:05d}.txt"
                obj_path = slot_dir / "workspace" / "obj" / f"unit_{idx:05d}.o"
                src_payload = f"writer={writer_index} iter={iteration} loop={loop_index} src={idx}\n"
                self._write_text_atomic(src_path, src_payload)
                estimated_bytes += len(src_payload.encode("utf-8"))
                size = max(8192, blob_bytes // 8)
                self._write_binary_atomic(obj_path, size)
                estimated_bytes += size
            self._write_text_atomic(
                slot_dir / "workspace" / "build.json",
                json.dumps(
                    {
                        "writer": writer_index,
                        "iteration": iteration,
                        "loop": loop_index,
                        "units": object_count,
                    },
                    sort_keys=True,
                ),
            )
        return f"host_benchmark_{profile}", estimated_bytes

    def _run_host_write_once(
        self,
        writer_index: int,
        iteration: int,
        loop_index: int,
    ) -> tuple[str, int]:
        root = self._host_writer_root()
        if self.cfg.host_writer_mode == "dd":
            slot = loop_index % max(1, self.cfg.host_writer_slots)
            target = root / "dd" / f"writer_{writer_index}" / f"slot_{slot}.bin"
            total_bytes = max(1, self.cfg.host_writer_total_mb) * MIB
            self._write_binary_atomic(target, total_bytes)
            return "host_dd", total_bytes
        if self.cfg.host_writer_mode == "smallfiles":
            return self._run_host_smallfiles_write(
                root=root,
                writer_index=writer_index,
                iteration=iteration,
                loop_index=loop_index,
                profile="host_smallfiles",
            )
        if self.cfg.host_writer_mode == "process":
            return self._run_host_process_like_write(
                root=root,
                writer_index=writer_index,
                iteration=iteration,
                loop_index=loop_index,
                profile="host_process",
            )
        return self._run_host_benchmark_write(
            root=root,
            writer_index=writer_index,
            iteration=iteration,
            loop_index=loop_index,
        )

    def _run_host_writer(
        self,
        writer_index: int,
        iteration: int,
        stop_event: threading.Event,
        started_event: threading.Event,
    ) -> None:
        t0 = now_ms()
        ops = 0
        profile = f"host_{self.cfg.host_writer_mode}"
        requested_write_bytes: int | None = None
        failure: Exception | None = None
        try:
            while not stop_event.is_set():
                profile, requested_write_bytes = self._run_host_write_once(writer_index, iteration, ops)
                ops += 1
                started_event.set()
        except Exception as exc:
            failure = exc
        finally:
            t1 = now_ms()
            self.record(
                phase="host_write",
                profile=profile,
                iteration=iteration,
                duration_ms=round(t1 - t0, 3),
                write_ops=ops,
                requested_write_bytes=None if requested_write_bytes is None else requested_write_bytes * max(1, ops),
                host_writer_mode=self.cfg.host_writer_mode,
                host_writer_index=writer_index,
                host_writer_root=str(self._host_writer_root()),
                rc=0 if failure is None else 1,
                stderr="" if failure is None else str(failure),
            )
        if failure is not None:
            raise failure

    def _run_live_writer(
        self,
        c: ContainerSpec,
        iteration: int,
        stop_event: threading.Event,
        started_event: threading.Event,
    ) -> None:
        t0 = now_ms()
        ops = 0
        rc = 0
        stderr = ""
        stdout = ""
        profile = self.cfg.writer_mode
        requested_write_bytes: int | None = None
        failure: Exception | None = None
        try:
            while not stop_event.is_set():
                spec = self._build_live_write_spec(c, iteration, loop_index=ops)
                p = self._exec_in_container(c.name, spec.command)
                rc = p.returncode
                stdout = p.stdout or ""
                stderr = p.stderr or ""
                profile = spec.profile
                requested_write_bytes = spec.estimated_write_bytes
                ops += 1
                started_event.set()
                if p.returncode != 0:
                    break
        except Exception as exc:
            rc = 1
            stderr = str(exc)
            failure = exc
        finally:
            t1 = now_ms()
            self.record(
                phase="live_write",
                profile=profile,
                container=c.name,
                dataset=c.dataset,
                iteration=iteration,
                duration_ms=round(t1 - t0, 3),
                write_ops=ops,
                requested_write_bytes=None if requested_write_bytes is None else requested_write_bytes * max(1, ops),
                writer_mode=self.cfg.writer_mode,
                rc=rc,
                stdout=stdout[:4000],
                stderr=stderr[:4000],
            )
        if failure is not None:
            raise failure

    def _snapshot_stats(self, dataset: str, snap_name: str) -> tuple[int | None, int | None]:
        p = sh(
            [self.cfg.zfs_bin, "get", "-Hp", "-o", "property,value", "written,used", f"{dataset}@{snap_name}"],
            check=False,
        )
        if p.returncode != 0:
            return None, None
        values: dict[str, int] = {}
        for line in (p.stdout or "").splitlines():
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                values[parts[0].strip()] = int(parts[1].strip())
            except ValueError:
                continue
        return values.get("written"), values.get("used")

    # -------- snapshot rounds --------

    def _pause(self, c: ContainerSpec) -> float:
        t0 = now_ms()
        sh([self.cfg.runc_bin, "--root", str(self.cfg.state_root), "pause", c.name])
        t1 = now_ms()
        return round(t1 - t0, 3)

    def _resume(self, c: ContainerSpec) -> float:
        t0 = now_ms()
        sh([self.cfg.runc_bin, "--root", str(self.cfg.state_root), "resume", c.name])
        t1 = now_ms()
        return round(t1 - t0, 3)

    def _snapshot(
        self,
        c: ContainerSpec,
        snap_name: str,
        iteration: int,
        kind: str,
        barrier: threading.Barrier | None = None,
    ) -> None:
        pause_ms = None
        resume_ms = None
        if self.cfg.pause_before_snapshot:
            pause_ms = self._pause(c)

        if barrier is not None:
            barrier.wait()
        t0 = now_ms()
        sh([self.cfg.zfs_bin, "snapshot", f"{c.dataset}@{snap_name}"])
        t1 = now_ms()

        if self.cfg.pause_before_snapshot:
            resume_ms = self._resume(c)

        written_bytes, used_bytes = self._snapshot_stats(c.dataset, snap_name)

        self.record(
            phase=kind,
            container=c.name,
            dataset=c.dataset,
            iteration=iteration,
            snapshot=f"{c.dataset}@{snap_name}",
            duration_ms=round(t1 - t0, 3),
            written_bytes=written_bytes,
            used_bytes=used_bytes,
            pause_ms=pause_ms,
            resume_ms=resume_ms,
            rc=0,
        )

    def _restore_to_snapshot(
        self,
        c: ContainerSpec,
        snap_name: str,
        iteration: int,
        barrier: threading.Barrier | None = None,
    ) -> None:
        delete_started = now_ms()
        sh([self.cfg.runc_bin, "--root", str(self.cfg.state_root), "delete", "-f", c.name], check=False)
        delete_finished = now_ms()
        if barrier is not None:
            barrier.wait()
        rollback_started = now_ms()
        sh([self.cfg.zfs_bin, "rollback", "-r", f"{c.dataset}@{snap_name}"])
        rollback_finished = now_ms()
        restart_started = now_ms()
        self._runc_create_start(c.name, c.bundle_dir)
        restart_finished = now_ms()
        self.record(
            phase="restore_fs",
            container=c.name,
            dataset=c.dataset,
            iteration=iteration,
            snapshot=f"{c.dataset}@{snap_name}",
            duration_ms=round(rollback_finished - rollback_started, 3),
            delete_ms=round(delete_finished - delete_started, 3),
            restart_ms=round(restart_finished - restart_started, 3),
            rc=0,
        )
        self.record(
            phase="restore_cycle",
            container=c.name,
            dataset=c.dataset,
            iteration=iteration,
            snapshot=f"{c.dataset}@{snap_name}",
            duration_ms=round(restart_finished - delete_started, 3),
            delete_ms=round(delete_finished - delete_started, 3),
            rollback_ms=round(rollback_finished - rollback_started, 3),
            restart_ms=round(restart_finished - restart_started, 3),
            rc=0,
        )

    def _concurrent_initial_snapshot(self, containers: list[ContainerSpec]) -> None:
        self._snapshot_round(containers, iteration=0, kind="snapshot_init")

    def _concurrent_snapshot_round(self, containers: list[ContainerSpec], it: int) -> None:
        self._snapshot_round(containers, iteration=it, kind="snapshot")

    def _snapshot_round(self, containers: list[ContainerSpec], iteration: int, kind: str) -> None:
        start_barrier = threading.Barrier(len(containers))
        with ThreadPoolExecutor(max_workers=min(self.cfg.max_workers, len(containers))) as ex:
            futs = []
            for c in containers:
                snap_name = f"{self.cfg.snapshot_prefix}-{iteration}"
                futs.append(ex.submit(self._snapshot_with_barrier, c, snap_name, iteration, kind, start_barrier))
            for fut in as_completed(futs):
                fut.result()

    def _snapshot_with_barrier(
        self,
        c: ContainerSpec,
        snap_name: str,
        iteration: int,
        kind: str,
        barrier: threading.Barrier,
    ) -> None:
        self._snapshot(c, snap_name, iteration, kind, barrier)

    def _select_restore_targets(self, containers: list[ContainerSpec], iteration: int) -> list[ContainerSpec]:
        if self.cfg.restore_ratio <= 0.0 or not containers:
            return []
        requested = int(round(len(containers) * self.cfg.restore_ratio))
        if len(containers) > 1:
            requested = min(requested, len(containers) - 1)
        requested = max(0, min(requested, len(containers)))
        if requested == 0:
            return []
        offset = ((iteration - 1) * requested) % len(containers)
        return [containers[(offset + i) % len(containers)] for i in range(requested)]

    def _run_mixed_round(self, containers: list[ContainerSpec], iteration: int) -> None:
        restore_targets = self._select_restore_targets(containers, iteration)
        restore_names = {c.name for c in restore_targets}
        snapshot_targets = [c for c in containers if c.name not in restore_names]
        if not snapshot_targets and restore_targets:
            snapshot_targets = [restore_targets.pop()]
            restore_names = {c.name for c in restore_targets}

        writer_targets: list[ContainerSpec] = []
        if self.cfg.snapshot_while_writing and snapshot_targets:
            writer_count = self.cfg.writer_containers or len(snapshot_targets)
            writer_targets = snapshot_targets[: min(writer_count, len(snapshot_targets))]
        host_writer_count = max(0, self.cfg.host_writer_count)

        stop_event = threading.Event()
        started_events = [threading.Event() for _ in writer_targets]
        host_started_events = [threading.Event() for _ in range(host_writer_count)]
        writer_workers = min(len(writer_targets), max(1, self.cfg.max_workers))
        host_writer_workers = min(host_writer_count, max(1, self.cfg.max_workers))

        with (
            ThreadPoolExecutor(max_workers=max(1, writer_workers)) as writer_pool,
            ThreadPoolExecutor(max_workers=max(1, host_writer_workers)) as host_writer_pool,
        ):
            writer_futures = [
                writer_pool.submit(self._run_live_writer, c, iteration, stop_event, started_event)
                for c, started_event in zip(writer_targets, started_events)
            ]
            host_writer_futures = [
                host_writer_pool.submit(self._run_host_writer, writer_index, iteration, stop_event, started_event)
                for writer_index, started_event in enumerate(host_started_events)
            ]
            if writer_targets or host_started_events:
                deadline = time.monotonic() + (max(0, self.cfg.writer_warmup_ms) / 1000.0)
                for started_event in [*started_events, *host_started_events]:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    started_event.wait(timeout=remaining)

            snapshot_count = len(snapshot_targets)
            restore_count = len(restore_targets)
            zfs_barrier = threading.Barrier(snapshot_count + restore_count) if (snapshot_count + restore_count) > 1 else None
            round_workers = min(
                max(1, self.cfg.max_workers),
                snapshot_count + max(restore_count, 0),
            )
            restore_workers = self.cfg.restore_workers or round_workers
            mixed_workers = max(1, min(round_workers, snapshot_count + max(restore_workers, 0)))
            restore_snapshot_name = f"{self.cfg.snapshot_prefix}-{iteration - 1}"

            try:
                with ThreadPoolExecutor(max_workers=mixed_workers) as ex:
                    futs = []
                    for c in snapshot_targets:
                        futs.append(
                            ex.submit(
                                self._snapshot,
                                c,
                                f"{self.cfg.snapshot_prefix}-{iteration}",
                                iteration,
                                "snapshot_mixed",
                                zfs_barrier,
                            )
                        )
                    for c in restore_targets:
                        futs.append(ex.submit(self._restore_to_snapshot, c, restore_snapshot_name, iteration, zfs_barrier))
                    for fut in as_completed(futs):
                        fut.result()
            finally:
                stop_event.set()
                for fut in writer_futures:
                    fut.result()
                for fut in host_writer_futures:
                    fut.result()

    # -------- cleanup --------

    def _cleanup_containers(self, containers: Iterable[ContainerSpec]) -> None:
        for c in containers:
            sh([self.cfg.runc_bin, "--root", str(self.cfg.state_root), "delete", "-f", c.name], check=False)
            sh([self.cfg.zfs_bin, "destroy", "-r", c.dataset], check=False)
            if c.bundle_dir.exists():
                shutil.rmtree(c.bundle_dir, ignore_errors=True)

    # -------- outputs --------

    def _write_results(self) -> None:
        out_dir = self.cfg.base_dir / "results"
        mkdirp(out_dir)

        run_meta = {
            "pool_name": self.cfg.pool_name,
            "file_vdev": None if self.cfg.file_vdev is None else str(self.cfg.file_vdev),
            "device": self.cfg.device,
            "pool_size": self.cfg.pool_size,
            "image": self.cfg.image,
            "containers": self.cfg.containers,
            "iterations": self.cfg.iterations,
            "workload": self.cfg.workload,
            "custom_cmd": self.cfg.custom_cmd,
            "files_per_iter": self.cfg.files_per_iter,
            "pause_before_snapshot": self.cfg.pause_before_snapshot,
            "snapshot_while_writing": self.cfg.snapshot_while_writing,
            "writer_containers": self.cfg.writer_containers,
            "writer_mode": self.cfg.writer_mode,
            "live_writer_cmd": self.cfg.live_writer_cmd,
            "writer_chunk_mb": self.cfg.writer_chunk_mb,
            "writer_files_per_loop": self.cfg.writer_files_per_loop,
            "writer_slots": self.cfg.writer_slots,
            "writer_warmup_ms": self.cfg.writer_warmup_ms,
            "restore_ratio": self.cfg.restore_ratio,
            "restore_workers": self.cfg.restore_workers,
            "host_writer_count": self.cfg.host_writer_count,
            "host_writer_mode": self.cfg.host_writer_mode,
            "host_writer_dir": None if self.cfg.host_writer_dir is None else str(self.cfg.host_writer_dir),
            "host_writer_total_mb": self.cfg.host_writer_total_mb,
            "host_writer_files_per_loop": self.cfg.host_writer_files_per_loop,
            "host_writer_slots": self.cfg.host_writer_slots,
            "effective_workers": min(self.cfg.max_workers, self.cfg.containers),
        }
        write_json(out_dir / "run_meta.json", run_meta)
        write_json(out_dir / "results.json", {"records": self.results})

        csv_path = out_dir / "results.csv"
        fieldnames = sorted({k for row in self.results for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for row in self.results:
                w.writerow(row)

        summary = self._summarize()
        write_json(out_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2, sort_keys=True))

    def _summarize(self) -> dict:
        def row_profile(row: dict) -> str:
            return str(row.get("profile") or row.get("container_profile") or "")

        phases = {}
        profile_summary: dict[str, dict[str, dict[str, int | float]]] = {}
        ordered_phases = []
        for row in self.results:
            phase = str(row.get("phase", ""))
            if phase and phase not in ordered_phases:
                ordered_phases.append(phase)
        for phase in ordered_phases:
            vals = [r["duration_ms"] for r in self.results if r.get("phase") == phase and r.get("rc") == 0]
            if not vals:
                continue
            vals_sorted = sorted(vals)
            phases[phase] = {
                "count": len(vals),
                "min_ms": round(vals_sorted[0], 3),
                "p50_ms": round(vals_sorted[len(vals_sorted) // 2], 3),
                "p95_ms": round(vals_sorted[min(len(vals_sorted) - 1, int(len(vals_sorted) * 0.95))], 3),
                "max_ms": round(vals_sorted[-1], 3),
                "avg_ms": round(sum(vals_sorted) / len(vals_sorted), 3),
            }
            written_vals = [
                int(r["written_bytes"])
                for r in self.results
                if r.get("phase") == phase and r.get("rc") == 0 and r.get("written_bytes") is not None
            ]
            if written_vals:
                written_vals.sort()
                phases[phase]["avg_written_bytes"] = round(sum(written_vals) / len(written_vals), 3)
                phases[phase]["p50_written_bytes"] = written_vals[len(written_vals) // 2]
                phases[phase]["p95_written_bytes"] = written_vals[
                    min(len(written_vals) - 1, int(len(written_vals) * 0.95))
                ]
            used_vals = [
                int(r["used_bytes"])
                for r in self.results
                if r.get("phase") == phase and r.get("rc") == 0 and r.get("used_bytes") is not None
            ]
            if used_vals:
                used_vals.sort()
                phases[phase]["avg_used_bytes"] = round(sum(used_vals) / len(used_vals), 3)
                phases[phase]["p50_used_bytes"] = used_vals[len(used_vals) // 2]
                phases[phase]["p95_used_bytes"] = used_vals[
                    min(len(used_vals) - 1, int(len(used_vals) * 0.95))
                ]
            ordered_profiles: list[str] = []
            for row in self.results:
                if row.get("phase") != phase:
                    continue
                profile = row_profile(row)
                if profile and profile not in ordered_profiles:
                    ordered_profiles.append(profile)
            phase_profiles: dict[str, dict[str, int | float]] = {}
            for profile in ordered_profiles:
                profile_vals = [
                    r["duration_ms"]
                    for r in self.results
                    if r.get("phase") == phase and row_profile(r) == profile and r.get("rc") == 0
                ]
                if not profile_vals:
                    continue
                profile_vals_sorted = sorted(profile_vals)
                phase_profiles[profile] = {
                    "count": len(profile_vals_sorted),
                    "min_ms": round(profile_vals_sorted[0], 3),
                    "p50_ms": round(profile_vals_sorted[len(profile_vals_sorted) // 2], 3),
                    "p95_ms": round(
                        profile_vals_sorted[
                            min(len(profile_vals_sorted) - 1, int(len(profile_vals_sorted) * 0.95))
                        ],
                        3,
                    ),
                    "max_ms": round(profile_vals_sorted[-1], 3),
                    "avg_ms": round(sum(profile_vals_sorted) / len(profile_vals_sorted), 3),
                }
                profile_written_vals = [
                    int(r["written_bytes"])
                    for r in self.results
                    if r.get("phase") == phase
                    and row_profile(r) == profile
                    and r.get("rc") == 0
                    and r.get("written_bytes") is not None
                ]
                if profile_written_vals:
                    profile_written_vals.sort()
                    phase_profiles[profile]["avg_written_bytes"] = round(
                        sum(profile_written_vals) / len(profile_written_vals), 3
                    )
                    phase_profiles[profile]["p50_written_bytes"] = profile_written_vals[
                        len(profile_written_vals) // 2
                    ]
                    phase_profiles[profile]["p95_written_bytes"] = profile_written_vals[
                        min(len(profile_written_vals) - 1, int(len(profile_written_vals) * 0.95))
                    ]
            if phase_profiles:
                profile_summary[phase] = phase_profiles
        return {"summary": phases, "profile_summary": profile_summary}


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> BenchConfig:
    p = argparse.ArgumentParser(description="Benchmark ZFS snapshot latency with runc containers")
    p.add_argument("--base-dir", default="/var/tmp/zfs-runc-bench")
    p.add_argument("--state-root", default="/run/zfs-runc-bench-runc")
    p.add_argument("--pool-name", default="benchpool")
    p.add_argument("--file-vdev", default=None, help="Path to sparse file vdev, e.g. /var/tmp/zpool.img")
    p.add_argument("--device", default=None, help="Block device for zpool, e.g. /dev/vdb")
    p.add_argument("--pool-size", default="100G")
    p.add_argument("--image", default="debian:bookworm-slim")
    p.add_argument("--containers", type=int, default=1)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--workload", choices=["smallfiles", "apt", "pip", "benchmark"], default="smallfiles")
    p.add_argument("--custom-cmd", default=None)
    p.add_argument("--files-per-iter", type=int, default=2000)
    p.add_argument("--snapshot-prefix", default="snap")
    p.add_argument("--keep", action="store_true")
    p.add_argument("--docker-bin", default="docker")
    p.add_argument("--runc-bin", default="runc")
    p.add_argument("--zpool-bin", default="zpool")
    p.add_argument("--zfs-bin", default="zfs")
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--pause-before-snapshot", action="store_true")
    p.add_argument("--snapshot-while-writing", action="store_true")
    p.add_argument("--writer-containers", type=int, default=0)
    p.add_argument("--writer-mode", choices=["dd", "smallfiles", "workload", "benchmark"], default="dd")
    p.add_argument("--live-writer-cmd", default=None)
    p.add_argument("--writer-chunk-mb", type=int, default=8)
    p.add_argument("--writer-files-per-loop", type=int, default=512)
    p.add_argument("--writer-slots", type=int, default=8)
    p.add_argument("--writer-warmup-ms", type=int, default=1000)
    p.add_argument("--restore-ratio", type=float, default=0.0)
    p.add_argument("--restore-workers", type=int, default=0)
    p.add_argument("--host-writer-count", type=int, default=0)
    p.add_argument("--host-writer-mode", choices=["dd", "smallfiles", "process", "benchmark"], default="process")
    p.add_argument("--host-writer-dir", default=None)
    p.add_argument("--host-writer-total-mb", type=int, default=256)
    p.add_argument("--host-writer-files-per-loop", type=int, default=64)
    p.add_argument("--host-writer-slots", type=int, default=8)
    args = p.parse_args()

    if os.geteuid() != 0:
        raise SystemExit("This benchmark must run as root.")
    if not args.file_vdev and not args.device:
        raise SystemExit("Provide either --file-vdev or --device")
    if args.file_vdev and args.device:
        raise SystemExit("Use only one of --file-vdev / --device")
    if args.writer_containers < 0:
        raise SystemExit("--writer-containers must be >= 0")
    if args.writer_chunk_mb <= 0:
        raise SystemExit("--writer-chunk-mb must be > 0")
    if args.writer_files_per_loop <= 0:
        raise SystemExit("--writer-files-per-loop must be > 0")
    if args.writer_slots <= 0:
        raise SystemExit("--writer-slots must be > 0")
    if args.writer_warmup_ms < 0:
        raise SystemExit("--writer-warmup-ms must be >= 0")
    if not 0.0 <= args.restore_ratio <= 1.0:
        raise SystemExit("--restore-ratio must be between 0.0 and 1.0")
    if args.restore_workers < 0:
        raise SystemExit("--restore-workers must be >= 0")
    if args.host_writer_count < 0:
        raise SystemExit("--host-writer-count must be >= 0")
    if args.host_writer_total_mb <= 0:
        raise SystemExit("--host-writer-total-mb must be > 0")
    if args.host_writer_files_per_loop <= 0:
        raise SystemExit("--host-writer-files-per-loop must be > 0")
    if args.host_writer_slots <= 0:
        raise SystemExit("--host-writer-slots must be > 0")

    return BenchConfig(
        base_dir=Path(args.base_dir),
        state_root=Path(args.state_root),
        pool_name=args.pool_name,
        file_vdev=None if args.file_vdev is None else Path(args.file_vdev),
        device=args.device,
        pool_size=args.pool_size,
        image=args.image,
        containers=args.containers,
        iterations=args.iterations,
        workload=args.workload,
        custom_cmd=args.custom_cmd,
        files_per_iter=args.files_per_iter,
        snapshot_prefix=args.snapshot_prefix,
        keep=args.keep,
        docker_bin=args.docker_bin,
        runc_bin=args.runc_bin,
        zpool_bin=args.zpool_bin,
        zfs_bin=args.zfs_bin,
        max_workers=args.max_workers,
        pause_before_snapshot=args.pause_before_snapshot,
        snapshot_while_writing=bool(args.snapshot_while_writing),
        writer_containers=args.writer_containers,
        writer_mode=str(args.writer_mode),
        live_writer_cmd=None if args.live_writer_cmd is None else str(args.live_writer_cmd),
        writer_chunk_mb=args.writer_chunk_mb,
        writer_files_per_loop=args.writer_files_per_loop,
        writer_slots=args.writer_slots,
        writer_warmup_ms=args.writer_warmup_ms,
        restore_ratio=float(args.restore_ratio),
        restore_workers=args.restore_workers,
        host_writer_count=args.host_writer_count,
        host_writer_mode=str(args.host_writer_mode),
        host_writer_dir=None if args.host_writer_dir is None else Path(args.host_writer_dir),
        host_writer_total_mb=args.host_writer_total_mb,
        host_writer_files_per_loop=args.host_writer_files_per_loop,
        host_writer_slots=args.host_writer_slots,
    )


def main() -> int:
    cfg = parse_args()
    bench = ZfsRuncBenchmark(cfg)
    bench.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

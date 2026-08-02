from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from dataclasses import dataclass


@dataclass(frozen=True)
class ResolvedSandbox:
    runtime: str
    object_id: str
    runtime_name: str
    is_running: bool
    init_pid: int | None
    cgroup_path: str | None
    cgroup_id: int | None


class RuntimeResolver:
    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        runc_bin: str = "runc",
        runc_state_root: str | Path | None = None,
    ) -> None:
        self._docker_bin = docker_bin
        self._runc_bin = runc_bin
        self._runc_state_root = None if runc_state_root is None else str(runc_state_root)

    def resolve(self, runtime: str, object_id: str) -> ResolvedSandbox:
        if runtime == "docker":
            return self._resolve_docker(object_id)
        if runtime == "runc":
            return self._resolve_runc(object_id)
        raise ValueError(f"unsupported runtime: {runtime}")

    def _resolve_docker(self, object_id: str) -> ResolvedSandbox:
        raw = subprocess.run(
            [self._docker_bin, "inspect", object_id],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(raw.stdout)[0]
        state = payload["State"]
        pid = int(state.get("Pid", 0) or 0)
        running = bool(state.get("Running", False) or state.get("Paused", False) or state.get("Status") == "paused")
        cgroup_path, cgroup_id = self._resolve_cgroup(pid if running else 0)
        return ResolvedSandbox(
            runtime="docker",
            object_id=object_id,
            runtime_name="docker",
            is_running=running,
            init_pid=pid if pid > 0 else None,
            cgroup_path=cgroup_path,
            cgroup_id=cgroup_id,
        )

    def _resolve_runc(self, object_id: str) -> ResolvedSandbox:
        command = [self._runc_bin]
        if self._runc_state_root is not None:
            command.extend(["--root", self._runc_state_root])
        command.extend(["state", object_id])
        try:
            raw = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            return ResolvedSandbox(
                runtime="runc",
                object_id=object_id,
                runtime_name="runc",
                is_running=False,
                init_pid=None,
                cgroup_path=None,
                cgroup_id=None,
            )
        payload = json.loads(raw.stdout)
        status = str(payload.get("status", ""))
        pid = int(payload.get("pid", 0) or 0)
        running = status in {"running", "paused", "created"} and pid > 0
        cgroup_path, cgroup_id = self._resolve_cgroup(pid if running else 0)
        return ResolvedSandbox(
            runtime="runc",
            object_id=object_id,
            runtime_name="runc",
            is_running=running,
            init_pid=pid if pid > 0 else None,
            cgroup_path=cgroup_path,
            cgroup_id=cgroup_id,
        )

    def _resolve_cgroup(self, pid: int) -> tuple[str | None, int | None]:
        if pid <= 0:
            return None, None
        try:
            with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("0::"):
                        rel = line.strip().split(":", 2)[2]
                        st = os.stat(f"/sys/fs/cgroup{rel}")
                        return rel, int(st.st_ino)
        except FileNotFoundError:
            return None, None
        return None, None

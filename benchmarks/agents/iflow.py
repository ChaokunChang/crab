from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time

from agents.iflow_integration.harness import (
    IFLOW_HOME_MOUNT_PATH,
    LOGS_MOUNT_PATH,
    NPM_HOME_MOUNT_PATH,
    RUNTIME_MOUNT_PATH,
    prepare_iflow_runtime,
    prepare_iflow_state,
)
from benchmarks.agents.base import BaseAgent


class IFlowAgent(BaseAgent):
    agent_type = "iflow"
    requires_manual_task_launch = True
    requires_network_namespace = True

    def __init__(self, harness, sandbox, task_description, task_config) -> None:
        super().__init__(harness, sandbox, task_description, task_config)
        self._tick_seconds = max(0.001, float(self.task_config.options.get("action_tick_seconds", 0.2)))
        self._state_lock = threading.Lock()
        self._started_at_monotonic: float | None = None
        self._finished_at_monotonic: float | None = None

    def prepare_sandbox(self) -> None:
        assert self.harness.root is not None
        assert self.harness.interceptor is not None
        sandbox_root = self.harness.root / "iflow" / str(self.sandbox.sandbox_id)
        prepared_runtime = prepare_iflow_runtime(work_root=sandbox_root)
        prepared_state = prepare_iflow_state(
            work_root=sandbox_root,
            base_url=f"http://{self.harness._benchmark_bridge_ip}:{self.harness.interceptor.port}/v1",
            model_name=str(os.environ.get("AGENT_CR_IFLOW_MODEL_NAME", "agent-cr-iflow-scripted")),
        )
        self.sandbox.launch_metadata["iflow"] = {
            "runtime_root": str(prepared_runtime.root),
            "iflow_home": str(prepared_state.iflow_home),
            "npm_home": str(prepared_state.npm_home),
            "logs_dir": str(prepared_state.logs_dir),
            "entrypoint": prepared_runtime.mounted_entrypoint,
            "ignore_process_rules": prepared_runtime.ignore_process_rules,
        }

    def configure_bundle(self) -> None:
        metadata = self.sandbox.launch_metadata.get("iflow", {})
        if not metadata:
            raise RuntimeError(f"missing iflow launch metadata for sandbox {self.sandbox.sandbox_id}")
        config_path = self.sandbox.bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg["process"]["terminal"] = False
        cfg["process"]["cwd"] = "/work"
        cfg["process"]["args"] = ["/bin/sh", "-lc", "trap : TERM INT; while true; do sleep 3600; done"]
        cfg["process"]["env"] = [
            f"PATH={RUNTIME_MOUNT_PATH}/global/bin:{RUNTIME_MOUNT_PATH}/node/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONUNBUFFERED=1",
            "UV_USE_IO_URING=0",
            "HOME=/root",
            "IFLOW_NON_INTERACTIVE=true",
        ]
        mounts = [
            mount
            for mount in cfg.get("mounts", [])
            if mount.get("destination")
            not in {RUNTIME_MOUNT_PATH, IFLOW_HOME_MOUNT_PATH, NPM_HOME_MOUNT_PATH, LOGS_MOUNT_PATH}
        ]
        mounts.extend(
            [
                {
                    "destination": RUNTIME_MOUNT_PATH,
                    "type": "bind",
                    "source": metadata["runtime_root"],
                    "options": ["rbind", "ro"],
                },
                {
                    "destination": IFLOW_HOME_MOUNT_PATH,
                    "type": "bind",
                    "source": metadata["iflow_home"],
                    "options": ["rbind", "rw"],
                },
                {
                    "destination": NPM_HOME_MOUNT_PATH,
                    "type": "bind",
                    "source": metadata["npm_home"],
                    "options": ["rbind", "rw"],
                },
                {
                    "destination": LOGS_MOUNT_PATH,
                    "type": "bind",
                    "source": metadata["logs_dir"],
                    "options": ["rbind", "rw"],
                },
            ]
        )
        cfg["mounts"] = mounts
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def extra_launch_metadata(self) -> dict[str, object]:
        metadata = self.sandbox.launch_metadata.get("iflow", {})
        return {
            "host_inspector_ignore_process_rules": metadata.get("ignore_process_rules", []),
        }

    def rootfs_init_dirs(self) -> list[str]:
        return super().rootfs_init_dirs() + [
            "root",
            "root/.iflow",
            "root/.npm",
            "opt/iflow-runtime",
            "opt/iflow-logs",
        ]

    def perform_task(self) -> None:
        metadata = self.sandbox.launch_metadata.get("iflow", {})
        entrypoint = metadata.get("entrypoint")
        if entrypoint is None:
            raise RuntimeError(f"missing iflow entrypoint for sandbox {self.sandbox.sandbox_id}")
        assert self.harness.runtime_state_root is not None
        with self._state_lock:
            self._started_at_monotonic = time.monotonic()
            self._finished_at_monotonic = None
        escaped_task = shlex.quote(self.task_description.prompt)
        command = (
            "export HOME=/root; "
            "export IFLOW_NON_INTERACTIVE=true; "
            "cd /work && "
            f"exec {RUNTIME_MOUNT_PATH}/node/bin/node {entrypoint} -p {escaped_task} "
            ">/dev/null 2>/dev/null"
        )
        exec_command = [
            "runc",
            "--root",
            str(self.harness.runtime_state_root),
            "exec",
        ]
        for key, value in self.task_config.options.items():
            exec_command.extend(["--env", f"{key}={value}"])
        exec_command.extend([str(self.sandbox.sandbox_id), "/bin/sh", "-lc", command])
        try:
            subprocess.run(exec_command, check=True)
        finally:
            with self._state_lock:
                self._finished_at_monotonic = time.monotonic()

    def poll_status(self) -> dict[str, object]:
        actions = self._synthetic_action_count()
        payload = {
            "agent_type": self.agent_type,
            "state": self._task_state(),
            "total_actions": actions,
            "filesystem_actions": actions,
            "process_actions": actions,
            "network_actions": actions,
            "stateful_actions": actions,
        }
        self.sandbox.last_status = payload
        return payload

    def wait_for_progress(self, *, minimum_actions: int) -> dict[str, object]:
        self._wait_for_action_count(minimum_actions)
        payload = self.poll_status()
        self._record_activity(payload)
        return payload

    def wait_for_action_delta(self, *, delta: int) -> dict[str, object]:
        baseline = int(self.sandbox.last_status.get("total_actions", 0))
        self._wait_for_action_count(baseline + delta)
        payload = self.poll_status()
        self._record_activity(payload)
        return payload

    def _synthetic_action_count(self) -> int:
        with self._state_lock:
            started_at = self._started_at_monotonic
            finished_at = self._finished_at_monotonic
        if started_at is None:
            return 0
        end_time = finished_at if finished_at is not None else time.monotonic()
        return max(0, int((end_time - started_at) / self._tick_seconds))

    def _task_state(self) -> str:
        with self._state_lock:
            started_at = self._started_at_monotonic
            finished_at = self._finished_at_monotonic
        if started_at is None:
            return "idle"
        if finished_at is None:
            return "running"
        return "finished"

    def _record_activity(self, payload: dict[str, object]) -> None:
        record_activity = getattr(self.harness, "record_activity", None)
        if callable(record_activity):
            record_activity(self.sandbox, payload)
            return
        self.sandbox.last_status = payload

    def _wait_for_action_count(self, target_actions: int) -> None:
        deadline = time.monotonic() + max(45.0, target_actions * self._tick_seconds * 4.0)
        while time.monotonic() < deadline:
            current_actions = self._synthetic_action_count()
            if current_actions >= target_actions:
                return
            if self._task_state() == "finished":
                raise RuntimeError(
                    f"iflow task finished before reaching synthetic action count {target_actions}; "
                    f"last observed count was {current_actions}"
                )
            time.sleep(min(0.2, self._tick_seconds))
        raise RuntimeError(f"timed out waiting for iflow synthetic action count {target_actions}")

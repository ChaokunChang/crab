from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shlex
import subprocess
import threading
import time

from integrations.agents.base import BaseAgent
from integrations.sandboxes.iflow.harness import (
    IFLOW_HOME_MOUNT_PATH,
    LOGS_MOUNT_PATH,
    NPM_HOME_MOUNT_PATH,
    RUNTIME_MOUNT_PATH,
    _IO_URING_SECCOMP,
    prepare_iflow_runtime,
    prepare_iflow_state,
)
from integrations.sandboxes.runtime.bundle import merge_environment_defaults

logger = logging.getLogger(__name__)

SANDBOX_TASK_OUTPUT_DIR = Path("/data/iflow-task-logs")
IFLOW_WRAPPER_ARG = "--agent-cr-iflow-wrapper"
_IFLOW_INLINE_WRAPPER = """
const fs = require("fs");
const { spawn } = require("child_process");

const entrypoint = process.env.AGENT_CR_IFLOW_ENTRYPOINT;
const task = process.env.AGENT_CR_IFLOW_TASK || "";
const donePath = process.env.AGENT_CR_IFLOW_DONE_PATH;
const exitPath = process.env.AGENT_CR_IFLOW_EXIT_PATH;

let settled = false;

function finish(code) {
  if (settled) {
    return;
  }
  settled = true;
  if (code === 0) {
    fs.writeFileSync(donePath, "done\\n", "utf8");
  }
  fs.writeFileSync(exitPath, `${code}\\n`, "utf8");
}

const child = spawn(process.execPath, [entrypoint, "-p", task], {
  cwd: "/work",
  env: process.env,
  stdio: "inherit",
});

child.once("error", (error) => {
  console.error(error && error.stack ? error.stack : String(error));
  finish(127);
  process.exit(127);
});

child.once("exit", (code, signal) => {
  const exitCode = code === null ? 1 : code;
  finish(exitCode);
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }
  process.exit(exitCode);
});
""".strip()


class IFlowAgent(BaseAgent):
    agent_type = "iflow"
    requires_manual_task_launch = True
    requires_network_namespace = True
    DEFAULT_ACTION_TICK_SECONDS = 1
    DEFAULT_MAX_SESSION_TURNS = 4096
    TASK_POLL_INTERVAL_SECONDS = 0.1

    def __init__(
        self,
        sandbox,
        task_description,
        task_config,
        *,
        runtime_state_root=None,
        agent_host_dir=None,
        llm_base_url=None,
    ) -> None:
        super().__init__(
            sandbox,
            task_description,
            task_config,
            runtime_state_root=runtime_state_root,
            agent_host_dir=agent_host_dir,
            llm_base_url=llm_base_url,
        )
        self._tick_seconds = max(
            0.001,
            float(self.task_config.options.get("action_tick_seconds", self.DEFAULT_ACTION_TICK_SECONDS)),
        )
        self._state_lock = threading.Lock()
        self._restore_complete_event = threading.Event()
        self._stop_requested = threading.Event()
        self._started_at_monotonic: float | None = None
        self._finished_at_monotonic: float | None = None

    def survives_fault_relaunch(self) -> bool:
        return True

    def prepare_sandbox(self) -> None:
        assert self.agent_host_dir is not None
        assert self.llm_base_url is not None
        logger.debug(f"preparing sandbox in {self.agent_host_dir=}")
        sandbox_root = self.agent_host_dir
        prepared_runtime = prepare_iflow_runtime(work_root=sandbox_root)
        prepared_state = prepare_iflow_state(
            work_root=sandbox_root,
            base_url=self.llm_base_url,
            model_name=str(os.environ.get("AGENT_CR_IFLOW_MODEL_NAME", "agent-cr-iflow-scripted")),
            max_session_turns=int(
                os.environ.get("AGENT_CR_IFLOW_BENCHMARK_MAX_SESSION_TURNS", str(self.DEFAULT_MAX_SESSION_TURNS))
            ),
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
        entrypoint = metadata.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint:
            raise RuntimeError(f"missing iflow entrypoint for sandbox {self.sandbox.sandbox_id}")
        logs_dir = self._resolve_logs_dir(metadata)
        marker_paths = self._task_marker_paths(logs_dir)
        marker_mount_paths = self._task_marker_mount_paths(marker_paths)
        output_paths = self._sandbox_task_output_paths()
        escaped_task = shlex.quote(self.task_description.prompt)
        config_path = self.sandbox.bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

        linux_cfg = cfg.get("linux", {})
        linux_cfg["seccomp"] = _IO_URING_SECCOMP
        cfg["linux"] = linux_cfg

        cfg["process"]["terminal"] = False
        cfg["process"]["cwd"] = "/work"
        command = ";\n".join(
            [
                "export HOME=/root",
                "export IFLOW_NON_INTERACTIVE=true",
                f"export AGENT_CR_IFLOW_ENTRYPOINT={shlex.quote(entrypoint)}",
                f"export AGENT_CR_IFLOW_TASK={escaped_task}",
                f"export AGENT_CR_IFLOW_DONE_PATH={shlex.quote(marker_mount_paths['done'])}",
                f"export AGENT_CR_IFLOW_EXIT_PATH={shlex.quote(marker_mount_paths['exit'])}",
                "cd /work",
                f"mkdir -p {shlex.quote(str(SANDBOX_TASK_OUTPUT_DIR))}",
                (
                    "rm -f "
                    f"{shlex.quote(str(output_paths['stdout']))} "
                    f"{shlex.quote(str(output_paths['stderr']))} "
                    f"{shlex.quote(marker_mount_paths['exit'])} "
                    f"{shlex.quote(marker_mount_paths['done'])}"
                ),
                f"touch {shlex.quote(marker_mount_paths['done'])}",
                f"touch {shlex.quote(marker_mount_paths['exit'])}",
                (
                    f"exec {RUNTIME_MOUNT_PATH}/node/bin/node -e {shlex.quote(_IFLOW_INLINE_WRAPPER)} "
                    f"-- {shlex.quote(IFLOW_WRAPPER_ARG)} "
                    f">{shlex.quote(str(output_paths['stdout']))} "
                    f"2>{shlex.quote(str(output_paths['stderr']))}"
                ),
            ]
        )
        cfg["process"]["args"] = [
            "/bin/sh",
            "-lc",
            command,
        ]
        current_env = cfg["process"].get("env", [])
        if not isinstance(current_env, list):
            raise ValueError(f"unsupported process env in {config_path}: {current_env!r}")
        cfg["process"]["env"] = merge_environment_defaults(
            [str(item) for item in current_env],
            [
            f"PATH={RUNTIME_MOUNT_PATH}/global/bin:{RUNTIME_MOUNT_PATH}/node/bin:/usr/local/bin:/usr/bin:/bin",
            "PYTHONUNBUFFERED=1",
            "UV_USE_IO_URING=0",
            "HOME=/root",
            "IFLOW_NON_INTERACTIVE=true",
            ] + [f"{key}={value}" for key, value in self.task_config.options.items()],
        )
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
        assert self.runtime_state_root is not None
        logs_dir = self._resolve_logs_dir(metadata)
        marker_paths = self._task_marker_paths(logs_dir)
        output_paths = self._sandbox_task_output_paths()
        self._stop_requested.clear()
        self._restore_complete_event.clear()
        with self._state_lock:
            self._started_at_monotonic = time.monotonic()
            self._finished_at_monotonic = None
        try:
            logger.info(
                "Waiting for configured iflow task sandbox=%s logs_dir=%s tick_s=%.3f",
                self.sandbox.sandbox_id,
                logs_dir,
                self._tick_seconds,
            )
            logger.debug(
                "Iflow task markers sandbox=%s stdout=%s stderr=%s exit=%s done=%s",
                self.sandbox.sandbox_id,
                output_paths["stdout"],
                output_paths["stderr"],
                marker_paths["exit"],
                marker_paths["done"],
            )
            self._wait_for_task_completion(exit_path=marker_paths["exit"], done_path=marker_paths["done"])
            logger.info("Completed iflow task sandbox=%s", self.sandbox.sandbox_id)
        except Exception as exc:
            logger.error("Iflow task failed sandbox=%s error=%s", self.sandbox.sandbox_id, exc)
            raise
        finally:
            with self._state_lock:
                self._finished_at_monotonic = time.monotonic()
            self.post_task_finish()

    def poll_status(self) -> dict[str, object]:
        actions = self._synthetic_action_count()
        return {
            "agent_type": self.agent_type,
            "state": self._task_state(),
            "total_actions": actions,
            "filesystem_actions": actions,
            "process_actions": actions,
            "network_actions": actions,
            "stateful_actions": actions,
        }

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

    def request_stop(self) -> None:
        self._stop_requested.set()
        self._restore_complete_event.set()
        logger.info("Stop requested for iflow task sandbox=%s", self.sandbox.sandbox_id)

    def on_restore_complete(self) -> None:
        with self._state_lock:
            started_at = self._started_at_monotonic
            if started_at is None:
                baseline = None
            else:
                baseline = max(0, int(self.sandbox.last_status.get("total_actions", 0)))
                self._started_at_monotonic = time.monotonic() - (baseline * self._tick_seconds)
            self._finished_at_monotonic = None
        self._restore_complete_event.set()
        if baseline is None:
            logger.info("Observed restore completion for idle iflow task sandbox=%s", self.sandbox.sandbox_id)
            return
        logger.info(
            "Resumed synthetic iflow progress after restore sandbox=%s baseline_actions=%d",
            self.sandbox.sandbox_id,
            baseline,
        )

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

    def _resolve_logs_dir(self, metadata: dict[str, object]) -> Path:
        raw_logs_dir = metadata.get("logs_dir")
        if not isinstance(raw_logs_dir, str) or not raw_logs_dir:
            raise RuntimeError(f"missing iflow logs dir for sandbox {self.sandbox.sandbox_id}")
        logs_dir = Path(raw_logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def _task_marker_paths(self, logs_dir: Path) -> dict[str, Path]:
        return {
            "exit": logs_dir / "iflow.task.exit",
            "done": logs_dir / "iflow.task.done",
        }

    def _task_marker_mount_paths(self, marker_paths: dict[str, Path]) -> dict[str, str]:
        return {name: f"{LOGS_MOUNT_PATH}/{path.name}" for name, path in marker_paths.items()}

    def _sandbox_task_output_paths(self) -> dict[str, Path]:
        return {
            "stdout": SANDBOX_TASK_OUTPUT_DIR / "iflow.task.stdout",
            "stderr": SANDBOX_TASK_OUTPUT_DIR / "iflow.task.stderr",
        }

    def _runc_state_command(self) -> list[str]:
        assert self.runtime_state_root is not None
        return ["runc", "--root", str(self.runtime_state_root), "state", str(self.sandbox.sandbox_id)]

    def _sandbox_is_live(self) -> bool:
        result = subprocess.run(
            self._runc_state_command(),
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return False
        try:
            payload = json.loads(result.stdout or "{}")
        except json.JSONDecodeError:
            return False
        return str(payload.get("status", "")).lower() not in {"", "stopped", "missing"}

    def _read_marker_int(self, path: Path) -> int | None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise RuntimeError(f"invalid iflow marker {path}: {exc}") from exc

    def _wait_for_task_completion(self, *, exit_path: Path, done_path: Path) -> None:
        saw_live = False
        while True:
            if self._stop_requested.is_set():
                logger.info("Stopping iflow task wait loop sandbox=%s", self.sandbox.sandbox_id)
                return
            if self._task_markers_indicate_completion(exit_path=exit_path, done_path=done_path):
                return
            if self._sandbox_is_live():
                saw_live = True
                time.sleep(min(self.TASK_POLL_INTERVAL_SECONDS, self._tick_seconds))
                continue
            if self._task_markers_indicate_completion(exit_path=exit_path, done_path=done_path):
                return
            if not saw_live:
                logger.error(
                    "Iflow task sandbox=%s stopped before writing completion markers exit_path=%s done_path=%s",
                    self.sandbox.sandbox_id,
                    exit_path,
                    done_path,
                )
                raise RuntimeError(
                    f"iflow sandbox {self.sandbox.sandbox_id} stopped before writing task completion markers"
                )
            logger.warning(
                "Iflow task sandbox=%s became unavailable while waiting; pausing until restore",
                self.sandbox.sandbox_id,
            )
            if not self._wait_for_restore_or_stop():
                return

    def _task_markers_indicate_completion(self, *, exit_path: Path, done_path: Path) -> bool:
        done_raw = self._read_marker_text(done_path)
        if done_raw is not None:
            logger.debug(
                "Observed iflow completion marker sandbox=%s marker=%s value=%s",
                self.sandbox.sandbox_id,
                done_path,
                done_raw,
            )
            return True
        exit_code = self._read_marker_int(exit_path)
        if exit_code is None:
            return False
        if exit_code != 0:
            output_paths = self._sandbox_task_output_paths()
            logger.error(
                "Iflow task exited with failure sandbox=%s exit_code=%d stdout=%s stderr=%s",
                self.sandbox.sandbox_id,
                exit_code,
                output_paths["stdout"],
                output_paths["stderr"],
            )
            raise RuntimeError(f"iflow task failed in sandbox {self.sandbox.sandbox_id} with exit code {exit_code}")
        logger.error(
            "Iflow task exited without completion marker sandbox=%s exit_path=%s done_path=%s",
            self.sandbox.sandbox_id,
            exit_path,
            done_path,
        )
        raise RuntimeError(
            f"iflow task in sandbox {self.sandbox.sandbox_id} exited without writing the completion marker"
        )

    def _read_marker_text(self, path: Path) -> str | None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return raw or None

    def _wait_for_restore_or_stop(self) -> bool:
        wait_started = time.monotonic()
        while True:
            if self._stop_requested.is_set():
                logger.info(
                    "Stop requested while waiting for iflow restore sandbox=%s wait_s=%.3f",
                    self.sandbox.sandbox_id,
                    time.monotonic() - wait_started,
                )
                return False
            if self._restore_complete_event.wait(timeout=self.TASK_POLL_INTERVAL_SECONDS):
                self._restore_complete_event.clear()
                if self._stop_requested.is_set():
                    logger.info(
                        "Stop requested after restore signal for iflow task sandbox=%s wait_s=%.3f",
                        self.sandbox.sandbox_id,
                        time.monotonic() - wait_started,
                    )
                    return False
                logger.info(
                    "Observed restore completion signal for iflow task sandbox=%s wait_s=%.3f",
                    self.sandbox.sandbox_id,
                    time.monotonic() - wait_started,
                )
                return True

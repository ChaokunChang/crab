from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shlex
import threading
import time

from integrations.agents.base import BaseAgent
from integrations.sandboxes.claude_code.harness import (
    CLAUDE_CODE_WRAPPER_ARG,
    CLAUDE_HOME_ROOT_MOUNT_PATH,
    LOGS_MOUNT_PATH,
    RUNTIME_MOUNT_PATH,
    _IO_URING_SECCOMP,
    prepare_claude_code_runtime,
    prepare_claude_code_state,
)
from integrations.sandboxes.runtime.bundle import merge_environment_defaults

logger = logging.getLogger(__name__)

_DEFAULT_DEBUG_LOG_MOUNT_PATH = f"{LOGS_MOUNT_PATH}/claude_code.debug.log"
# Keep the live Claude stdout/stderr sink out of checkpointed mutable state.
# The debug file above still captures Claude's internal logs on the mounted host path.
_DEFAULT_OUTPUT_SINK_PATH = "/dev/null"

# Inline shell wrapper that launches Claude Code CLI and monitors completion.
# Similar to the iFlow inline wrapper but adapted for Claude Code.
_CLAUDE_CODE_INLINE_WRAPPER = """
#!/bin/sh

CLAUDE_BIN="$AGENT_CR_CLAUDE_CODE_BIN"
TASK="$AGENT_CR_CLAUDE_CODE_TASK"
DONE_PATH="$AGENT_CR_CLAUDE_CODE_DONE_PATH"
EXIT_PATH="$AGENT_CR_CLAUDE_CODE_EXIT_PATH"
TASK_CWD="$AGENT_CR_CLAUDE_CODE_CWD"
DEBUG_LOG_PATH="$AGENT_CR_CLAUDE_CODE_DEBUG_LOG_PATH"
OUTPUT_LOG_PATH="$AGENT_CR_CLAUDE_CODE_OUTPUT_LOG_PATH"
BARE_FLAG="$AGENT_CR_CLAUDE_CODE_BARE_FLAG"

cd "$TASK_CWD"

# Run Claude Code in print mode with all permissions bypassed
"$CLAUDE_BIN" \\
    ${BARE_FLAG:+$BARE_FLAG} \\
    --dangerously-skip-permissions \\
    --debug-file "$DEBUG_LOG_PATH" \\
    --model "$ANTHROPIC_MODEL" \\
    -p "$TASK" \\
    > "$OUTPUT_LOG_PATH" 2>&1
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
    echo "done" > "$DONE_PATH"
fi
echo "$EXIT_CODE" > "$EXIT_PATH"

# Keep alive if requested
if [ "$AGENT_CR_CLAUDE_CODE_KEEPALIVE_AFTER_TASK" = "true" ]; then
    exec sleep infinity > /dev/null 2>&1
fi

exit $EXIT_CODE
""".strip()

_INSTALL_AGENT_SETUP_COMMANDS = [
    "if [ -f /installed-agent/setup-env.sh ]; then . /installed-agent/setup-env.sh; fi",
    (
        "wait_for_apt_lock() { "
        "while pgrep -x apt-get >/dev/null 2>&1 || pgrep -x apt >/dev/null 2>&1 || pgrep -x dpkg >/dev/null 2>&1; "
        "do sleep 1; done; "
        "}; "
        "install -d -m 755 /usr/local/bin"
    ),
    (
        "if [ -f /installed-agent/install-agent.sh ]; then "
        ". /installed-agent/install-agent.sh || echo 'INSTALL_FAIL_STATUS'; "
        "elif command -v apt-get >/dev/null 2>&1; then "
        "export DEBIAN_FRONTEND=noninteractive; "
        "wait_for_apt_lock; "
        "apt-get update && "
        "wait_for_apt_lock && "
        "apt-get install -y "
        "curl git wget xz-utils openssh-client patch xauth build-essential dpkg-dev "
        "procps net-tools psmisc; "
        "fi"
    ),
    (
        "if command -v apt-get >/dev/null 2>&1; then "
        "export DEBIAN_FRONTEND=noninteractive; "
        "wait_for_apt_lock; "
        "apt-get update >/dev/null && "
        "wait_for_apt_lock && "
        "apt-get install -y python3-venv python3-pip >/dev/null; "
        "fi"
    ),
]


class ClaudeCodeAgent(BaseAgent):
    agent_type = "claude_code"
    requires_manual_task_launch = True
    requires_network_namespace = True
    DEFAULT_ACTION_TICK_SECONDS = 1
    TASK_POLL_INTERVAL_SECONDS = 0.1
    SANDBOX_LIVENESS_REFRESH_SECONDS = 1.0

    def __init__(
        self,
        sandbox,
        task_description,
        task_config,
        *,
        runtime_state_root=None,
        runtime=None,
        agent_host_dir=None,
        llm_base_url=None,
        telemetry=None,
    ) -> None:
        super().__init__(
            sandbox,
            task_description,
            task_config,
            runtime_state_root=runtime_state_root,
            runtime=runtime,
            agent_host_dir=agent_host_dir,
            llm_base_url=llm_base_url,
            telemetry=telemetry,
        )
        self._tick_seconds = max(
            0.001,
            float(self.task_config.options.get("action_tick_seconds", self.DEFAULT_ACTION_TICK_SECONDS)),
        )
        self._state_lock = threading.Lock()
        self._restore_complete_event = threading.Event()
        self._restore_reactivation_pending = threading.Event()
        self._stop_requested = threading.Event()
        self._started_at_monotonic: float | None = None
        self._finished_at_monotonic: float | None = None

    def _is_replay_mode(self) -> bool:
        return self.sandbox.llm_service_type == "claude_code_trace_replay"

    def _is_compose_replay_mode(self) -> bool:
        return self.sandbox.launch_source == "compose" and self._is_replay_mode()

    def survives_fault_relaunch(self) -> bool:
        return True

    def prepare_sandbox(self) -> None:
        assert self.agent_host_dir is not None
        assert self.llm_base_url is not None
        logger.debug(f"preparing claude_code sandbox in {self.agent_host_dir=}")
        sandbox_root = self.agent_host_dir
        requested_version_raw = self.task_config.options.get("trace_agent_version")
        requested_version = None
        if isinstance(requested_version_raw, str) and requested_version_raw.strip():
            requested_version = requested_version_raw.strip()
        prepared_runtime = prepare_claude_code_runtime(
            work_root=sandbox_root,
            requested_version=requested_version,
            telemetry=self.telemetry,
            sandbox_id=str(self.sandbox.sandbox_id),
        )
        model_name = str(os.environ.get("AGENT_CR_CLAUDE_CODE_MODEL_NAME", "claude-opus-4-6"))
        prepared_state = prepare_claude_code_state(
            work_root=sandbox_root,
            base_url=self.llm_base_url,
            model_name=model_name,
            telemetry=self.telemetry,
            sandbox_id=str(self.sandbox.sandbox_id),
        )
        self.sandbox.launch_metadata["claude_code"] = {
            "runtime_root": str(prepared_runtime.root),
            "claude_home_root": str(prepared_state.home_root),
            "claude_home": str(prepared_state.claude_home),
            "logs_dir": str(prepared_state.logs_dir),
            "claude_bin": prepared_runtime.mounted_claude_bin,
            "model_name": model_name,
            "resolved_version": prepared_runtime.resolved_version,
            "source_binary": str(prepared_runtime.source_binary),
            "runtime_strategy": prepared_runtime.runtime_strategy,
            "supports_bare_flag": prepared_runtime.supports_bare_flag,
            "ignore_process_rules": prepared_runtime.ignore_process_rules,
        }

    def configure_bundle(self) -> None:
        metadata = self.sandbox.launch_metadata.get("claude_code", {})
        if not metadata:
            raise RuntimeError(f"missing claude_code launch metadata for sandbox {self.sandbox.sandbox_id}")
        claude_bin = metadata.get("claude_bin")
        if not isinstance(claude_bin, str) or not claude_bin:
            raise RuntimeError(f"missing claude_code binary path for sandbox {self.sandbox.sandbox_id}")
        model_name = metadata.get("model_name", "claude-opus-4-6")
        resolved_version = metadata.get("resolved_version")
        supports_bare_flag = bool(metadata.get("supports_bare_flag", False))
        logs_dir = self._resolve_logs_dir(metadata)
        marker_paths = self._task_marker_paths(logs_dir)
        marker_mount_paths = self._task_marker_mount_paths(marker_paths)
        escaped_task = shlex.quote(self.task_description.prompt)
        config_path = self.sandbox.bundle_dir / "config.json"
        cfg = json.loads(config_path.read_text(encoding="utf-8"))

        linux_cfg = cfg.get("linux", {})
        linux_cfg["seccomp"] = _IO_URING_SECCOMP
        cfg["linux"] = linux_cfg
        current_env = cfg["process"].get("env", [])
        if not isinstance(current_env, list):
            raise ValueError(f"unsupported process env in {config_path}: {current_env!r}")
        claude_home_root = metadata.get("claude_home_root")
        if not isinstance(claude_home_root, str) or not claude_home_root:
            raise RuntimeError(f"missing claude_code home root for sandbox {self.sandbox.sandbox_id}")

        # Determine the API base URL for the LLM proxy.
        # Claude Code's Anthropic SDK appends /v1/messages to the base URL,
        # so strip any trailing /v1 to avoid double-prefixing.
        api_base_url = self.llm_base_url or ""
        if api_base_url.endswith("/v1"):
            api_base_url = api_base_url[:-3]

        cfg["process"]["env"] = merge_environment_defaults(
            [str(item) for item in current_env],
            [
                (
                    f"PATH={RUNTIME_MOUNT_PATH}:{CLAUDE_HOME_ROOT_MOUNT_PATH}/.local/bin:"
                    "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
                ),
                "PYTHONUNBUFFERED=1",
                "PYTHONDONTWRITEBYTECODE=1",
                "UV_USE_IO_URING=0",
                f"HOME={CLAUDE_HOME_ROOT_MOUNT_PATH}",
                f"ANTHROPIC_BASE_URL={api_base_url}",
                "ANTHROPIC_API_KEY=sk-agent-cr-claude-code",
                f"ANTHROPIC_MODEL={model_name}",
                "CLAUDE_CODE_SIMPLE=1",
                "IS_SANDBOX=1",
                *(
                    [f"AGENT_CR_CLAUDE_CODE_VERSION={resolved_version}"]
                    if isinstance(resolved_version, str) and resolved_version
                    else []
                ),
            ] + [f"{key}={value}" for key, value in self.task_config.options.items()],
        )
        mounts = [
            mount
            for mount in cfg.get("mounts", [])
            if mount.get("destination")
            not in {RUNTIME_MOUNT_PATH, CLAUDE_HOME_ROOT_MOUNT_PATH, LOGS_MOUNT_PATH}
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
                    "destination": CLAUDE_HOME_ROOT_MOUNT_PATH,
                    "type": "bind",
                    "source": claude_home_root,
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

        if self._is_compose_replay_mode():
            cfg["process"]["terminal"] = False
            compose_cwd = self._compose_replay_cwd()
            command = ";\n".join(
                [
                    f"export HOME={shlex.quote(CLAUDE_HOME_ROOT_MOUNT_PATH)}",
                    f"export AGENT_CR_CLAUDE_CODE_BIN={shlex.quote(claude_bin)}",
                    f"export AGENT_CR_CLAUDE_CODE_TASK={escaped_task}",
                    f"export AGENT_CR_CLAUDE_CODE_CWD={shlex.quote(compose_cwd)}",
                    "export AGENT_CR_CLAUDE_CODE_KEEPALIVE_AFTER_TASK=true",
                    f"export AGENT_CR_CLAUDE_CODE_DONE_PATH={shlex.quote(marker_mount_paths['done'])}",
                    f"export AGENT_CR_CLAUDE_CODE_EXIT_PATH={shlex.quote(marker_mount_paths['exit'])}",
                    (
                        "export AGENT_CR_CLAUDE_CODE_DEBUG_LOG_PATH="
                        f"{shlex.quote(_DEFAULT_DEBUG_LOG_MOUNT_PATH)}"
                    ),
                    (
                        "export AGENT_CR_CLAUDE_CODE_OUTPUT_LOG_PATH="
                        f"{shlex.quote(_DEFAULT_OUTPUT_SINK_PATH)}"
                    ),
                    (
                        "export AGENT_CR_CLAUDE_CODE_BARE_FLAG="
                        f"{shlex.quote('--bare' if supports_bare_flag else '')}"
                    ),
                    f"export ANTHROPIC_BASE_URL={shlex.quote(api_base_url)}",
                    "export ANTHROPIC_API_KEY=sk-agent-cr-claude-code",
                    f"export ANTHROPIC_MODEL={shlex.quote(model_name)}",
                    "export CLAUDE_CODE_SIMPLE=1",
                    "export IS_SANDBOX=1",
                    f"cd {shlex.quote(compose_cwd)}",
                    *_INSTALL_AGENT_SETUP_COMMANDS,
                    (
                        "rm -f "
                        f"{shlex.quote(marker_mount_paths['exit'])} "
                        f"{shlex.quote(marker_mount_paths['done'])}"
                    ),
                    (
                        "exec /bin/sh -c "
                        f"{shlex.quote(_CLAUDE_CODE_INLINE_WRAPPER)} "
                        f"{shlex.quote(CLAUDE_CODE_WRAPPER_ARG)} > /dev/null 2>&1"
                    ),
                ]
            )
            cfg["process"]["cwd"] = compose_cwd
            cfg["process"]["args"] = [
                "/bin/sh",
                "-lc",
                command,
            ]
        else:
            cfg["process"]["terminal"] = False
            cfg["process"]["cwd"] = "/work"
            command = ";\n".join(
                [
                    f"export HOME={shlex.quote(CLAUDE_HOME_ROOT_MOUNT_PATH)}",
                    f"export AGENT_CR_CLAUDE_CODE_BIN={shlex.quote(claude_bin)}",
                    f"export AGENT_CR_CLAUDE_CODE_TASK={escaped_task}",
                    "export AGENT_CR_CLAUDE_CODE_CWD=/work",
                    f"export AGENT_CR_CLAUDE_CODE_DONE_PATH={shlex.quote(marker_mount_paths['done'])}",
                    f"export AGENT_CR_CLAUDE_CODE_EXIT_PATH={shlex.quote(marker_mount_paths['exit'])}",
                    (
                        "export AGENT_CR_CLAUDE_CODE_DEBUG_LOG_PATH="
                        f"{shlex.quote(_DEFAULT_DEBUG_LOG_MOUNT_PATH)}"
                    ),
                    (
                        "export AGENT_CR_CLAUDE_CODE_OUTPUT_LOG_PATH="
                        f"{shlex.quote(_DEFAULT_OUTPUT_SINK_PATH)}"
                    ),
                    (
                        "export AGENT_CR_CLAUDE_CODE_BARE_FLAG="
                        f"{shlex.quote('--bare' if supports_bare_flag else '')}"
                    ),
                    f"export ANTHROPIC_BASE_URL={shlex.quote(api_base_url)}",
                    "export ANTHROPIC_API_KEY=sk-agent-cr-claude-code",
                    f"export ANTHROPIC_MODEL={shlex.quote(model_name)}",
                    "export CLAUDE_CODE_SIMPLE=1",
                    "export IS_SANDBOX=1",
                    "cd /work",
                    *_INSTALL_AGENT_SETUP_COMMANDS,
                    (
                        "rm -f "
                        f"{shlex.quote(marker_mount_paths['exit'])} "
                        f"{shlex.quote(marker_mount_paths['done'])}"
                    ),
                    (
                        "exec /bin/sh -c "
                        f"{shlex.quote(_CLAUDE_CODE_INLINE_WRAPPER)} "
                        f"{shlex.quote(CLAUDE_CODE_WRAPPER_ARG)} > /dev/null 2>&1"
                    ),
                ]
            )
            cfg["process"]["args"] = [
                "/bin/sh",
                "-lc",
                command,
            ]
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def extra_launch_metadata(self) -> dict[str, object]:
        metadata = self.sandbox.launch_metadata.get("claude_code", {})
        ignored_path_prefixes = [
            f"{CLAUDE_HOME_ROOT_MOUNT_PATH}/",
            f"{LOGS_MOUNT_PATH}/",
            "/tmp/claude-",
        ]
        for key in ("claude_home_root", "claude_home", "logs_dir"):
            raw_path = metadata.get(key)
            if isinstance(raw_path, str) and raw_path:
                ignored_path_prefixes.append(f"{raw_path.rstrip('/')}/")
        return {
            "host_inspector_ignore_process_rules": metadata.get("ignore_process_rules", []),
            "host_inspector_ignored_path_prefixes": ignored_path_prefixes,
            "rootfs_copy_paths": [],
        }

    def rootfs_init_dirs(self) -> list[str]:
        return super().rootfs_init_dirs() + [
            "opt/claude-code-home",
            "opt/claude-code-home/.claude",
            "opt/claude-code-runtime",
            "opt/claude-code-logs",
        ]

    def perform_task(self) -> None:
        metadata = self.sandbox.launch_metadata.get("claude_code", {})
        assert self.runtime_state_root is not None
        logs_dir = self._resolve_logs_dir(metadata)
        marker_paths = self._task_marker_paths(logs_dir)
        self._stop_requested.clear()
        self._restore_complete_event.clear()
        self._restore_reactivation_pending.clear()
        with self._state_lock:
            self._started_at_monotonic = time.monotonic()
            self._finished_at_monotonic = None
        try:
            logger.info(
                "Waiting for configured claude_code task sandbox=%s logs_dir=%s tick_s=%.3f",
                self.sandbox.sandbox_id,
                logs_dir,
                self._tick_seconds,
            )
            self._wait_for_task_completion(exit_path=marker_paths["exit"], done_path=marker_paths["done"])
            logger.info("Completed claude_code task sandbox=%s", self.sandbox.sandbox_id)
        except Exception as exc:
            logger.error("Claude_code task failed sandbox=%s error=%s", self.sandbox.sandbox_id, exc)
            raise
        finally:
            with self._state_lock:
                self._finished_at_monotonic = time.monotonic()
            self.post_task_finish()

    def poll_status(self) -> dict[str, object]:
        if self._is_compose_replay_mode():
            return self._replay_progress_payload()
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
        self._restore_reactivation_pending.set()
        logger.info("Stop requested for claude_code task sandbox=%s", self.sandbox.sandbox_id)

    def on_restore_complete(self) -> None:
        if self._is_compose_replay_mode():
            self._clear_host_task_markers()
            self._restore_reactivation_pending.set()
            self._restore_complete_event.set()
            logger.info("Observed restore completion for replay claude_code task sandbox=%s", self.sandbox.sandbox_id)
            return
        with self._state_lock:
            started_at = self._started_at_monotonic
            if started_at is None:
                baseline = None
            else:
                baseline = max(0, int(self.sandbox.last_status.get("total_actions", 0)))
                self._started_at_monotonic = time.monotonic() - (baseline * self._tick_seconds)
            self._finished_at_monotonic = None
        self._restore_reactivation_pending.set()
        self._restore_complete_event.set()
        if baseline is None:
            logger.info("Observed restore completion for idle claude_code task sandbox=%s", self.sandbox.sandbox_id)
            return
        logger.info(
            "Resumed synthetic claude_code progress after restore sandbox=%s baseline_actions=%d",
            self.sandbox.sandbox_id,
            baseline,
        )

    # ── Internal ──────────────────────────────────────────────────────

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
        if self._is_compose_replay_mode():
            self._wait_for_replay_action_count(target_actions)
            return
        deadline = time.monotonic() + max(45.0, target_actions * self._tick_seconds * 4.0)
        while time.monotonic() < deadline:
            current_actions = self._synthetic_action_count()
            if current_actions >= target_actions:
                return
            if self._task_state() == "finished":
                raise RuntimeError(
                    f"claude_code task finished before reaching synthetic action count {target_actions}; "
                    f"last observed count was {current_actions}"
                )
            time.sleep(min(0.2, self._tick_seconds))
        raise RuntimeError(f"timed out waiting for claude_code synthetic action count {target_actions}")

    def _resolve_logs_dir(self, metadata: dict[str, object]) -> Path:
        raw_logs_dir = metadata.get("logs_dir")
        if not isinstance(raw_logs_dir, str) or not raw_logs_dir:
            raise RuntimeError(f"missing claude_code logs dir for sandbox {self.sandbox.sandbox_id}")
        logs_dir = Path(raw_logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir

    def _task_marker_paths(self, logs_dir: Path) -> dict[str, Path]:
        return {
            "exit": logs_dir / "claude_code.task.exit",
            "done": logs_dir / "claude_code.task.done",
        }

    def _task_marker_mount_paths(self, marker_paths: dict[str, Path]) -> dict[str, str]:
        return {name: f"{LOGS_MOUNT_PATH}/{path.name}" for name, path in marker_paths.items()}

    def _sandbox_is_live(self) -> bool:
        if self.runtime is None:
            return False
        payload = self.runtime.inspect_runtime(self.sandbox.sandbox_id)
        return payload.is_running

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
            raise RuntimeError(f"invalid claude_code marker {path}: {exc}") from exc

    def _read_marker_text(self, path: Path) -> str | None:
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return None
        return raw or None

    def _wait_for_task_completion(self, *, exit_path: Path, done_path: Path) -> None:
        saw_live = False
        next_liveness_check_at = 0.0
        while True:
            if self._stop_requested.is_set():
                logger.info("Stopping claude_code task wait loop sandbox=%s", self.sandbox.sandbox_id)
                return
            if self._task_markers_indicate_completion(exit_path=exit_path, done_path=done_path):
                return
            sandbox_is_live = True
            now = time.monotonic()
            if not saw_live or now >= next_liveness_check_at:
                sandbox_is_live = self._sandbox_is_live()
                if sandbox_is_live:
                    saw_live = True
                    next_liveness_check_at = time.monotonic() + self.SANDBOX_LIVENESS_REFRESH_SECONDS
            if sandbox_is_live:
                saw_live = True
                time.sleep(min(self.TASK_POLL_INTERVAL_SECONDS, self._tick_seconds))
                continue
            if self._task_markers_indicate_completion(exit_path=exit_path, done_path=done_path):
                return
            if not saw_live:
                logger.error(
                    "Claude_code task sandbox=%s stopped before writing completion markers exit_path=%s done_path=%s",
                    self.sandbox.sandbox_id,
                    exit_path,
                    done_path,
                )
                raise RuntimeError(
                    f"claude_code sandbox {self.sandbox.sandbox_id} stopped before writing task completion markers"
                )
            logger.warning(
                "Claude_code task sandbox=%s became unavailable while waiting; pausing until restore",
                self.sandbox.sandbox_id,
            )
            if not self._wait_for_restore_or_stop():
                return
            next_liveness_check_at = 0.0

    def _task_markers_indicate_completion(self, *, exit_path: Path, done_path: Path) -> bool:
        done_raw = self._read_marker_text(done_path)
        if done_raw is not None:
            logger.debug(
                "Observed claude_code completion marker sandbox=%s marker=%s value=%s",
                self.sandbox.sandbox_id,
                done_path,
                done_raw,
            )
            return True
        exit_code = self._read_marker_int(exit_path)
        if exit_code is None:
            if done_path.exists():
                logger.debug(
                    "Ignoring empty claude_code completion marker until a non-empty marker or exit code appears "
                    "sandbox=%s done_path=%s exit_path=%s",
                    self.sandbox.sandbox_id,
                    done_path,
                    exit_path,
                )
            return False
        if exit_code != 0:
            logger.error(
                "Claude_code task exited with failure sandbox=%s exit_code=%d",
                self.sandbox.sandbox_id,
                exit_code,
            )
            raise RuntimeError(f"claude_code task failed in sandbox {self.sandbox.sandbox_id} with exit code {exit_code}")
        logger.warning(
            "Claude_code task exited cleanly without completion marker; treating exit=0 as completion "
            "sandbox=%s exit_path=%s done_path=%s",
            self.sandbox.sandbox_id,
            exit_path,
            done_path,
        )
        return True

    def _wait_for_restore_or_stop(self) -> bool:
        wait_started = time.monotonic()
        observed_restore_signal = False
        while True:
            if self._stop_requested.is_set():
                logger.info(
                    "Stop requested while waiting for claude_code restore sandbox=%s wait_s=%.3f",
                    self.sandbox.sandbox_id,
                    time.monotonic() - wait_started,
                )
                return False
            if self._restore_reactivation_pending.is_set() and self._sandbox_is_live():
                self._restore_reactivation_pending.clear()
                if observed_restore_signal:
                    logger.info(
                        "Observed restore completion signal for claude_code task sandbox=%s wait_s=%.3f",
                        self.sandbox.sandbox_id,
                        time.monotonic() - wait_started,
                    )
                else:
                    logger.info(
                        "Observed sandbox live again after restore for claude_code task sandbox=%s wait_s=%.3f",
                        self.sandbox.sandbox_id,
                        time.monotonic() - wait_started,
                    )
                return True
            if self._restore_complete_event.wait(timeout=self.TASK_POLL_INTERVAL_SECONDS):
                self._restore_complete_event.clear()
                if self._stop_requested.is_set():
                    logger.info(
                        "Stop requested after restore signal for claude_code task sandbox=%s wait_s=%.3f",
                        self.sandbox.sandbox_id,
                        time.monotonic() - wait_started,
                    )
                    return False
                observed_restore_signal = True

    def post_task_finish(self) -> None:
        if self._is_compose_replay_mode():
            return
        super().post_task_finish()

    def _bundle_process_config(self) -> dict[str, object]:
        config_path = self.sandbox.bundle_dir / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        process = payload.get("process", {})
        if not isinstance(process, dict):
            raise RuntimeError(f"unsupported process config for sandbox {self.sandbox.sandbox_id}")
        return process

    def _compose_replay_cwd(self) -> str:
        process = self._bundle_process_config()
        cwd = process.get("cwd")
        return str(cwd) if isinstance(cwd, str) and cwd else "/app"

    def _clear_host_task_markers(self) -> None:
        metadata = self.sandbox.launch_metadata.get("claude_code", {})
        logs_dir = self._resolve_logs_dir(metadata)
        marker_paths = self._task_marker_paths(logs_dir)
        for path in marker_paths.values():
            try:
                path.unlink()
            except FileNotFoundError:
                continue

    def _replay_router_state(self) -> dict[str, object]:
        base_url = self.sandbox.llm_control_base_url
        if not base_url:
            raise RuntimeError(f"missing llm control base url for sandbox {self.sandbox.sandbox_id}")
        payload = self.wait_for_http_json(
            f"{base_url}/control/state?sandbox_id={self.sandbox.sandbox_id}",
            timeout_s=10.0,
        )
        state = payload.get("state")
        if not isinstance(state, dict):
            return {}
        nested_state = state.get("state")
        if not isinstance(nested_state, dict):
            return {}
        return nested_state

    def _replay_action_count(self) -> int:
        state = self._replay_router_state()
        raw_value = state.get("trace_cursor", 0)
        try:
            return max(0, int(raw_value))
        except (TypeError, ValueError):
            return 0

    def _replay_progress_payload(self) -> dict[str, object]:
        state = self._replay_router_state()
        raw_value = state.get("trace_cursor", 0)
        try:
            actions = max(0, int(raw_value))
        except (TypeError, ValueError):
            actions = 0
        return {
            "agent_type": self.agent_type,
            "state": self._task_state(),
            "total_actions": actions,
            "filesystem_actions": actions,
            "process_actions": actions,
            "network_actions": actions,
            "stateful_actions": actions,
            "replay_total_responses": state.get("total_responses"),
            "replay_is_complete": bool(state.get("is_complete", False)),
            "replay_trace_cursor": actions,
        }

    def _replay_action_wait_timeout_seconds(self, target_actions: int) -> float:
        raw_task_timeout = self.task_config.options.get("max_agent_timeout_sec", 900.0)
        try:
            task_timeout = float(raw_task_timeout)
        except (TypeError, ValueError):
            task_timeout = 900.0
        return max(45.0, task_timeout, target_actions * 10.0)

    def _wait_for_replay_action_count(self, target_actions: int) -> None:
        deadline = time.monotonic() + self._replay_action_wait_timeout_seconds(target_actions)
        last_error: RuntimeError | None = None
        while time.monotonic() < deadline:
            try:
                current_actions = self._replay_action_count()
            except RuntimeError as exc:
                last_error = exc
                if self.sandbox.task_future is not None and self.sandbox.task_future.done():
                    self.sandbox.task_future.result()
                time.sleep(min(0.2, self.TASK_POLL_INTERVAL_SECONDS))
                continue
            if current_actions >= target_actions:
                return
            if self.sandbox.task_future is not None and self.sandbox.task_future.done():
                self.sandbox.task_future.result()
                raise RuntimeError(
                    f"claude_code replay task finished before reaching replay action count {target_actions}; "
                    f"last observed count was {current_actions}"
                )
            time.sleep(min(0.2, self.TASK_POLL_INTERVAL_SECONDS))
        if last_error is not None:
            raise RuntimeError(
                f"timed out waiting for claude_code replay action count {target_actions}: {last_error}"
            )
        raise RuntimeError(f"timed out waiting for claude_code replay action count {target_actions}")

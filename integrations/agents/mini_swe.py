from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from integrations.sandboxes.runtime import bundle as sandbox_bundle
from integrations.sandboxes.swebench import ensure_instance_image_available

from .base import BaseAgent, TaskConfig, TaskDescription

logger = logging.getLogger(__name__)

_SYSTEM_TEMPLATE = """You are a helpful assistant that can interact multiple times with a computer shell to solve programming tasks.
Your response must contain exactly ONE bash code block with ONE command (or commands connected with && or ||).

Include a THOUGHT section before your command where you explain your reasoning process.
Format your response as shown in <format_example>.

<format_example>
THOUGHT: Your reasoning and analysis here

<mswea_bash_command>your_command_here</mswea_bash_command>
</format_example>

Failure to follow these rules will cause your response to be rejected."""

_INSTANCE_TEMPLATE = """<pr_description>
Consider the following PR description:
{task}
</pr_description>

<instructions>
# Task Instructions

## Overview

You're a software engineer interacting continuously with a computer by submitting commands.
You'll be helping implement necessary changes to meet requirements in the PR description.
Your task is specifically to make changes to non-test files in the current directory in order to fix the issue described in the PR description in a way that is general and consistent with the codebase.

<IMPORTANT>This is an interactive process where you will think and issue ONE command, see its result, then think and issue your next command.</IMPORTANT>

For each response:

1. Include a THOUGHT section explaining your reasoning and what you're trying to accomplish
2. Provide exactly ONE bash command to execute

## Important Boundaries

- MODIFY: Regular source code files in /testbed (this is the working directory for all your subsequent commands)
- DO NOT MODIFY: Tests, configuration files (pyproject.toml, setup.cfg, etc.)

## Recommended Workflow

1. Analyze the codebase by finding and reading relevant files
2. Create a script to reproduce the issue
3. Edit the source code to resolve the issue
4. Verify your fix works by running your script again
5. Test edge cases to ensure your fix is robust

## Command Execution Rules

You are operating in an environment where

1. You write a single command
2. The system executes that command in a subshell
3. You see the result
4. You write your next command

Each response should include:

1. A THOUGHT section where you explain your reasoning and plan
2. A single bash code block with your command

Commands must be specified in a single bash XML tag:

<mswea_bash_command>your_command_here</mswea_bash_command>

## Submission

When you've completed your work, you MUST submit your changes as a git patch.
Use this EXACT command to submit:

<mswea_bash_command>echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat patch.txt</mswea_bash_command>
</instructions>"""

_OBSERVATION_WARNING = """<warning>
The output of your last command was too long.
Please try a different command that produces less output.
If you're looking at a file you can try use head, tail or sed to view a smaller number of lines selectively.
If you're using grep or find and it produced too much output, you can use a more selective search pattern.
If you really need to see something from the full command's output, you can redirect output to a file and then search in that file.
</warning>"""
_COMMAND_PATTERN = re.compile(r"<mswea_bash_command>(.*?)</mswea_bash_command>", re.DOTALL)
_OBSERVATION_MAX_CHARS = 10000
_OBSERVATION_HEAD_CHARS = 5000
_OBSERVATION_TAIL_CHARS = 5000
_RESTORE_WAIT_TIMEOUT_S = 300.0
_RETRYABLE_EXEC_ERROR_FRAGMENTS = (
    "container does not exist",
    "container not running",
    "container not found",
    "unable to start container process",
    "failed to exec in container",
    "cannot allocate tty",
)


@dataclass(frozen=True)
class _RecordedCommand:
    action_index: int
    command: str
    argv: tuple[str, ...]
    cwd: str
    env: dict[str, object]
    user: str | None
    timeout_s: float


class MiniSweAgent(BaseAgent):
    agent_type = "mini_swe"

    def __init__(
        self,
        sandbox,
        task_description: TaskDescription,
        task_config: TaskConfig,
        *,
        runtime_state_root: Path | None = None,
        runtime=None,
        sandbox_manager=None,
        agent_host_dir: Path | None = None,
        llm_base_url: str | None = None,
    ) -> None:
        super().__init__(
            sandbox,
            task_description,
            task_config,
            runtime_state_root=runtime_state_root,
            runtime=runtime,
            sandbox_manager=sandbox_manager,
            agent_host_dir=agent_host_dir,
            llm_base_url=llm_base_url,
        )
        self._lock = threading.Lock()
        self._state = "idle"
        self._messages: list[dict[str, str]] = []
        self._completed_actions = 0
        self._command_in_flight = False
        self._submission = ""
        self._error = ""
        self._command_history: list[_RecordedCommand] = []
        self._current_command: _RecordedCommand | None = None
        self._pending_restore_trace_cursor: int | None = None
        self._pending_restore_generation = 0
        self._handled_restore_generation = 0
        self._restore_event = threading.Event()

    def prepare_sandbox(self) -> None:
        if self.sandbox.llm_service_type != "mini_swe_trace_replay":
            raise RuntimeError("MiniSweAgent currently only supports llm_service_type=mini_swe_trace_replay")
        raw_instance_id = self.task_config.options.get("swebench_instance_id")
        if not isinstance(raw_instance_id, str) or not raw_instance_id:
            raise RuntimeError("MiniSweAgent requires task_config.options.swebench_instance_id")
        ensure_instance_image_available(raw_instance_id)

    def configure_bundle(self) -> None:
        config_path = self.sandbox.bundle_dir / "config.json"
        if not config_path.exists():
            return
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        process = cfg.get("process")
        if not isinstance(process, dict):
            raise RuntimeError(f"unsupported process config for sandbox {self.sandbox.sandbox_id}")
        current_env = process.get("env", [])
        if not isinstance(current_env, list):
            raise RuntimeError(f"unsupported process env for sandbox {self.sandbox.sandbox_id}: {current_env!r}")
        env_items = [str(item) for item in current_env]
        env_overrides = [
            "HOME=/root",
            "PAGER=cat",
            "MANPAGER=cat",
            "LESS=-R",
            "PIP_PROGRESS_BAR=off",
            "TQDM_DISABLE=1",
            "PYTHONDONTWRITEBYTECODE=1",
        ]
        if not any(item.startswith("PATH=") for item in env_items):
            env_overrides.insert(1, "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        # Use an absolute sleep path for the long-lived keepalive so CRIU can
        # resolve the executable reliably during restore across SWEBench images.
        process["args"] = ["/bin/sh", "-c", "exec /bin/sleep infinity >/dev/null 2>&1"]
        process["cwd"] = "/testbed"
        process["terminal"] = False
        process["env"] = sandbox_bundle.merge_environment_defaults(
            env_items,
            env_overrides,
        )
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def rootfs_init_dirs(self) -> list[str]:
        dirs = super().rootfs_init_dirs()
        if "testbed" not in dirs:
            dirs.append("testbed")
        return dirs

    def extra_launch_metadata(self) -> dict[str, object]:
        return {
            "host_inspector_ignore_process_rules": [
                {"executable_basename": "sleep"},
            ]
        }

    def perform_task(self) -> None:
        if self.runtime is None:
            raise RuntimeError("MiniSweAgent requires a runtime for sandbox command execution")
        self._stop_requested.clear()
        self._restore_event.clear()
        with self._lock:
            self._state = "running"
            self._messages = self._initial_messages()
            self._completed_actions = 0
            self._command_in_flight = False
            self._submission = ""
            self._error = ""
            self._command_history = []
            self._current_command = None
            self._pending_restore_trace_cursor = None
            self._pending_restore_generation = 0
            self._handled_restore_generation = 0
        self._set_status(self.poll_status())
        try:
            logger.info(
                "Starting mini_swe task sandbox=%s timeout_s=%.1f",
                self.sandbox.sandbox_id,
                self._llm_request_timeout_seconds(),
            )
            logger.debug(
                "Mini_swe task initialized sandbox=%s cwd=%s user=%s initial_messages=%d",
                self.sandbox.sandbox_id,
                self._process_cwd(),
                self._process_user(),
                len(self._messages),
            )
            while not self._stop_requested.is_set():
                self._drain_pending_restore_replays()
                assistant_content = self._next_assistant_content()
                command = self._parse_command(assistant_content)
                result = self._exec_command(command)
                if self._is_submission_result(result):
                    submission = self._submission_from_result(result)
                    with self._lock:
                        self._submission = submission
                        self._state = "finished"
                    payload = self.poll_status()
                    self._set_status(payload)
                    logger.info(
                        "Completed mini_swe task sandbox=%s actions=%d submission_chars=%d",
                        self.sandbox.sandbox_id,
                        int(payload.get("total_actions", 0)),
                        len(submission),
                    )
                    return
                observation = self._format_observation(
                    output=self._merged_exec_output(result),
                    returncode=result.returncode,
                    exception_info="",
                )
                with self._lock:
                    self._messages.append({"role": "user", "content": observation})
                payload = self.poll_status()
                self._record_activity(payload)
            with self._lock:
                if self._state == "running":
                    self._state = "finished"
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._error = str(exc) or exc.__class__.__name__
            self._set_status(self.poll_status())
            logger.error("Mini_swe task failed sandbox=%s error=%s", self.sandbox.sandbox_id, exc)
            raise
        finally:
            self.post_task_finish()

    def post_task_finish(self) -> None:
        if self._is_compose_replay_mode():
            return
        super().post_task_finish()

    def survives_fault_relaunch(self) -> bool:
        return True

    def poll_status(self) -> dict[str, object]:
        with self._lock:
            state = self._state
            completed_actions = self._completed_actions
            command_in_flight = self._command_in_flight
            submission = self._submission
            error = self._error
        replay_state = self._replay_router_state()
        total_responses = replay_state.get("total_responses")
        replay_is_complete = state == "finished"
        if isinstance(total_responses, int) and total_responses > 0:
            replay_is_complete = replay_is_complete or completed_actions >= total_responses
        payload = {
            "agent_type": self.agent_type,
            "state": state,
            "total_actions": completed_actions,
            "filesystem_actions": 0,
            "process_actions": 0,
            "network_actions": 0,
            "stateful_actions": completed_actions,
            "replay_trace_cursor": completed_actions,
            "replay_total_responses": total_responses,
            "replay_is_complete": replay_is_complete,
            "command_in_flight": command_in_flight,
        }
        if submission:
            payload["submission"] = submission
        if error:
            payload["error"] = error
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

    def wait_for_task_ready(self) -> None:
        self._set_status(self.poll_status())

    def on_restore_complete(self) -> None:
        self._restore_event.set()

    def record_restore_trace_cursor(self, trace_cursor: int) -> None:
        with self._lock:
            self._pending_restore_trace_cursor = max(0, int(trace_cursor))
            self._pending_restore_generation += 1

    def _is_compose_replay_mode(self) -> bool:
        return self.sandbox.launch_source == "compose" and self.sandbox.llm_service_type == "mini_swe_trace_replay"

    def _initial_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_TEMPLATE},
            {"role": "user", "content": _INSTANCE_TEMPLATE.format(task=self.task_description.prompt)},
        ]

    def _next_assistant_content(self) -> str:
        if not self.llm_base_url:
            raise RuntimeError(f"missing llm base url for sandbox {self.sandbox.sandbox_id}")
        with self._lock:
            messages = list(self._messages)
        request_payload = {
            "model": "mini-swe-trace-replay",
            "messages": messages,
        }
        request = urllib.request.Request(
            f"{self.llm_base_url}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Agent-Sandbox-Id": str(self.sandbox.sandbox_id),
            },
        )
        with urllib.request.urlopen(request, timeout=self._llm_request_timeout_seconds()) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("mini_swe llm replay returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("mini_swe llm replay returned a malformed assistant message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("mini_swe llm replay returned empty assistant content")
        with self._lock:
            self._messages.append({"role": "assistant", "content": content})
        return content

    def _llm_request_timeout_seconds(self) -> float:
        raw_timeout = self.task_config.options.get("max_agent_timeout_sec", 900.0)
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_s = 900.0
        return max(60.0, timeout_s)

    def _parse_command(self, assistant_content: str) -> str:
        commands = [item.strip() for item in _COMMAND_PATTERN.findall(assistant_content)]
        if len(commands) != 1 or not commands[0]:
            raise RuntimeError(f"MiniSweAgent expected exactly one mswea_bash_command tag, found {len(commands)}")
        return commands[0]

    def _bundle_process_config(self) -> dict[str, object]:
        config_path = self.sandbox.bundle_dir / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        process = payload.get("process", {})
        if not isinstance(process, dict):
            raise RuntimeError(f"unsupported process config for sandbox {self.sandbox.sandbox_id}")
        return process

    def _process_cwd(self) -> str:
        process = self._bundle_process_config()
        cwd = process.get("cwd")
        return str(cwd) if isinstance(cwd, str) and cwd else "/testbed"

    def _process_env(self) -> dict[str, object]:
        process = self._bundle_process_config()
        env_items = process.get("env", [])
        defaults = [str(item) for item in env_items] if isinstance(env_items, list) else []
        merged = sandbox_bundle.merge_environment_defaults(
            defaults,
            [
                "PAGER=cat",
                "MANPAGER=cat",
                "LESS=-R",
                "PIP_PROGRESS_BAR=off",
                "TQDM_DISABLE=1",
            ],
        )
        env_map: dict[str, object] = {}
        for item in merged:
            key, _, value = item.partition("=")
            env_map[key] = value
        return env_map

    def _process_user(self) -> str | None:
        process = self._bundle_process_config()
        user = process.get("user")
        if not isinstance(user, dict):
            return None
        uid = user.get("uid")
        gid = user.get("gid")
        if uid is None:
            return None
        return f"{uid}:{gid}" if gid is not None else str(uid)

    def _exec_command(self, command: str):
        assert self.runtime is not None
        raw_timeout = self.task_config.options.get("max_agent_timeout_sec", 900.0)
        try:
            timeout_s = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            timeout_s = 900.0
        command_record = _RecordedCommand(
            action_index=self._next_action_index(),
            command=command,
            argv=("bash", "-c", command),
            cwd=self._process_cwd(),
            env=self._process_env(),
            user=self._process_user(),
            timeout_s=timeout_s,
        )
        with self._lock:
            self._command_in_flight = True
            self._current_command = command_record
        self._restore_event.clear()
        self._set_status(self.poll_status())
        logger.debug(
            "Executing mini_swe command sandbox=%s action=%d cwd=%s command=%s",
            self.sandbox.sandbox_id,
            command_record.action_index,
            command_record.cwd,
            command_record.command,
        )
        try:
            while True:
                result = self._run_command_record(command_record, resilient=False)
                if not self._is_retryable_exec_failure(result):
                    self._record_completed_command(command_record)
                    logger.debug(
                        "Completed mini_swe command sandbox=%s action=%d returncode=%d",
                        self.sandbox.sandbox_id,
                        command_record.action_index,
                        int(result.returncode),
                    )
                    return result
                logger.warning(
                    "Retrying mini_swe command after sandbox interruption sandbox=%s action=%d command=%s",
                    self.sandbox.sandbox_id,
                    command_record.action_index,
                    command_record.command,
                )
                self._wait_for_restore_completion()
                self._drain_pending_restore_replays(max_replay_action_index=command_record.action_index - 1)
        finally:
            with self._lock:
                self._command_in_flight = False
                self._current_command = None
            self._set_status(self.poll_status())

    def _next_action_index(self) -> int:
        with self._lock:
            return self._completed_actions + 1

    def _record_completed_command(self, command_record: _RecordedCommand) -> None:
        with self._lock:
            self._completed_actions += 1
            self._command_history.append(command_record)

    def _wait_for_restore_completion(self) -> None:
        if self._restore_event.wait(timeout=_RESTORE_WAIT_TIMEOUT_S):
            self._restore_event.clear()
            return
        raise RuntimeError(
            f"timed out waiting for restore completion for sandbox {self.sandbox.sandbox_id} after "
            f"{_RESTORE_WAIT_TIMEOUT_S:.1f}s"
        )

    def _drain_pending_restore_replays(self, *, max_replay_action_index: int | None = None) -> None:
        while True:
            with self._lock:
                pending_generation = self._pending_restore_generation
                if pending_generation <= self._handled_restore_generation:
                    return
                restore_trace_cursor = self._pending_restore_trace_cursor
                replay_target = self._completed_actions
                if max_replay_action_index is not None:
                    replay_target = min(replay_target, max_replay_action_index)
                if restore_trace_cursor is None or replay_target <= restore_trace_cursor:
                    self._handled_restore_generation = pending_generation
                    return
                commands = [
                    record
                    for record in self._command_history
                    if restore_trace_cursor < record.action_index <= replay_target
                ]
            logger.info(
                "Replaying mini_swe commands after restore sandbox=%s restore_trace_cursor=%s replay_target=%s count=%s",
                self.sandbox.sandbox_id,
                restore_trace_cursor,
                replay_target,
                len(commands),
            )
            for record in commands:
                replay_result = self._run_command_record(record, resilient=True)
                if int(replay_result.returncode) != 0:
                    logger.warning(
                        "Replayed mini_swe command returned non-zero sandbox=%s action=%d returncode=%d command=%s",
                        self.sandbox.sandbox_id,
                        record.action_index,
                        int(replay_result.returncode),
                        record.command,
                    )
            with self._lock:
                if self._pending_restore_generation == pending_generation:
                    self._handled_restore_generation = pending_generation
                    return

    def _run_command_record(self, command_record: _RecordedCommand, *, resilient: bool):
        assert self.runtime is not None
        exec_fn = self.runtime.resilient_exec if resilient else self.runtime.exec
        return exec_fn(
            self.sandbox.sandbox_id,
            list(command_record.argv),
            cwd=command_record.cwd,
            env=dict(command_record.env),
            user=command_record.user,
            timeout_s=command_record.timeout_s,
            capture_output=True,
        )

    def _is_retryable_exec_failure(self, result) -> bool:
        if int(result.returncode) == 0:
            return False
        stdout = "" if result.stdout is None else str(result.stdout)
        stderr = "" if result.stderr is None else str(result.stderr)
        merged = f"{stdout}\n{stderr}".lower()
        return any(fragment in merged for fragment in _RETRYABLE_EXEC_ERROR_FRAGMENTS)

    def _is_submission_result(self, result) -> bool:
        if int(result.returncode) != 0:
            return False
        lines = result.stdout.lstrip().splitlines()
        return bool(lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")

    def _submission_from_result(self, result) -> str:
        lines = result.stdout.lstrip().splitlines(keepends=True)
        if not lines:
            return ""
        return "".join(lines[1:])

    @staticmethod
    def _merged_exec_output(result) -> str:
        stdout = "" if result.stdout is None else result.stdout
        stderr = "" if result.stderr is None else result.stderr
        if not stderr:
            return stdout
        return f"{stdout}{stderr}"

    def _format_observation(self, *, output: str, returncode: int, exception_info: str) -> str:
        parts: list[str] = []
        if exception_info:
            parts.append(f"<exception>{exception_info}</exception>")
        parts.append(f"<returncode>{returncode}</returncode>")
        if len(output) < _OBSERVATION_MAX_CHARS:
            parts.append(f"<output>\n{output}</output>")
            return "\n".join(parts)
        elided = len(output) - _OBSERVATION_MAX_CHARS
        parts.extend(
            [
                _OBSERVATION_WARNING,
                "<output_head>",
                output[:_OBSERVATION_HEAD_CHARS],
                "</output_head>",
                f"<elided_chars>\n{elided} characters elided\n</elided_chars>",
                "<output_tail>",
                output[-_OBSERVATION_TAIL_CHARS:],
                "</output_tail>",
            ]
        )
        return "\n".join(parts)

    def _replay_router_state(self) -> dict[str, Any]:
        if not self.sandbox.llm_control_base_url:
            return {}
        try:
            payload = self.wait_for_http_json(
                f"{self.sandbox.llm_control_base_url}/control/state?sandbox_id={self.sandbox.sandbox_id}",
                timeout_s=5.0,
            )
        except Exception:
            logger.debug("Failed to read mini_swe replay router state sandbox=%s", self.sandbox.sandbox_id, exc_info=True)
            return {}
        state = payload.get("state")
        if not isinstance(state, dict):
            return {}
        nested_state = state.get("state")
        if not isinstance(nested_state, dict):
            return {}
        return nested_state

    def _wait_for_action_count(self, target_actions: int) -> None:
        deadline = time.monotonic() + self._action_wait_timeout_seconds(target_actions)
        while time.monotonic() < deadline:
            payload = self.poll_status()
            if int(payload.get("total_actions", 0)) >= target_actions:
                return
            if self.sandbox.task_future is not None and self.sandbox.task_future.done():
                self.sandbox.task_future.result()
                raise RuntimeError(
                    f"mini_swe replay task finished before reaching replay action count {target_actions}; "
                    f"last observed count was {int(payload.get('total_actions', 0))}"
                )
            time.sleep(0.2)
        raise RuntimeError(f"timed out waiting for mini_swe replay action count {target_actions}")

    def _action_wait_timeout_seconds(self, target_actions: int) -> float:
        raw_task_timeout = self.task_config.options.get("max_agent_timeout_sec", 900.0)
        try:
            task_timeout = float(raw_task_timeout)
        except (TypeError, ValueError):
            task_timeout = 900.0
        return max(45.0, task_timeout, target_actions * 10.0)

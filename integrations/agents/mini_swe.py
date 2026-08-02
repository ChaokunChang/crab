from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
import json
import logging
import re
import threading
import time
import urllib.request
import uuid
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
_SPEC_FUTURE_DRAIN_TIMEOUT_S = 5.0
_SPEC_REQUEST_GROUP_KIND = "spec_pair"
_SPEC_ROLE_DRAFT = "draft"
_SPEC_ROLE_ORACLE = "oracle"
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


@dataclass(frozen=True)
class _AssistantTurn:
    role: str
    pair_id: str | None
    content: str
    command: str


@dataclass(frozen=True)
class _SpeculativeForkHandle:
    handle: Any | None
    created: bool = False
    reused: bool = False


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
        telemetry=None,
        speculation_controller=None,
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
            telemetry=telemetry,
        )
        self._speculation_controller = speculation_controller
        self._spec_executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
            max_workers=4,
            thread_name_prefix=f"mini-swe-spec-{self.sandbox.sandbox_id}",
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
        self._spec_total_turns = 0
        self._spec_accept_count = 0
        self._spec_reject_count = 0
        self._spec_saved_ms = 0.0
        self._spec_penalty_ms = 0.0
        self._spec_hidden_penalty_ms = 0.0
        self._spec_fork_create_count = 0
        self._spec_fork_reuse_count = 0
        self._active_spec_futures: set[Future[Any]] = set()

    def prepare_sandbox(self) -> None:
        if self.sandbox.llm_service_type not in {"mini_swe_trace_replay", "mini_swe_spec_trace_replay"}:
            raise RuntimeError(
                "MiniSweAgent currently only supports llm_service_type in "
                "{'mini_swe_trace_replay', 'mini_swe_spec_trace_replay'}"
            )
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
            self._spec_total_turns = 0
            self._spec_accept_count = 0
            self._spec_reject_count = 0
            self._spec_saved_ms = 0.0
            self._spec_penalty_ms = 0.0
            self._spec_hidden_penalty_ms = 0.0
            self._spec_fork_create_count = 0
            self._spec_fork_reuse_count = 0
        self._set_status(self.poll_status())
        turn_index = 0
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
                turn_index += 1
                with self._lock:
                    completed_actions = self._completed_actions
                    message_count = len(self._messages)
                    pending_restore_generation = self._pending_restore_generation
                    handled_restore_generation = self._handled_restore_generation
                logger.debug(
                    (
                        "Mini_swe turn start sandbox=%s turn=%d mode=%s completed_actions=%d "
                        "messages=%d restore_generation=%d handled_restore_generation=%d"
                    ),
                    self.sandbox.sandbox_id,
                    turn_index,
                    "spec" if self._is_spec_replay_mode() else "serial",
                    completed_actions,
                    message_count,
                    pending_restore_generation,
                    handled_restore_generation,
                )
                self._drain_pending_restore_replays()
                if self._is_spec_replay_mode():
                    result = self._spec_perform_step()
                else:
                    assistant_turn = self._request_assistant_turn()
                    logger.debug(
                        "Mini_swe turn assistant response sandbox=%s turn=%d role=%s command=%s",
                        self.sandbox.sandbox_id,
                        turn_index,
                        assistant_turn.role,
                        assistant_turn.command,
                    )
                    self._append_assistant_message(assistant_turn.content)
                    result = self._exec_command(assistant_turn.command)
                output = self._merged_exec_output(result)
                logger.debug(
                    "Mini_swe turn command completed sandbox=%s turn=%d returncode=%d output_chars=%d",
                    self.sandbox.sandbox_id,
                    turn_index,
                    int(result.returncode),
                    len(output),
                )
                if self._is_submission_result(result):
                    submission = self._submission_from_result(result)
                    logger.debug(
                        "Mini_swe submission detected sandbox=%s turn=%d submission_chars=%d",
                        self.sandbox.sandbox_id,
                        turn_index,
                        len(submission),
                    )
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
                    output=output,
                    returncode=result.returncode,
                    exception_info="",
                )
                with self._lock:
                    self._messages.append({"role": "user", "content": observation})
                payload = self.poll_status()
                self._record_activity(payload)
                logger.debug(
                    "Mini_swe turn end sandbox=%s turn=%d total_actions=%d state=%s",
                    self.sandbox.sandbox_id,
                    turn_index,
                    int(payload.get("total_actions", 0)),
                    payload.get("state"),
                )
            with self._lock:
                if self._state == "running":
                    self._state = "finished"
            logger.debug(
                "Mini_swe task loop exited due to stop request sandbox=%s turns=%d",
                self.sandbox.sandbox_id,
                turn_index,
            )
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
        if self._spec_executor is not None:
            self._wait_for_active_spec_futures(timeout_s=_SPEC_FUTURE_DRAIN_TIMEOUT_S)
            self._spec_executor.shutdown(wait=False, cancel_futures=True)
            self._spec_executor = None
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
            spec_total_turns = self._spec_total_turns
            spec_accept_count = self._spec_accept_count
            spec_reject_count = self._spec_reject_count
            spec_saved_ms = self._spec_saved_ms
            spec_penalty_ms = self._spec_penalty_ms
            spec_hidden_penalty_ms = self._spec_hidden_penalty_ms
            spec_fork_create_count = self._spec_fork_create_count
            spec_fork_reuse_count = self._spec_fork_reuse_count
        replay_state = self._replay_router_state()
        total_responses = replay_state.get("total_responses")
        replay_is_complete = state == "finished"
        if isinstance(total_responses, int) and total_responses > 0:
            replay_is_complete = replay_is_complete or completed_actions >= total_responses
        spec_net_gain_ms = spec_saved_ms - spec_penalty_ms
        spec_total_decisions = spec_accept_count + spec_reject_count
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
            "spec_total_turns": spec_total_turns,
            "spec_accept_count": spec_accept_count,
            "spec_reject_count": spec_reject_count,
            "spec_accept_rate": 0.0 if spec_total_decisions <= 0 else spec_accept_count / spec_total_decisions,
            "spec_saved_ms": spec_saved_ms,
            "spec_penalty_ms": spec_penalty_ms,
            "spec_hidden_penalty_ms": spec_hidden_penalty_ms,
            "spec_net_gain_ms": spec_net_gain_ms,
            "spec_fork_create_count": spec_fork_create_count,
            "spec_fork_reuse_count": spec_fork_reuse_count,
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
        return self.sandbox.launch_source == "compose" and self.sandbox.llm_service_type in {
            "mini_swe_trace_replay",
            "mini_swe_spec_trace_replay",
        }

    def _is_spec_replay_mode(self) -> bool:
        return self.sandbox.llm_service_type == "mini_swe_spec_trace_replay"

    def _initial_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": _SYSTEM_TEMPLATE},
            {"role": "user", "content": _INSTANCE_TEMPLATE.format(task=self.task_description.prompt)},
        ]

    def _append_assistant_message(self, content: str) -> None:
        with self._lock:
            self._messages.append({"role": "assistant", "content": content})

    def _request_assistant_turn(
        self,
        *,
        role: str | None = None,
        pair_id: str | None = None,
    ) -> _AssistantTurn:
        if not self.llm_base_url:
            raise RuntimeError(f"missing llm base url for sandbox {self.sandbox.sandbox_id}")
        with self._lock:
            messages = list(self._messages)
        logger.debug(
            "Requesting assistant turn sandbox=%s role=%s pair_id=%s messages=%d timeout_s=%.1f",
            self.sandbox.sandbox_id,
            _SPEC_ROLE_ORACLE if role is None else role,
            "" if pair_id is None else pair_id,
            len(messages),
            self._llm_request_timeout_seconds(),
        )
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
        if pair_id:
            request.add_header("X-Crab-Spec-Pair-Id", pair_id)
        if role:
            request.add_header("X-Crab-Spec-Role", role)
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
        parsed_command = self._parse_command(content)
        resolved_role = _SPEC_ROLE_ORACLE if role is None else role
        logger.debug(
            "Received assistant turn sandbox=%s role=%s pair_id=%s content_chars=%d command=%s",
            self.sandbox.sandbox_id,
            resolved_role,
            "" if pair_id is None else pair_id,
            len(content),
            parsed_command,
        )
        return _AssistantTurn(
            role=resolved_role,
            pair_id=pair_id,
            content=content,
            command=parsed_command,
        )

    def _submit_spec_future(self, fn, /, *args, **kwargs) -> Future[Any]:
        if self._spec_executor is None:
            raise RuntimeError("speculative executor is unavailable")
        future = self._spec_executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._active_spec_futures.add(future)

        def _discard(completed_future: Future[Any]) -> None:
            with self._lock:
                self._active_spec_futures.discard(completed_future)

        future.add_done_callback(_discard)
        return future

    def _wait_for_active_spec_futures(self, *, timeout_s: float) -> None:
        with self._lock:
            active = list(self._active_spec_futures)
        if not active:
            return
        done, not_done = wait(active, timeout=timeout_s)
        for future in done:
            try:
                future.result()
            except Exception as exc:
                logger.debug(
                    "Speculative future completed with error during task shutdown sandbox=%s error=%s",
                    self.sandbox.sandbox_id,
                    exc,
                )
        if not_done:
            logger.warning(
                "Timed out waiting for speculative futures during task shutdown sandbox=%s pending=%d timeout_s=%.1f",
                self.sandbox.sandbox_id,
                len(not_done),
                timeout_s,
            )

    def _spec_perform_step(self):
        step_started = time.perf_counter()
        pair_id = uuid.uuid4().hex
        logger.debug(
            "Starting speculative step sandbox=%s pair_id=%s",
            self.sandbox.sandbox_id,
            pair_id,
        )
        draft_future = self._submit_spec_future(
            self._request_assistant_turn,
            role=_SPEC_ROLE_DRAFT,
            pair_id=pair_id,
        )
        oracle_future = self._submit_spec_future(
            self._request_assistant_turn,
            role=_SPEC_ROLE_ORACLE,
            pair_id=pair_id,
        )
        done, _ = wait([draft_future, oracle_future], return_when=FIRST_COMPLETED)
        logger.debug(
            "Speculative assistant race completed sandbox=%s pair_id=%s draft_done=%s oracle_done=%s",
            self.sandbox.sandbox_id,
            pair_id,
            draft_future.done(),
            oracle_future.done(),
        )
        if oracle_future in done:
            try:
                oracle_turn = oracle_future.result()
            except Exception:
                wait([draft_future], timeout=_SPEC_FUTURE_DRAIN_TIMEOUT_S)
                raise
            logger.debug(
                "Oracle turn arrived first; falling back to non-spec command execution sandbox=%s pair_id=%s command=%s",
                self.sandbox.sandbox_id,
                pair_id,
                oracle_turn.command,
            )
            self._invalidate_speculative_fork()
            self._append_assistant_message(oracle_turn.content)
            result = self._exec_command(oracle_turn.command)
            self._record_spec_step_metrics(
                pair_id=pair_id,
                accepted=False,
                draft_first=False,
                fork_created=False,
                fork_reused=False,
                fork_restore_ms=0.0,
                speculative_exec_ms=0.0,
                saved_ms=0.0,
                penalty_ms=0.0,
                hidden_penalty_ms=0.0,
                current_state_changed=False,
                fork_state_changed=False,
            )
            return result

        try:
            draft_turn = draft_future.result()
            logger.debug(
                "Draft turn ready sandbox=%s pair_id=%s command=%s",
                self.sandbox.sandbox_id,
                pair_id,
                draft_turn.command,
            )
        except Exception:
            logger.debug(
                "Draft turn failed; waiting for oracle and falling back sandbox=%s pair_id=%s",
                self.sandbox.sandbox_id,
                pair_id,
                exc_info=True,
            )
            oracle_turn = oracle_future.result()
            self._invalidate_speculative_fork()
            self._append_assistant_message(oracle_turn.content)
            result = self._exec_command(oracle_turn.command)
            self._record_spec_step_metrics(
                pair_id=pair_id,
                accepted=False,
                draft_first=False,
                fork_created=False,
                fork_reused=False,
                fork_restore_ms=0.0,
                speculative_exec_ms=0.0,
                saved_ms=0.0,
                penalty_ms=0.0,
                hidden_penalty_ms=0.0,
                current_state_changed=False,
                fork_state_changed=False,
            )
            return result
        fork_future: Future[_SpeculativeForkHandle] | None = None
        if self._speculation_controller is not None and self._spec_executor is not None:
            logger.debug(
                "Submitting speculative fork preparation sandbox=%s pair_id=%s",
                self.sandbox.sandbox_id,
                pair_id,
            )
            fork_future = self._submit_spec_future(self._ensure_speculative_fork)

        fork_handle = _SpeculativeForkHandle(handle=None)
        fork_restore_ms = 0.0
        if fork_future is not None:
            fork_wait_started = time.perf_counter()
            wait([fork_future, oracle_future], return_when=FIRST_COMPLETED)
            fork_restore_ms = (time.perf_counter() - fork_wait_started) * 1000.0 if fork_future.done() else 0.0
            logger.debug(
                "Speculative fork/oracle race completed sandbox=%s pair_id=%s fork_done=%s oracle_done=%s fork_restore_ms=%.2f",
                self.sandbox.sandbox_id,
                pair_id,
                fork_future.done(),
                oracle_future.done(),
                fork_restore_ms,
            )
            if oracle_future.done() and not fork_future.done():
                oracle_turn = oracle_future.result()
                logger.debug(
                    "Oracle turn finished before fork restore; using non-spec execution sandbox=%s pair_id=%s command=%s",
                    self.sandbox.sandbox_id,
                    pair_id,
                    oracle_turn.command,
                )
                self._invalidate_speculative_fork()
                self._append_assistant_message(oracle_turn.content)
                result = self._exec_command(oracle_turn.command)
                self._record_spec_step_metrics(
                    pair_id=pair_id,
                    accepted=False,
                    draft_first=True,
                    fork_created=False,
                    fork_reused=False,
                    fork_restore_ms=0.0,
                    speculative_exec_ms=0.0,
                    saved_ms=0.0,
                    penalty_ms=0.0,
                    hidden_penalty_ms=0.0,
                    current_state_changed=False,
                    fork_state_changed=False,
                )
                return result
            if fork_future.done():
                fork_handle = fork_future.result()
                logger.debug(
                    "Speculative fork prepared sandbox=%s pair_id=%s has_handle=%s created=%s reused=%s",
                    self.sandbox.sandbox_id,
                    pair_id,
                    fork_handle.handle is not None,
                    fork_handle.created,
                    fork_handle.reused,
                )
        
        if fork_handle.handle is None:
            oracle_turn = oracle_future.result()
            logger.debug(
                "No speculative fork available; executing oracle command directly sandbox=%s pair_id=%s command=%s",
                self.sandbox.sandbox_id,
                pair_id,
                oracle_turn.command,
            )
            self._invalidate_speculative_fork()
            self._append_assistant_message(oracle_turn.content)
            result = self._exec_command(oracle_turn.command)
            self._record_spec_step_metrics(
                pair_id=pair_id,
                accepted=False,
                draft_first=True,
                fork_created=False,
                fork_reused=False,
                fork_restore_ms=fork_restore_ms,
                speculative_exec_ms=0.0,
                saved_ms=0.0,
                penalty_ms=0.0,
                hidden_penalty_ms=0.0,
                current_state_changed=False,
                fork_state_changed=False,
            )
            return result

        speculative_record = self._build_command_record(draft_turn.command)
        speculative_started = time.perf_counter()
        # Captured strictly inside the worker thread (via a wrapper whose
        # finally runs before set_result), so saved_ms below can read an
        # accurate draft-exec finish time without a callback race.
        draft_exec_done_at: list[float] = []

        def _run_draft_exec_timed() -> Any:
            try:
                return self._run_command_record_for_sandbox(
                    speculative_record,
                    fork_handle.handle.sandbox_id,
                )
            finally:
                draft_exec_done_at.append(time.perf_counter())

        speculative_exec_future = self._submit_spec_future(_run_draft_exec_timed)

        oracle_turn = oracle_future.result()
        oracle_ready_at = time.perf_counter()
        commands_match = self._commands_match(draft_turn.command, oracle_turn.command)
        logger.debug(
            "Speculative command comparison sandbox=%s pair_id=%s commands_match=%s draft_command=%s oracle_command=%s",
            self.sandbox.sandbox_id,
            pair_id,
            commands_match,
            draft_turn.command,
            oracle_turn.command,
        )
        accepted = False
        current_state_changed = False
        fork_state_changed = False
        saved_ms = 0.0
        penalty_ms = 0.0
        hidden_penalty_ms = 0.0
        reuse_candidate_value: bool | None = None
        if commands_match:
            speculative_result = speculative_exec_future.result()
            speculative_exec_ms = (time.perf_counter() - speculative_started) * 1000.0
            if self._is_retryable_exec_failure(speculative_result):
                logger.warning(
                    "Speculative command hit retryable sandbox interruption; falling back to active execution "
                    "sandbox=%s pair_id=%s command=%s",
                    self.sandbox.sandbox_id,
                    pair_id,
                    oracle_turn.command,
                )
                self._append_assistant_message(oracle_turn.content)
                result = self._exec_command(oracle_turn.command)
                # Retryable speculative failure: the fork was interrupted
                # mid-exec so its post-action state is undefined. Never cache.
                reuse_candidate_value = False
                current_state_changed, fork_state_changed = self._finalize_speculative_fork(
                    accepted=False,
                    fork=fork_handle.handle,
                    reuse_candidate=reuse_candidate_value,
                )
                penalty_ms = max(0.0, speculative_exec_ms)
                hidden_penalty_ms = max(0.0, speculative_exec_ms)
            else:
                accepted = True
                self._append_assistant_message(oracle_turn.content)
                committed_record = self._build_command_record(
                    oracle_turn.command,
                    action_index=speculative_record.action_index,
                )
                self._record_completed_command(committed_record)
                result = speculative_result
                if self._speculation_controller is not None:
                    self._promote_speculative_fork(fork_handle.handle)
                # After promotion, `fork_handle.handle` points at the
                # pre-command (old active) sandbox. Reuse is safe iff the
                # executed command was a no-op on both sides (so both
                # sandboxes still sit at the common pre-command state).
                reuse_candidate_value = True
                current_state_changed, fork_state_changed = self._finalize_speculative_fork(
                    accepted=accepted,
                    fork=fork_handle.handle,
                    reuse_candidate=reuse_candidate_value,
                )
                # Honest wall-clock savings: the non-spec path would have
                # started command execution only after the oracle reply
                # arrived, so the overlap window we actually saved is
                # bounded by both how long the draft exec took and how long
                # the oracle made us wait for it.
                draft_exec_done_time = (
                    draft_exec_done_at[0] if draft_exec_done_at else time.perf_counter()
                )
                draft_exec_duration_ms = max(
                    0.0, (draft_exec_done_time - speculative_started) * 1000.0
                )
                oracle_wait_ms = max(
                    0.0, (oracle_ready_at - speculative_started) * 1000.0
                )
                saved_ms = max(0.0, min(draft_exec_duration_ms, oracle_wait_ms))
                logger.debug(
                    "Speculative command accepted sandbox=%s pair_id=%s speculative_exec_ms=%.2f "
                    "draft_exec_duration_ms=%.2f oracle_wait_ms=%.2f saved_ms=%.2f",
                    self.sandbox.sandbox_id,
                    pair_id,
                    speculative_exec_ms,
                    draft_exec_duration_ms,
                    oracle_wait_ms,
                    saved_ms,
                )
        else:
            result = self._exec_command(oracle_turn.command)
            oracle_finished = time.perf_counter()
            speculative_exec_ms = (oracle_finished - speculative_started) * 1000.0
            eager_cleanup = bool(
                self._speculation_controller is not None
                and getattr(self._speculation_controller, "eager_cleanup_on_reject", False)
            )
            if eager_cleanup:
                # Tear the fork down immediately so the draft exec
                # subprocess fails and stops consuming host CPU. The
                # fork's post-action state is undefined once destroyed,
                # so reuse is never safe in this branch.
                reuse_candidate_value = False
                current_state_changed, fork_state_changed = self._finalize_speculative_fork(
                    accepted=accepted,
                    fork=fork_handle.handle,
                    reuse_candidate=reuse_candidate_value,
                )
                drain_done, _ = wait(
                    [speculative_exec_future], timeout=_SPEC_FUTURE_DRAIN_TIMEOUT_S
                )
                if drain_done:
                    try:
                        speculative_exec_future.result()
                    except Exception as exc:
                        logger.debug(
                            "Rejected speculative command exited after eager cleanup sandbox=%s pair_id=%s error=%s",
                            self.sandbox.sandbox_id,
                            pair_id,
                            exc,
                        )
                else:
                    logger.debug(
                        "Rejected speculative future still running after eager cleanup drain "
                        "sandbox=%s pair_id=%s timeout_s=%.1f",
                        self.sandbox.sandbox_id,
                        pair_id,
                        _SPEC_FUTURE_DRAIN_TIMEOUT_S,
                    )
                hidden_penalty_ms = max(
                    0.0, (time.perf_counter() - speculative_started) * 1000.0
                )
            else:
                # The fork's post-action state is only known once the draft
                # exec finishes. If it's still running, we cannot decide
                # reuse safely.
                draft_done = speculative_exec_future.done()
                if draft_done:
                    hidden_penalty_ms = max(0.0, (time.perf_counter() - speculative_started) * 1000.0)
                    try:
                        speculative_exec_future.result()
                    except Exception as exc:
                        logger.debug(
                            "Rejected speculative command completed with error sandbox=%s pair_id=%s error=%s",
                            self.sandbox.sandbox_id,
                            pair_id,
                            exc,
                        )
                else:
                    self._track_hidden_spec_penalty(
                        future=speculative_exec_future,
                        pair_id=pair_id,
                        speculative_started=speculative_started,
                    )
                reuse_candidate_value = draft_done
                current_state_changed, fork_state_changed = self._finalize_speculative_fork(
                    accepted=accepted,
                    fork=fork_handle.handle,
                    reuse_candidate=reuse_candidate_value,
                )
            penalty_ms = max(0.0, (time.perf_counter() - oracle_finished) * 1000.0)
            logger.debug(
                "Speculative command rejected sandbox=%s pair_id=%s speculative_exec_ms=%.2f penalty_ms=%.2f "
                "hidden_penalty_ms=%.2f",
                self.sandbox.sandbox_id,
                pair_id,
                speculative_exec_ms,
                penalty_ms,
                hidden_penalty_ms,
            )
        self._record_spec_step_metrics(
            pair_id=pair_id,
            accepted=accepted,
            draft_first=True,
            fork_created=fork_handle.created,
            fork_reused=fork_handle.reused,
            fork_restore_ms=fork_restore_ms,
            speculative_exec_ms=speculative_exec_ms,
            saved_ms=saved_ms,
            penalty_ms=penalty_ms,
            hidden_penalty_ms=hidden_penalty_ms,
            current_state_changed=current_state_changed,
            fork_state_changed=fork_state_changed,
            reuse_candidate=reuse_candidate_value,
        )
        logger.debug(
            "Speculative step completed sandbox=%s pair_id=%s accepted=%s",
            self.sandbox.sandbox_id,
            pair_id,
            accepted,
        )
        return result

    def _ensure_speculative_fork(self) -> _SpeculativeForkHandle:
        if self._speculation_controller is None:
            logger.debug("Speculative fork skipped: controller unavailable sandbox=%s", self.sandbox.sandbox_id)
            return _SpeculativeForkHandle(handle=None)
        prepared = self._speculation_controller.ensure_fork()
        if prepared is None:
            logger.debug("Speculative fork controller returned no fork sandbox=%s", self.sandbox.sandbox_id)
            return _SpeculativeForkHandle(handle=None)
        if isinstance(prepared, _SpeculativeForkHandle):
            logger.debug(
                "Speculative fork prepared from handle wrapper sandbox=%s has_handle=%s created=%s reused=%s",
                self.sandbox.sandbox_id,
                prepared.handle is not None,
                prepared.created,
                prepared.reused,
            )
            return prepared
        if isinstance(prepared, tuple) and len(prepared) >= 1:
            handle = prepared[0]
            created = bool(prepared[1]) if len(prepared) >= 2 else False
            reused = bool(prepared[2]) if len(prepared) >= 3 else False
            logger.debug(
                "Speculative fork prepared from tuple sandbox=%s has_handle=%s created=%s reused=%s",
                self.sandbox.sandbox_id,
                handle is not None,
                created,
                reused,
            )
            return _SpeculativeForkHandle(handle=handle, created=created, reused=reused)
        logger.debug(
            "Speculative fork prepared from raw handle sandbox=%s has_handle=%s",
            self.sandbox.sandbox_id,
            prepared is not None,
        )
        return _SpeculativeForkHandle(handle=prepared)

    def _promote_speculative_fork(self, fork) -> None:
        if self._speculation_controller is None:
            logger.debug("Skipping speculative fork promotion: controller unavailable sandbox=%s", self.sandbox.sandbox_id)
            return
        promoter = getattr(self._speculation_controller, "promote_fork", None)
        if callable(promoter):
            logger.debug("Promoting speculative fork sandbox=%s", self.sandbox.sandbox_id)
            promoter(fork)

    def _invalidate_speculative_fork(self) -> None:
        if self._speculation_controller is None:
            return
        invalidator = getattr(self._speculation_controller, "invalidate_fork", None)
        if callable(invalidator):
            logger.debug("Invalidating speculative fork sandbox=%s", self.sandbox.sandbox_id)
            invalidator()

    def _finalize_speculative_fork(
        self,
        *,
        accepted: bool,
        fork,
        reuse_candidate: bool,
    ) -> tuple[bool, bool]:
        if self._speculation_controller is None or fork is None:
            logger.debug(
                "Skipping speculative fork finalization sandbox=%s accepted=%s controller_available=%s fork_available=%s",
                self.sandbox.sandbox_id,
                accepted,
                self._speculation_controller is not None,
                fork is not None,
            )
            return False, False
        current_state_changed = bool(self._speculation_controller.state_changed(self.sandbox))
        fork_state_changed = bool(self._speculation_controller.state_changed(fork))
        reusable = (
            reuse_candidate
            and not current_state_changed
            and not fork_state_changed
        )
        logger.debug(
            "Finalizing speculative fork sandbox=%s accepted=%s current_state_changed=%s "
            "fork_state_changed=%s reuse_candidate=%s reusable=%s",
            self.sandbox.sandbox_id,
            accepted,
            current_state_changed,
            fork_state_changed,
            reuse_candidate,
            reusable,
        )
        releaser = getattr(self._speculation_controller, "release_fork", None)
        if callable(releaser):
            releaser(
                fork,
                reusable=reusable,
                accepted=accepted,
            )
        return current_state_changed, fork_state_changed

    def _record_spec_step_metrics(
        self,
        *,
        pair_id: str,
        accepted: bool,
        draft_first: bool,
        fork_created: bool,
        fork_reused: bool,
        fork_restore_ms: float,
        speculative_exec_ms: float,
        saved_ms: float,
        penalty_ms: float,
        hidden_penalty_ms: float,
        current_state_changed: bool,
        fork_state_changed: bool,
        reuse_candidate: bool | None = None,
    ) -> None:
        with self._lock:
            self._spec_total_turns += 1
            if accepted:
                self._spec_accept_count += 1
            elif draft_first:
                self._spec_reject_count += 1
            self._spec_saved_ms += max(0.0, saved_ms)
            self._spec_penalty_ms += max(0.0, penalty_ms)
            self._spec_hidden_penalty_ms += max(0.0, hidden_penalty_ms)
            if fork_created:
                self._spec_fork_create_count += 1
            if fork_reused:
                self._spec_fork_reuse_count += 1
            total_turns = self._spec_total_turns
            accept_count = self._spec_accept_count
            reject_count = self._spec_reject_count
            saved_total_ms = self._spec_saved_ms
            penalty_total_ms = self._spec_penalty_ms
            hidden_penalty_total_ms = self._spec_hidden_penalty_ms
        logger.debug(
            (
                "Recorded speculative step metrics sandbox=%s pair_id=%s accepted=%s draft_first=%s "
                "saved_ms=%.2f penalty_ms=%.2f hidden_penalty_ms=%.2f fork_restore_ms=%.2f "
                "speculative_exec_ms=%.2f totals(turns=%d accepted=%d rejected=%d saved_ms=%.2f "
                "penalty_ms=%.2f hidden_penalty_ms=%.2f)"
            ),
            self.sandbox.sandbox_id,
            pair_id,
            accepted,
            draft_first,
            max(0.0, saved_ms),
            max(0.0, penalty_ms),
            max(0.0, hidden_penalty_ms),
            max(0.0, fork_restore_ms),
            max(0.0, speculative_exec_ms),
            total_turns,
            accept_count,
            reject_count,
            saved_total_ms,
            penalty_total_ms,
            hidden_penalty_total_ms,
        )
        if self.telemetry is not None:
            task_run_id = self._benchmark_task_run_id()
            attributes = {
                "sandbox_id": str(self.sandbox.sandbox_id),
                "task_run_id": task_run_id,
                "pair_id": pair_id,
                "accepted": 1 if accepted else 0,
                "draft_first": 1 if draft_first else 0,
                "oracle_first": 0 if draft_first else 1,
                "fork_created": 1 if fork_created else 0,
                "fork_reused": 1 if fork_reused else 0,
                "current_state_changed": 1 if current_state_changed else 0,
                "fork_state_changed": 1 if fork_state_changed else 0,
                "fork_finalized": 1 if reuse_candidate is not None else 0,
                "reuse_candidate": 1 if reuse_candidate else 0,
            }
            self.telemetry.emit_event("spec.turn.finish", attributes)
            self.telemetry.emit_metric("benchmark.spec.saved_ms", max(0.0, saved_ms), attributes)
            self.telemetry.emit_metric("benchmark.spec.penalty_ms", max(0.0, penalty_ms), attributes)
            self.telemetry.emit_metric("benchmark.spec.hidden_penalty_ms", max(0.0, hidden_penalty_ms), attributes)
            self.telemetry.emit_metric(
                "benchmark.spec.net_gain_ms",
                max(0.0, saved_ms) - max(0.0, penalty_ms),
                attributes,
            )
            self.telemetry.emit_metric("benchmark.spec.fork_restore_ms", max(0.0, fork_restore_ms), attributes)
            self.telemetry.emit_metric("benchmark.spec.speculative_exec_ms", max(0.0, speculative_exec_ms), attributes)
            self.telemetry.emit_metric(
                "benchmark.spec.accept_rate",
                0.0 if accept_count + reject_count <= 0 else accept_count / (accept_count + reject_count),
                attributes,
            )

    def _record_hidden_spec_penalty(self, *, pair_id: str, hidden_penalty_ms: float) -> None:
        hidden_penalty_ms = max(0.0, hidden_penalty_ms)
        if hidden_penalty_ms <= 0:
            return
        with self._lock:
            self._spec_hidden_penalty_ms += hidden_penalty_ms
        logger.debug(
            "Recorded hidden speculative penalty sandbox=%s pair_id=%s hidden_penalty_ms=%.2f",
            self.sandbox.sandbox_id,
            pair_id,
            hidden_penalty_ms,
        )
        if self.telemetry is not None:
            task_run_id = self._benchmark_task_run_id()
            attributes = {
                "sandbox_id": str(self.sandbox.sandbox_id),
                "task_run_id": task_run_id,
                "pair_id": pair_id,
                "accepted": 0,
                "draft_first": 1,
                "oracle_first": 0,
            }
            self.telemetry.emit_metric("benchmark.spec.hidden_penalty_ms", hidden_penalty_ms, attributes)

    def _track_hidden_spec_penalty(
        self,
        *,
        future: Future[Any],
        pair_id: str,
        speculative_started: float,
    ) -> None:
        def _finalize_hidden_penalty(completed_future: Future[Any]) -> None:
            hidden_penalty_ms = (time.perf_counter() - speculative_started) * 1000.0
            try:
                completed_future.result()
            except Exception as exc:
                logger.debug(
                    "Rejected speculative future finished with error after background cleanup "
                    "sandbox=%s pair_id=%s error=%s",
                    self.sandbox.sandbox_id,
                    pair_id,
                    exc,
                )
            self._record_hidden_spec_penalty(pair_id=pair_id, hidden_penalty_ms=hidden_penalty_ms)

        future.add_done_callback(_finalize_hidden_penalty)

    def _llm_request_timeout_seconds(self) -> float:
        raw_timeout = self.task_config.options.get("max_agent_timeout_sec", 900.0)
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_s = 900.0
        return max(60.0, timeout_s)

    def _benchmark_task_run_id(self) -> str:
        metadata = self.sandbox.launch_metadata.get("benchmark", {})
        if isinstance(metadata, dict):
            raw_task_run_id = metadata.get("task_run_id")
            if isinstance(raw_task_run_id, str) and raw_task_run_id:
                return raw_task_run_id
        return str(self.sandbox.sandbox_id)

    def _parse_command(self, assistant_content: str) -> str:
        commands = [item.strip() for item in _COMMAND_PATTERN.findall(assistant_content)]
        if len(commands) != 1 or not commands[0]:
            raise RuntimeError(f"MiniSweAgent expected exactly one mswea_bash_command tag, found {len(commands)}")
        return commands[0]

    @staticmethod
    def _commands_match(left: str, right: str) -> bool:
        return " ".join(left.split()) == " ".join(right.split())

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

    def _build_command_record(self, command: str, *, action_index: int | None = None) -> _RecordedCommand:
        raw_timeout = self.task_config.options.get("max_agent_timeout_sec", 900.0)
        try:
            timeout_s = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            timeout_s = 900.0
        return _RecordedCommand(
            action_index=self._next_action_index() if action_index is None else int(action_index),
            command=command,
            argv=("bash", "-c", command),
            cwd=self._process_cwd(),
            env=self._process_env(),
            user=self._process_user(),
            timeout_s=timeout_s,
        )

    def _exec_command(self, command: str):
        assert self.runtime is not None
        command_record = self._build_command_record(command)
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

    def _run_command_record_for_sandbox(self, command_record: _RecordedCommand, sandbox_id) -> Any:
        assert self.runtime is not None
        return self.runtime.exec(
            sandbox_id,
            list(command_record.argv),
            cwd=command_record.cwd,
            env=dict(command_record.env),
            user=command_record.user,
            timeout_s=command_record.timeout_s,
            capture_output=True,
        )

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

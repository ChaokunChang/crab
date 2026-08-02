from __future__ import annotations

import http.client
import json
import tempfile
import time
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
from unittest.mock import patch

from crab import SandboxExecResult, SandboxId
from integrations.agents import SandboxHandle, TaskConfig, TaskDescription
from integrations.agents.mini_swe import MiniSweAgent


class _DummyHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        _ = (exc_type, exc, tb)
        return None

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")


@dataclass
class _FakeRuntime:
    exec_results: list[SandboxExecResult]
    resilient_results: list[SandboxExecResult] | None = None
    exec_results_by_sandbox: dict[str, list[SandboxExecResult]] | None = None
    on_exec_call: Callable[[int], None] | None = None

    def __post_init__(self) -> None:
        self.exec_calls: list[dict[str, object]] = []
        self.resilient_calls: list[dict[str, object]] = []

    def exec(
        self,
        sandbox_id,
        argv,
        *,
        cwd=None,
        env=None,
        user=None,
        timeout_s=None,
        capture_output=True,
    ):
        self.exec_calls.append(
            {
                "sandbox_id": sandbox_id,
                "argv": list(argv),
                "cwd": cwd,
                "env": dict(env or {}),
                "user": user,
                "timeout_s": timeout_s,
                "capture_output": capture_output,
            }
        )
        if self.on_exec_call is not None:
            self.on_exec_call(len(self.exec_calls))
        queue = None
        if self.exec_results_by_sandbox is not None:
            queue = self.exec_results_by_sandbox.get(str(sandbox_id))
        if queue is None:
            queue = self.exec_results
        if not queue:
            raise AssertionError("no fake exec results remaining")
        return queue.pop(0)

    def resilient_exec(
        self,
        sandbox_id,
        argv,
        *,
        cwd=None,
        env=None,
        user=None,
        timeout_s=None,
        capture_output=True,
    ):
        self.resilient_calls.append(
            {
                "sandbox_id": sandbox_id,
                "argv": list(argv),
                "cwd": cwd,
                "env": dict(env or {}),
                "user": user,
                "timeout_s": timeout_s,
                "capture_output": capture_output,
            }
        )
        queue = self.resilient_results if self.resilient_results is not None else self.exec_results
        if not queue:
            raise AssertionError("no fake resilient exec results remaining")
        return queue.pop(0)


class MiniSweAgentTests(unittest.TestCase):
    def _bundle_dir(self, tmp: str, *, env: list[str] | None = None) -> Path:
        bundle_dir = Path(tmp) / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        (bundle_dir / "config.json").write_text(
            json.dumps(
                {
                    "linux": {"namespaces": [], "cgroupsPath": ""},
                    "mounts": [],
                    "process": {
                        "terminal": False,
                        "cwd": "/testbed",
                        "args": ["tail", "-f", "/dev/null"],
                        "env": list(env or ["PATH=/usr/bin", "HOME=/root"]),
                        "user": {"uid": 0, "gid": 0},
                    },
                    "root": {"path": "rootfs", "readonly": False},
                }
            ),
            encoding="utf-8",
        )
        return bundle_dir

    def _spec_controller(self, *, fork_handle: SandboxHandle):
        class _Controller:
            def __init__(self, handle: SandboxHandle) -> None:
                self.handle = handle
                self.promoted: list[SandboxHandle] = []
                self.release_calls: list[tuple[SandboxHandle, bool, bool]] = []
                self.state_changed_map: dict[str, bool] = {}
                self.invalidated = 0

            def ensure_fork(self):
                return self.handle, True, False

            def promote_fork(self, fork: SandboxHandle) -> None:
                self.promoted.append(fork)

            def state_changed(self, sandbox: SandboxHandle) -> bool:
                return self.state_changed_map.get(str(sandbox.sandbox_id), False)

            def invalidate_fork(self) -> None:
                self.invalidated += 1

            def release_fork(self, fork: SandboxHandle, *, reusable: bool, accepted: bool) -> None:
                self.release_calls.append((fork, reusable, accepted))

        return _Controller(fork_handle)

    def test_perform_task_runs_commands_updates_progress_and_collects_submission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = _FakeRuntime(
                exec_results=[
                    SandboxExecResult(
                        args=("runc", "exec", "sbx-mini"),
                        returncode=0,
                        stdout="hello\n",
                        stderr="warn\n",
                    ),
                    SandboxExecResult(
                        args=("runc", "exec", "sbx-mini"),
                        returncode=0,
                        stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch\n",
                        stderr="",
                    ),
                ]
            )
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820", "max_agent_timeout_sec": 321}),
                runtime=runtime,
                llm_base_url=sandbox.llm_base_url,
            )
            responses = [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "THOUGHT: inspect\n<mswea_bash_command>echo hello</mswea_bash_command>",
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    "THOUGHT: submit\n"
                                    "<mswea_bash_command>"
                                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && printf 'patch\\n'"
                                    "</mswea_bash_command>"
                                ),
                            }
                        }
                    ]
                },
            ]

            with self.assertLogs("integrations.agents.mini_swe", level="DEBUG") as captured:
                with patch("integrations.agents.mini_swe.urllib.request.urlopen", side_effect=[_DummyHTTPResponse(item) for item in responses]):
                    agent.perform_task()

        self.assertEqual(len(runtime.exec_calls), 2)
        self.assertEqual(runtime.exec_calls[0]["argv"], ["bash", "-c", "echo hello"])
        self.assertEqual(runtime.exec_calls[0]["timeout_s"], 321.0)
        self.assertEqual(runtime.exec_calls[1]["timeout_s"], 321.0)
        payload = agent.poll_status()
        self.assertEqual(payload["state"], "finished")
        self.assertEqual(payload["total_actions"], 2)
        self.assertEqual(payload["filesystem_actions"], 0)
        self.assertEqual(payload["process_actions"], 0)
        self.assertEqual(payload["network_actions"], 0)
        self.assertEqual(payload["stateful_actions"], 2)
        self.assertEqual(payload["submission"], "patch\n")
        joined_messages = "\n".join(message["content"] for message in agent._messages if message["role"] == "user")
        self.assertIn("hello", joined_messages)
        self.assertIn("warn", joined_messages)
        self.assertEqual(agent._recorded_activity_events(), [])
        joined_logs = "\n".join(captured.output)
        self.assertIn("Starting mini_swe task sandbox=sbx-mini", joined_logs)
        self.assertIn("Executing mini_swe command sandbox=sbx-mini action=1", joined_logs)
        self.assertIn("Completed mini_swe task sandbox=sbx-mini actions=2", joined_logs)

    def test_restore_replays_completed_commands_before_retrying_current_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-replay"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            agent: MiniSweAgent | None = None

            def _trigger_restore(call_index: int) -> None:
                if call_index != 2 or agent is None:
                    return
                agent.record_restore_trace_cursor(0)
                agent.on_restore_complete()

            runtime = _FakeRuntime(
                exec_results=[
                    SandboxExecResult(
                        args=("runc", "exec", "sbx-mini-replay"),
                        returncode=0,
                        stdout="first\n",
                        stderr="",
                    ),
                    SandboxExecResult(
                        args=("runc", "exec", "sbx-mini-replay"),
                        returncode=1,
                        stdout="",
                        stderr="container not running",
                    ),
                    SandboxExecResult(
                        args=("runc", "exec", "sbx-mini-replay"),
                        returncode=0,
                        stdout="second\n",
                        stderr="",
                    ),
                    SandboxExecResult(
                        args=("runc", "exec", "sbx-mini-replay"),
                        returncode=0,
                        stdout="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\npatch\n",
                        stderr="",
                    ),
                ],
                resilient_results=[
                    SandboxExecResult(
                        args=("runc", "exec", "sbx-mini-replay"),
                        returncode=0,
                        stdout="first\n",
                        stderr="",
                    ),
                ],
                on_exec_call=_trigger_restore,
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820", "max_agent_timeout_sec": 321}),
                runtime=runtime,
                llm_base_url=sandbox.llm_base_url,
            )
            responses = [
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "THOUGHT: first\n<mswea_bash_command>echo first</mswea_bash_command>",
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "THOUGHT: second\n<mswea_bash_command>echo second</mswea_bash_command>",
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": (
                                    "THOUGHT: submit\n"
                                    "<mswea_bash_command>"
                                    "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && printf 'patch\\n'"
                                    "</mswea_bash_command>"
                                ),
                            }
                        }
                    ]
                },
            ]

            with patch("integrations.agents.mini_swe.urllib.request.urlopen", side_effect=[_DummyHTTPResponse(item) for item in responses]):
                agent.perform_task()

        self.assertEqual([call["argv"] for call in runtime.resilient_calls], [["bash", "-c", "echo first"]])
        self.assertEqual(
            [call["argv"] for call in runtime.exec_calls],
            [
                ["bash", "-c", "echo first"],
                ["bash", "-c", "echo second"],
                ["bash", "-c", "echo second"],
                ["bash", "-c", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && printf 'patch\\n'"],
            ],
        )
        self.assertEqual(agent.poll_status()["total_actions"], 3)

    def test_on_restore_complete_preserves_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-restore"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_trace_replay",
                launch_source="compose",
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=_FakeRuntime(exec_results=[]),
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            agent._state = "running"
            agent._completed_actions = 3

            agent.on_restore_complete()

        self.assertEqual(agent._state, "running")
        self.assertEqual(agent._completed_actions, 3)
        self.assertTrue(agent._restore_event.is_set())

    def test_configure_bundle_sets_quiet_keepalive_process_and_testbed_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle_dir(
                tmp,
                env=[
                    "PATH=/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                    "HOME=/root",
                    "TZ=Etc/UTC",
                ],
            )
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-configure"),
                bundle_dir=bundle_dir,
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_trace_replay",
                launch_source="compose",
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=_FakeRuntime(exec_results=[]),
                llm_base_url="http://127.0.0.1:12345/v1",
            )

            agent.configure_bundle()

            payload = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
            process = payload["process"]
            self.assertEqual(process["args"], ["/bin/sh", "-c", "exec /bin/sleep infinity >/dev/null 2>&1"])
            self.assertEqual(process["cwd"], "/testbed")
            self.assertFalse(process["terminal"])
            env = process["env"]
            self.assertIn("HOME=/root", env)
            self.assertIn("PYTHONDONTWRITEBYTECODE=1", env)
            self.assertIn(
                "PATH=/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                env,
            )
            self.assertIn("TZ=Etc/UTC", env)
            self.assertIn("testbed", agent.rootfs_init_dirs())
            self.assertEqual(
                agent.extra_launch_metadata(),
                {"host_inspector_ignore_process_rules": [{"executable_basename": "sleep"}]},
            )

    def test_configure_bundle_sets_default_path_when_bundle_env_has_no_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bundle_dir = self._bundle_dir(tmp, env=["HOME=/root"])
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-configure-no-path"),
                bundle_dir=bundle_dir,
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_trace_replay",
                launch_source="compose",
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=_FakeRuntime(exec_results=[]),
                llm_base_url="http://127.0.0.1:12345/v1",
            )

            agent.configure_bundle()

            payload = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
            env = payload["process"]["env"]
            self.assertIn("PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", env)

    def test_spec_perform_step_accepts_matching_draft_and_runs_only_on_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            fork = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec-fork"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8124,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            controller = self._spec_controller(fork_handle=fork)
            runtime = _FakeRuntime(
                exec_results=[
                    SandboxExecResult(args=("runc", "exec", "sbx-mini-spec-fork"), returncode=0, stdout="ok\n", stderr=""),
                ]
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=runtime,
                llm_base_url=sandbox.llm_base_url,
                speculation_controller=controller,
            )
            self.addCleanup(agent.post_task_finish)
            agent._messages = agent._initial_messages()

            def _assistant_turn(*, role=None, pair_id=None):
                if role == "draft":
                    time.sleep(0.01)
                    return SimpleNamespace(
                        role="draft",
                        pair_id=pair_id,
                        content="THOUGHT: draft\n<mswea_bash_command>echo same</mswea_bash_command>",
                        command="echo same",
                    )
                time.sleep(0.02)
                return SimpleNamespace(
                    role="oracle",
                    pair_id=pair_id,
                    content="THOUGHT: oracle\n<mswea_bash_command>echo same</mswea_bash_command>",
                    command="echo same",
                )

            agent._request_assistant_turn = _assistant_turn  # type: ignore[method-assign]

            result = agent._spec_perform_step()

        self.assertEqual(result.stdout, "ok\n")
        self.assertEqual(len(runtime.exec_calls), 1)
        self.assertEqual(runtime.exec_calls[0]["sandbox_id"], SandboxId("sbx-mini-spec-fork"))
        self.assertEqual(agent.poll_status()["spec_accept_count"], 1)
        self.assertEqual(controller.promoted, [fork])
        # Accept + both state_changed=False (mock default) → reusable.
        self.assertEqual(controller.release_calls[0][1:], (True, True))

    def test_spec_perform_step_rejects_mismatch_and_runs_oracle_on_active_sandbox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            fork = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec-fork"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8124,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            controller = self._spec_controller(fork_handle=fork)
            runtime = _FakeRuntime(
                exec_results=[
                    SandboxExecResult(args=("runc", "exec", "sbx-mini-spec-fork"), returncode=0, stdout="draft\n", stderr=""),
                    SandboxExecResult(args=("runc", "exec", "sbx-mini-spec"), returncode=0, stdout="oracle\n", stderr=""),
                ]
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=runtime,
                llm_base_url=sandbox.llm_base_url,
                speculation_controller=controller,
            )
            self.addCleanup(agent.post_task_finish)
            agent._messages = agent._initial_messages()

            def _assistant_turn(*, role=None, pair_id=None):
                if role == "draft":
                    time.sleep(0.01)
                    return SimpleNamespace(
                        role="draft",
                        pair_id=pair_id,
                        content="THOUGHT: draft\n<mswea_bash_command>echo draft</mswea_bash_command>",
                        command="echo draft",
                    )
                time.sleep(0.02)
                return SimpleNamespace(
                    role="oracle",
                    pair_id=pair_id,
                    content="THOUGHT: oracle\n<mswea_bash_command>echo oracle</mswea_bash_command>",
                    command="echo oracle",
                )

            agent._request_assistant_turn = _assistant_turn  # type: ignore[method-assign]

            result = agent._spec_perform_step()

        self.assertEqual(result.stdout, "oracle\n")
        self.assertEqual([call["sandbox_id"] for call in runtime.exec_calls], [SandboxId("sbx-mini-spec-fork"), SandboxId("sbx-mini-spec")])
        self.assertEqual(agent.poll_status()["spec_reject_count"], 1)
        self.assertEqual(controller.promoted, [])
        # Reject + draft done before oracle (oracle sleeps 0.02s vs draft
        # 0.01s) + both state_changed=False (mock default) → reusable.
        self.assertEqual(controller.release_calls[0][1:], (True, False))

    def test_spec_perform_step_falls_back_when_speculative_exec_hits_retryable_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            fork = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec-fork"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8124,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            controller = self._spec_controller(fork_handle=fork)
            runtime = _FakeRuntime(
                exec_results=[
                    SandboxExecResult(
                        args=("runc", "exec", "sbx-mini-spec-fork"),
                        returncode=255,
                        stdout="",
                        stderr='time="2026-04-09T13:54:19+08:00" level=error msg="container does not exist"',
                    ),
                    SandboxExecResult(args=("runc", "exec", "sbx-mini-spec"), returncode=0, stdout="oracle\n", stderr=""),
                ]
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=runtime,
                llm_base_url=sandbox.llm_base_url,
                speculation_controller=controller,
            )
            self.addCleanup(agent.post_task_finish)
            agent._messages = agent._initial_messages()

            def _assistant_turn(*, role=None, pair_id=None):
                _ = pair_id
                time.sleep(0.01 if role == "draft" else 0.02)
                return SimpleNamespace(
                    role=str(role),
                    pair_id=pair_id,
                    content="THOUGHT: same\n<mswea_bash_command>echo same</mswea_bash_command>",
                    command="echo same",
                )

            agent._request_assistant_turn = _assistant_turn  # type: ignore[method-assign]

            result = agent._spec_perform_step()

        self.assertEqual(result.stdout, "oracle\n")
        self.assertEqual([call["sandbox_id"] for call in runtime.exec_calls], [SandboxId("sbx-mini-spec-fork"), SandboxId("sbx-mini-spec")])
        self.assertEqual(agent.poll_status()["spec_accept_count"], 0)
        self.assertEqual(agent.poll_status()["spec_reject_count"], 1)
        self.assertEqual(controller.promoted, [])
        self.assertEqual(controller.release_calls[0][1:], (False, False))

    def test_spec_perform_step_reject_hides_background_penalty_from_agent_loop(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            fork = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec-fork"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8124,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            controller = self._spec_controller(fork_handle=fork)

            def _sleep_on_fork_exec(call_index: int) -> None:
                if call_index == 1:
                    time.sleep(0.15)

            runtime = _FakeRuntime(
                exec_results=[],
                exec_results_by_sandbox={
                    "sbx-mini-spec-fork": [
                        SandboxExecResult(
                            args=("runc", "exec", "sbx-mini-spec-fork"),
                            returncode=0,
                            stdout="draft\n",
                            stderr="",
                        )
                    ],
                    "sbx-mini-spec": [
                        SandboxExecResult(
                            args=("runc", "exec", "sbx-mini-spec"),
                            returncode=0,
                            stdout="oracle\n",
                            stderr="",
                        )
                    ],
                },
                on_exec_call=_sleep_on_fork_exec,
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=runtime,
                llm_base_url=sandbox.llm_base_url,
                speculation_controller=controller,
            )
            self.addCleanup(agent.post_task_finish)
            agent._messages = agent._initial_messages()

            def _assistant_turn(*, role=None, pair_id=None):
                time.sleep(0.01 if role == "draft" else 0.02)
                if role == "draft":
                    return SimpleNamespace(
                        role="draft",
                        pair_id=pair_id,
                        content="THOUGHT: draft\n<mswea_bash_command>echo draft</mswea_bash_command>",
                        command="echo draft",
                    )
                return SimpleNamespace(
                    role="oracle",
                    pair_id=pair_id,
                    content="THOUGHT: oracle\n<mswea_bash_command>echo oracle</mswea_bash_command>",
                    command="echo oracle",
                )

            agent._request_assistant_turn = _assistant_turn  # type: ignore[method-assign]

            started = time.perf_counter()
            result = agent._spec_perform_step()
            elapsed_s = time.perf_counter() - started
            time.sleep(0.2)

        self.assertEqual(result.stdout, "oracle\n")
        self.assertLess(elapsed_s, 0.12)
        payload = agent.poll_status()
        self.assertEqual(payload["spec_reject_count"], 1)
        self.assertLess(float(payload["spec_penalty_ms"]), 50.0)
        self.assertGreater(float(payload["spec_hidden_penalty_ms"]), 100.0)
        self.assertEqual(controller.promoted, [])
        self.assertEqual(controller.release_calls[0][1:], (False, False))

    def test_spec_perform_step_falls_back_when_oracle_arrives_first(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            fork = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec-fork"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8124,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            controller = self._spec_controller(fork_handle=fork)
            runtime = _FakeRuntime(
                exec_results=[
                    SandboxExecResult(args=("runc", "exec", "sbx-mini-spec"), returncode=0, stdout="oracle\n", stderr=""),
                ]
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=runtime,
                llm_base_url=sandbox.llm_base_url,
                speculation_controller=controller,
            )
            self.addCleanup(agent.post_task_finish)
            agent._messages = agent._initial_messages()

            def _assistant_turn(*, role=None, pair_id=None):
                if role == "draft":
                    time.sleep(0.03)
                    return SimpleNamespace(
                        role="draft",
                        pair_id=pair_id,
                        content="THOUGHT: draft\n<mswea_bash_command>echo draft</mswea_bash_command>",
                        command="echo draft",
                    )
                time.sleep(0.01)
                return SimpleNamespace(
                    role="oracle",
                    pair_id=pair_id,
                    content="THOUGHT: oracle\n<mswea_bash_command>echo oracle</mswea_bash_command>",
                    command="echo oracle",
                )

            agent._request_assistant_turn = _assistant_turn  # type: ignore[method-assign]

            result = agent._spec_perform_step()

        self.assertEqual(result.stdout, "oracle\n")
        self.assertEqual(len(runtime.exec_calls), 1)
        self.assertEqual(runtime.exec_calls[0]["sandbox_id"], SandboxId("sbx-mini-spec"))
        self.assertEqual(agent.poll_status()["spec_accept_count"], 0)
        self.assertEqual(agent.poll_status()["spec_reject_count"], 0)
        self.assertEqual(controller.invalidated, 1)

    def test_spec_perform_step_falls_back_to_oracle_when_draft_request_disconnects(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-mini-spec"),
                bundle_dir=self._bundle_dir(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="mini_swe_spec_trace_replay",
                launch_source="compose",
                llm_base_url="http://127.0.0.1:12345/v1",
            )
            runtime = _FakeRuntime(
                exec_results=[
                    SandboxExecResult(args=("runc", "exec", "sbx-mini-spec"), returncode=0, stdout="oracle\n", stderr=""),
                ]
            )
            agent = MiniSweAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"swebench_instance_id": "django__django-13820"}),
                runtime=runtime,
                llm_base_url=sandbox.llm_base_url,
            )
            self.addCleanup(agent.post_task_finish)
            agent._messages = agent._initial_messages()

            def _assistant_turn(*, role=None, pair_id=None):
                if role == "draft":
                    time.sleep(0.01)
                    raise http.client.RemoteDisconnected("Remote end closed connection without response")
                time.sleep(0.02)
                return SimpleNamespace(
                    role="oracle",
                    pair_id=pair_id,
                    content="THOUGHT: oracle\n<mswea_bash_command>echo oracle</mswea_bash_command>",
                    command="echo oracle",
                )

            agent._request_assistant_turn = _assistant_turn  # type: ignore[method-assign]

            result = agent._spec_perform_step()

        self.assertEqual(result.stdout, "oracle\n")
        self.assertEqual(len(runtime.exec_calls), 1)
        self.assertEqual(runtime.exec_calls[0]["sandbox_id"], SandboxId("sbx-mini-spec"))
        self.assertEqual(agent.poll_status()["spec_accept_count"], 0)
        self.assertEqual(agent.poll_status()["spec_reject_count"], 0)


if __name__ == "__main__":
    unittest.main()

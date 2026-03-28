from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from agent_cr import SandboxExecResult, SandboxId
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
        if not self.exec_results:
            raise AssertionError("no fake exec results remaining")
        return self.exec_results.pop(0)

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


if __name__ == "__main__":
    unittest.main()

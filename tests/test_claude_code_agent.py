from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_cr import InMemoryTelemetrySink
from agent_cr import SandboxId
from integrations.agents import SandboxHandle, TaskConfig, TaskDescription
from integrations.agents.claude_code import (
    CLAUDE_CODE_WRAPPER_ARG,
    ClaudeCodeAgent,
    _DEFAULT_DEBUG_LOG_MOUNT_PATH,
    _DEFAULT_OUTPUT_SINK_PATH,
)
from integrations.sandboxes.claude_code.harness import (
    CLAUDE_HOME_ROOT_MOUNT_PATH,
    LOGS_MOUNT_PATH,
)


class ClaudeCodeAgentTests(unittest.TestCase):
    def _make_agent(self, tmp: str) -> ClaudeCodeAgent:
        sandbox = SandboxHandle(
            sandbox_id=SandboxId("sbx-claude"),
            bundle_dir=Path(tmp),
            status_port=8123,
            last_status={},
            llm_service_type="claude_code_trace_replay",
            launch_source="runc",
            llm_control_base_url="http://127.0.0.1:8124",
        )
        return ClaudeCodeAgent(
            sandbox,
            TaskDescription("solve the task"),
            TaskConfig(),
        )

    def test_empty_done_marker_is_treated_as_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._make_agent(tmp)
            logs_dir = Path(tmp)
            done_path = logs_dir / "claude_code.task.done"
            exit_path = logs_dir / "claude_code.task.exit"
            done_path.touch()
            exit_path.touch()

            self.assertFalse(
                agent._task_markers_indicate_completion(exit_path=exit_path, done_path=done_path)
            )

    def test_nonempty_done_marker_is_treated_as_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._make_agent(tmp)
            logs_dir = Path(tmp)
            done_path = logs_dir / "claude_code.task.done"
            exit_path = logs_dir / "claude_code.task.exit"
            done_path.write_text("done\n", encoding="utf-8")

            self.assertTrue(
                agent._task_markers_indicate_completion(exit_path=exit_path, done_path=done_path)
            )

    def test_zero_exit_marker_without_done_is_treated_as_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            agent = self._make_agent(tmp)
            logs_dir = Path(tmp)
            done_path = logs_dir / "claude_code.task.done"
            exit_path = logs_dir / "claude_code.task.exit"
            exit_path.write_text("0\n", encoding="utf-8")

            self.assertTrue(
                agent._task_markers_indicate_completion(exit_path=exit_path, done_path=done_path)
            )

    def test_configure_bundle_does_not_precreate_task_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-claude"),
                bundle_dir=root,
                status_port=8123,
                last_status={},
                llm_service_type="claude_code_trace_replay",
                launch_source="compose",
                launch_metadata={
                        "claude_code": {
                            "runtime_root": str(root / "runtime"),
                            "claude_home_root": str(root / "home-root"),
                            "claude_home": str(root / "home"),
                            "logs_dir": str(root / "logs"),
                            "claude_bin": "/opt/claude-code-runtime/claude",
                        "model_name": "claude-opus-4-6",
                        "resolved_version": "2.1.34",
                        "supports_bare_flag": True,
                    }
                },
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "process": {
                            "args": ["/bin/sh", "-lc", "echo hi"],
                            "env": [],
                            "cwd": "/app",
                        },
                        "mounts": [],
                    }
                ),
                encoding="utf-8",
            )

            agent = ClaudeCodeAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(),
                llm_base_url="http://127.0.0.1:8080/v1",
            )

            agent.configure_bundle()

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            command = payload["process"]["args"][2]
            self.assertNotIn("touch /opt/claude-code-logs/claude_code.task.done", command)
            self.assertNotIn("touch /opt/claude-code-logs/claude_code.task.exit", command)
            self.assertIn("install -d -m 755 /usr/local/bin", command)
            self.assertIn("python3-venv python3-pip", command)
            self.assertIn(CLAUDE_CODE_WRAPPER_ARG, command)

    def test_configure_bundle_mounts_claude_home_root_and_routes_debug_logs_to_mount_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime_root = root / "runtime"
            claude_home_root = root / "home-root"
            claude_home = claude_home_root / ".claude"
            logs_dir = root / "logs"
            runtime_root.mkdir(parents=True, exist_ok=True)
            claude_home.mkdir(parents=True, exist_ok=True)
            logs_dir.mkdir(parents=True, exist_ok=True)
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-claude"),
                bundle_dir=root,
                status_port=8123,
                last_status={},
                llm_service_type="claude_code_trace_replay",
                launch_source="compose",
                launch_metadata={
                    "claude_code": {
                        "runtime_root": str(runtime_root),
                        "claude_home_root": str(claude_home_root),
                        "claude_home": str(claude_home),
                        "logs_dir": str(logs_dir),
                        "claude_bin": "/opt/claude-code-runtime/claude",
                        "model_name": "claude-opus-4-6",
                        "resolved_version": "2.1.34",
                        "supports_bare_flag": True,
                    }
                },
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "process": {
                            "args": ["/bin/sh", "-lc", "echo hi"],
                            "env": [],
                            "cwd": "/app",
                        },
                        "mounts": [],
                    }
                ),
                encoding="utf-8",
            )

            agent = ClaudeCodeAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(),
                llm_base_url="http://127.0.0.1:8080/v1",
            )

            agent.configure_bundle()

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            mounts = {mount["destination"]: mount for mount in payload["mounts"]}
            self.assertEqual(mounts[LOGS_MOUNT_PATH]["source"], str(logs_dir))
            self.assertEqual(mounts[CLAUDE_HOME_ROOT_MOUNT_PATH]["source"], str(claude_home_root))
            command = payload["process"]["args"][2]
            self.assertIn(_DEFAULT_DEBUG_LOG_MOUNT_PATH, command)
            self.assertIn(_DEFAULT_OUTPUT_SINK_PATH, command)
            self.assertIn(CLAUDE_CODE_WRAPPER_ARG, command)
            self.assertIn(f"export HOME={CLAUDE_HOME_ROOT_MOUNT_PATH}", command)
            self.assertNotIn(str(logs_dir / "claude_code.debug.log"), command)
            self.assertNotIn(f"{LOGS_MOUNT_PATH}/claude_code.output.log", command)
            self.assertEqual(agent.extra_launch_metadata()["rootfs_copy_paths"], [])

    def test_prepare_sandbox_forwards_telemetry_to_harness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = InMemoryTelemetrySink()
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-claude"),
                bundle_dir=Path(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="claude_code_trace_replay",
                launch_source="runc",
            )
            agent = ClaudeCodeAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(options={"trace_agent_version": "2.1.34"}),
                agent_host_dir=Path(tmp) / "agent-root",
                llm_base_url="http://127.0.0.1:8080/v1",
                telemetry=telemetry,
            )
            runtime = SimpleNamespace(
                root=Path(tmp) / "runtime",
                mounted_claude_bin="/opt/claude-code-runtime/claude",
                resolved_version="2.1.34",
                source_binary=Path(tmp) / "versions" / "2.1.34",
                runtime_strategy="version_cache",
                supports_bare_flag=True,
                ignore_process_rules=[],
            )
            state = SimpleNamespace(
                home_root=Path(tmp) / "home-root",
                claude_home=Path(tmp) / "home-root" / ".claude",
                logs_dir=Path(tmp) / "logs",
            )

            with patch("integrations.agents.claude_code.prepare_claude_code_runtime", return_value=runtime) as prepare_runtime, patch(
                "integrations.agents.claude_code.prepare_claude_code_state",
                return_value=state,
            ) as prepare_state:
                agent.prepare_sandbox()

        self.assertIs(prepare_runtime.call_args.kwargs["telemetry"], telemetry)
        self.assertEqual(prepare_runtime.call_args.kwargs["sandbox_id"], "sbx-claude")
        self.assertEqual(prepare_runtime.call_args.kwargs["requested_version"], "2.1.34")
        self.assertIs(prepare_state.call_args.kwargs["telemetry"], telemetry)
        self.assertEqual(prepare_state.call_args.kwargs["sandbox_id"], "sbx-claude")

    def test_wait_for_progress_retries_transient_replay_state_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sandbox = SandboxHandle(
                sandbox_id=SandboxId("sbx-claude"),
                bundle_dir=Path(tmp),
                status_port=8123,
                last_status={},
                llm_service_type="claude_code_trace_replay",
                launch_source="compose",
                llm_control_base_url="http://127.0.0.1:8124",
            )
            agent = ClaudeCodeAgent(
                sandbox,
                TaskDescription("solve the task"),
                TaskConfig(),
            )
            status_payload = {
                "agent_type": agent.agent_type,
                "state": "running",
                "total_actions": 2,
                "filesystem_actions": 2,
                "process_actions": 2,
                "network_actions": 2,
                "stateful_actions": 2,
            }

            with patch.object(agent, "_replay_action_count", side_effect=[RuntimeError("timed out"), 2]), patch.object(
                agent,
                "poll_status",
                return_value=status_payload,
            ), patch("integrations.agents.claude_code.time.sleep", return_value=None):
                payload = agent.wait_for_progress(minimum_actions=2)

        self.assertEqual(payload["total_actions"], 2)

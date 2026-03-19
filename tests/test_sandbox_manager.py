from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_cr import RuncRuntime, RuncRuntimePaths, SandboxId
from agent_cr.runtime import CommandRunner

RuncSandboxManager = RuncRuntime
RuncSandboxManagerPaths = RuncRuntimePaths


class FakeCommandRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(self, command: list[str], *, cwd: Path | None = None):
        _ = cwd
        self.commands.append(tuple(command))
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class ResumeRaceCommandRunner(CommandRunner):
    def __init__(self, *, state_status_after_resume: str) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._state_status_after_resume = state_status_after_resume

    def run(self, command: list[str], *, cwd: Path | None = None):
        _ = cwd
        self.commands.append(tuple(command))
        if command[-2:] == ["resume", "sbx-test"]:
            return type(
                "Result",
                (),
                {
                    "command": tuple(command),
                    "returncode": 1,
                    "stdout": "",
                    "stderr": 'time="2026-03-19T22:23:26+08:00" level=error msg="container not paused"',
                },
            )()
        if command[-2:] == ["state", "sbx-test"]:
            payload = {"status": self._state_status_after_resume, "pid": 123 if self._state_status_after_resume == "running" else 0}
            return type(
                "Result",
                (),
                {"command": tuple(command), "returncode": 0, "stdout": json.dumps(payload), "stderr": ""},
            )()
        return type(
            "Result",
            (),
            {"command": tuple(command), "returncode": 0, "stdout": "", "stderr": ""},
        )()


class FakeHostInspectorClient:
    def __init__(self) -> None:
        self.register_calls: list[tuple[SandboxId, str, str, object | None]] = []
        self.unregister_calls: list[SandboxId] = []

    def register_sandbox(
        self,
        sandbox_id: SandboxId,
        runtime: str,
        object_id: str,
        *,
        ignore_process_rules=None,
    ) -> dict[str, object]:
        self.register_calls.append((sandbox_id, runtime, object_id, ignore_process_rules))
        return {"ok": True}

    def unregister_sandbox(self, sandbox_id: SandboxId) -> dict[str, object]:
        self.unregister_calls.append(sandbox_id)
        return {"ok": True}


class SandboxManagerTests(unittest.TestCase):
    def test_runc_sandbox_manager_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            host_inspector = FakeHostInspectorClient()
            manager = RuncSandboxManager(
                command_runner=runner,
                host_inspector_client=host_inspector,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-test",
                    "bundle_path": str(root / "bundles" / "sbx-test"),
                },
            )
            self.assertEqual(sandbox_id, SandboxId("sbx-test"))
            self.assertEqual(manager.describe(sandbox_id).status, "running")

            manager.pause(sandbox_id)
            self.assertEqual(manager.describe(sandbox_id).status, "paused")

            manager.resume(sandbox_id)
            self.assertEqual(manager.describe(sandbox_id).status, "running")

            manager.stop(sandbox_id)
            self.assertEqual(manager.describe(sandbox_id).status, "stopped")

            manager.delete(sandbox_id)
            self.assertEqual(
                host_inspector.register_calls,
                [(SandboxId("sbx-test"), "runc", "sbx-test", None)],
            )
            self.assertEqual(host_inspector.unregister_calls, [SandboxId("sbx-test")])
            self.assertEqual(
                runner.commands,
                [
                    ("zfs", "create", "-o", f"mountpoint={root / 'bundles' / 'sbx-test' / 'rootfs'}", "pool/agent-cr/sbx-test"),
                    ("runc", "--root", str(root / "state"), "create", "--bundle", str(root / "bundles" / "sbx-test"), "sbx-test"),
                    ("runc", "--root", str(root / "state"), "start", "sbx-test"),
                    ("runc", "--root", str(root / "state"), "pause", "sbx-test"),
                    ("runc", "--root", str(root / "state"), "resume", "sbx-test"),
                    ("runc", "--root", str(root / "state"), "kill", "sbx-test", "TERM"),
                    ("runc", "--root", str(root / "state"), "delete", "-f", "sbx-test"),
                    ("zfs", "destroy", "-r", "pool/agent-cr/sbx-test"),
                ],
            )

    def test_runc_sandbox_manager_registers_restored_sandbox_with_host_inspector(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = FakeCommandRunner()
            host_inspector = FakeHostInspectorClient()
            manager = RuncSandboxManager(
                command_runner=runner,
                host_inspector_client=host_inspector,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-restore",
                    "bundle_path": str(root / "bundles" / "sbx-restore"),
                    "host_inspector_ignore_process_rules": [{"executable_basename": "node"}],
                },
            )
            host_inspector.register_calls.clear()

            manager.prepare_for_restore(sandbox_id)
            manager.mark_restored(sandbox_id)

            self.assertEqual(manager.describe(sandbox_id).status, "running")
            self.assertEqual(
                host_inspector.register_calls,
                [
                    (
                        SandboxId("sbx-restore"),
                        "runc",
                        "sbx-restore",
                        [{"executable_basename": "node"}],
                    )
                ],
            )

    def test_runc_resume_treats_not_paused_as_benign_when_container_is_already_running(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = ResumeRaceCommandRunner(state_status_after_resume="running")
            manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-test",
                    "bundle_path": str(root / "bundles" / "sbx-test"),
                },
            )
            manager.pause(sandbox_id)
            manager.resume(sandbox_id)

            self.assertEqual(manager.describe(sandbox_id).status, "running")
            self.assertIn(
                ("runc", "--root", str(root / "state"), "state", "sbx-test"),
                runner.commands,
            )

    def test_runc_resume_treats_not_paused_as_benign_when_container_is_already_stopped(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_sandbox_mgr_") as tmp:
            root = Path(tmp)
            runner = ResumeRaceCommandRunner(state_status_after_resume="stopped")
            manager = RuncSandboxManager(
                command_runner=runner,
                paths=RuncSandboxManagerPaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            sandbox_id = manager.launch(
                "runc",
                {
                    "sandbox_id": "sbx-test",
                    "bundle_path": str(root / "bundles" / "sbx-test"),
                },
            )
            manager.pause(sandbox_id)
            manager.resume(sandbox_id)

            self.assertEqual(manager.describe(sandbox_id).status, "stopped")


if __name__ == "__main__":
    unittest.main()

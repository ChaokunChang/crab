from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_cr import RuncSandboxManager, RuncSandboxManagerPaths, SandboxId
from agent_cr.runtime import CommandRunner


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


class FakeHostInspectorClient:
    def __init__(self) -> None:
        self.register_calls: list[tuple[SandboxId, str, str]] = []
        self.unregister_calls: list[SandboxId] = []

    def register_sandbox(self, sandbox_id: SandboxId, runtime: str, object_id: str) -> dict[str, object]:
        self.register_calls.append((sandbox_id, runtime, object_id))
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
                [(SandboxId("sbx-test"), "runc", "sbx-test")],
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


if __name__ == "__main__":
    unittest.main()

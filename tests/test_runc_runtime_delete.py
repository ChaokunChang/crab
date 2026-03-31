from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_cr import RuncRuntime, RuncRuntimePaths, SandboxId
from agent_cr.runtime.base import CommandResult, CommandRunner


class _DeleteRetryRunner(CommandRunner):
    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._delete_attempts = 0

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        _ = cwd
        _ = timeout_seconds
        self.commands.append(tuple(command))
        if command[-3:] == ["delete", "-f", "sbx-test"]:
            self._delete_attempts += 1
            if self._delete_attempts == 1:
                return CommandResult(
                    command=tuple(command),
                    returncode=1,
                    stdout="",
                    stderr='time="2026-03-30T02:18:41+08:00" level=error msg="container init still running"',
                )
        return CommandResult(command=tuple(command), returncode=0, stdout="", stderr="")


class RuncRuntimeDeleteTests(unittest.TestCase):
    def test_delete_runtime_kills_and_retries_when_init_is_still_running(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_runc_delete_") as tmp:
            root = Path(tmp)
            runner = _DeleteRetryRunner()
            runtime = RuncRuntime(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            runtime.launch(
                "runc",
                {
                    "sandbox_id": "sbx-test",
                    "bundle_path": str(root / "bundles" / "sbx-test"),
                },
            )

            runtime.delete_runtime(SandboxId("sbx-test"), force=True, ignore_missing=True)

            self.assertEqual(
                runner.commands[-3:],
                [
                    ("runc", "--root", str(root / "state"), "delete", "-f", "sbx-test"),
                    ("runc", "--root", str(root / "state"), "kill", "sbx-test", "KILL"),
                    ("runc", "--root", str(root / "state"), "delete", "-f", "sbx-test"),
                ],
            )
            self.assertEqual(runtime.describe(SandboxId("sbx-test")).status, "stopped")


if __name__ == "__main__":
    unittest.main()

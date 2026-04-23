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


class _RecordingRunner(CommandRunner):
    def __init__(
        self,
        *,
        stderr_by_command: dict[tuple[str, ...], str] | None = None,
        results_by_command: dict[tuple[str, ...], list[CommandResult]] | None = None,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._stderr_by_command = stderr_by_command or {}
        self._results_by_command = {key: list(value) for key, value in (results_by_command or {}).items()}

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        _ = cwd
        _ = timeout_seconds
        key = tuple(command)
        self.commands.append(key)
        queued = self._results_by_command.get(key)
        if queued:
            return queued.pop(0)
        stderr = self._stderr_by_command.get(key, "")
        returncode = 1 if stderr else 0
        return CommandResult(command=key, returncode=returncode, stdout="", stderr=stderr)


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

    def test_promote_filesystem_dataset_promotes_clone_dataset(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_runc_delete_") as tmp:
            root = Path(tmp)
            runner = _RecordingRunner()
            runtime = RuncRuntime(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            runtime.promote_filesystem_dataset(SandboxId("sbx-fork"))

            self.assertEqual(runner.commands[-1], ("zfs", "promote", "pool/agent-cr/sbx-fork"))

    def test_promote_filesystem_dataset_ignores_non_clone_datasets(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_runc_delete_") as tmp:
            root = Path(tmp)
            command = ("zfs", "promote", "pool/agent-cr/sbx-fork")
            runner = _RecordingRunner(
                stderr_by_command={command: "cannot promote 'pool/agent-cr/sbx-fork': not a cloned filesystem"}
            )
            runtime = RuncRuntime(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            runtime.promote_filesystem_dataset(SandboxId("sbx-fork"))

            self.assertEqual(runner.commands[-1], command)

    def test_promote_filesystem_dataset_retries_after_conflicting_snapshot(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_runc_delete_") as tmp:
            root = Path(tmp)
            promote_command = ("zfs", "promote", "pool/agent-cr/sbx-fork")
            destroy_snapshot_command = ("zfs", "destroy", "pool/agent-cr/sbx-fork@ckpt-1")
            runner = _RecordingRunner(
                results_by_command={
                    promote_command: [
                        CommandResult(
                            command=promote_command,
                            returncode=1,
                            stdout="",
                            stderr=(
                                "cannot promote 'pool/agent-cr/sbx-fork': conflicting snapshot 'ckpt-1' "
                                "from parent 'pool/agent-cr/sbx-parent@ckpt-1'"
                            ),
                        ),
                        CommandResult(
                            command=promote_command,
                            returncode=0,
                            stdout="",
                            stderr="",
                        ),
                    ],
                    destroy_snapshot_command: [
                        CommandResult(
                            command=destroy_snapshot_command,
                            returncode=0,
                            stdout="",
                            stderr="",
                        )
                    ],
                }
            )
            runtime = RuncRuntime(
                command_runner=runner,
                paths=RuncRuntimePaths(
                    state_root=root / "state",
                    bundle_root=root / "bundles",
                    metadata_root=root / "metadata",
                    zfs_dataset_prefix="pool/agent-cr",
                ),
            )

            runtime.promote_filesystem_dataset(SandboxId("sbx-fork"))

            self.assertEqual(
                runner.commands,
                [
                    promote_command,
                    destroy_snapshot_command,
                    promote_command,
                ],
            )


if __name__ == "__main__":
    unittest.main()

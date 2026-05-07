from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_cr import RuncRuntime, RuncRuntimePaths, SandboxId
from agent_cr.ids import CheckpointId
from agent_cr.runtime.base import CommandResult, CommandRunner


class _CapturingRunner(CommandRunner):
    """Records every command and returns a configurable result. Anything
    not seeded returns success; the test asserts on `commands`.
    """

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
    ) -> CommandResult:
        _ = (cwd, timeout_seconds)
        key = tuple(command)
        self.commands.append(key)
        return CommandResult(command=key, returncode=0, stdout="", stderr="")


class RuncCheckpointCommandTests(unittest.TestCase):
    def _make_runtime(self, runner: CommandRunner, root: Path) -> RuncRuntime:
        runtime = RuncRuntime(
            command_runner=runner,
            paths=RuncRuntimePaths(
                state_root=root / "state",
                bundle_root=root / "bundles",
                checkpoint_root=root / "checkpoints",
                metadata_root=root / "metadata",
                zfs_dataset_prefix="pool/agent-cr",
            ),
        )
        runtime.launch(
            "runc",
            {"sandbox_id": "sbx-incr", "bundle_path": str(root / "bundles" / "sbx-incr")},
        )
        return runtime

    def _last_checkpoint_command(self, runner: _CapturingRunner) -> tuple[str, ...]:
        for cmd in reversed(runner.commands):
            if "checkpoint" in cmd:
                return cmd
        raise AssertionError("no checkpoint command captured")

    def test_full_checkpoint_omits_pre_dump_and_parent_path(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            sbx = SandboxId("sbx-incr")
            runtime.checkpoint_process(sbx, CheckpointId("ck-1"), leave_running=True)
        cmd = self._last_checkpoint_command(runner)
        self.assertIn("--leave-running=true", cmd)
        self.assertNotIn("--pre-dump", cmd)
        self.assertNotIn("--parent-path", cmd)

    def test_pre_dump_chain_root_omits_parent_path(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            sbx = SandboxId("sbx-incr")
            runtime.pre_dump_process(sbx, CheckpointId("ck-1"))
        cmd = self._last_checkpoint_command(runner)
        self.assertIn("--pre-dump", cmd)
        # Pre-dump implies process keeps running; we must not also pass
        # --leave-running, which runc rejects in combination.
        self.assertFalse(any(x.startswith("--leave-running") for x in cmd))
        self.assertNotIn("--parent-path", cmd)

    def test_pre_dump_with_parent_resolves_relative_path(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            sbx = SandboxId("sbx-incr")
            # Materialize the parent's pre_dump dir so the runtime doesn't
            # raise FileNotFoundError before constructing the command.
            parent_pd = Path(runtime.pre_dump_location(sbx, CheckpointId("ck-1")))
            parent_pd.mkdir(parents=True, exist_ok=True)
            runtime.pre_dump_process(
                sbx,
                CheckpointId("ck-2"),
                parent_checkpoint_id=CheckpointId("ck-1"),
            )
        cmd = self._last_checkpoint_command(runner)
        self.assertIn("--pre-dump", cmd)
        idx = cmd.index("--parent-path")
        # Two levels up from <ckpt-2>/pre_dump to reach <ckpt-1>/pre_dump.
        self.assertEqual(cmd[idx + 1], "../../ck-1/pre_dump")

    def test_incremental_dump_parent_points_at_sibling_pre_dump(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            runtime = self._make_runtime(runner, root)
            sbx = SandboxId("sbx-incr")
            # Final dump's parent is THIS checkpoint's own pre_dump sibling.
            sibling_pd = Path(runtime.pre_dump_location(sbx, CheckpointId("ck-2")))
            sibling_pd.mkdir(parents=True, exist_ok=True)
            runtime.checkpoint_process(
                sbx,
                CheckpointId("ck-2"),
                leave_running=False,
                parent_checkpoint_id=CheckpointId("ck-2"),
            )
        cmd = self._last_checkpoint_command(runner)
        idx = cmd.index("--parent-path")
        self.assertEqual(cmd[idx + 1], "../pre_dump")
        self.assertIn("--leave-running=false", cmd)
        self.assertNotIn("--pre-dump", cmd)

    def test_missing_parent_pre_dump_dir_raises(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            sbx = SandboxId("sbx-incr")
            with self.assertRaises(FileNotFoundError):
                runtime.pre_dump_process(
                    sbx,
                    CheckpointId("ck-2"),
                    parent_checkpoint_id=CheckpointId("ck-missing"),
                )

    def test_runtime_capability_advertises_incremental_process(self) -> None:
        runner = _CapturingRunner()
        with tempfile.TemporaryDirectory() as raw:
            runtime = self._make_runtime(runner, Path(raw))
            self.assertTrue(runtime.capabilities().supports_incremental_process)


if __name__ == "__main__":
    unittest.main()

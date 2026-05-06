from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from agent_cr import RuncRuntime, RuncRuntimePaths, SandboxId
from agent_cr.runtime.base import CommandResult, CommandRunner


class _PauseFailRunner(CommandRunner):
    """Records every command and lets the test seed targeted failure
    responses for specific (command,) keys. Anything not seeded
    returns success.
    """

    def __init__(
        self,
        *,
        results_by_key: dict[tuple[str, ...], list[CommandResult]] | None = None,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self._results = {k: list(v) for k, v in (results_by_key or {}).items()}

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
        queued = self._results.get(key)
        if queued:
            return queued.pop(0)
        return CommandResult(command=key, returncode=0, stdout="", stderr="")


class RuncRuntimePauseTests(unittest.TestCase):
    def _make_runtime(self, runner: CommandRunner, root: Path) -> RuncRuntime:
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
            {"sandbox_id": "sbx-test", "bundle_path": str(root / "bundles" / "sbx-test")},
        )
        return runtime

    def test_pause_freeze_timeout_triggers_defensive_resume(self) -> None:
        """When `runc pause` fails with the cgroup-freeze timeout, the
        kernel may still complete the freeze asynchronously. Pre-fix,
        the container could end up stuck in FROZEN forever — observed
        in benchmark 20260429_031243 where 30 minutes after the failed
        pause, terminus' `runc exec` failed with "cannot exec in a
        paused container". The fix issues a defensive `runc resume`
        before re-raising; resume is idempotent on a thawed cgroup
        (returns "container not paused")."""
        with tempfile.TemporaryDirectory(prefix="agent_cr_runc_pause_") as tmp:
            root = Path(tmp)
            pause_cmd = ("runc", "--root", str(root / "state"), "pause", "sbx-test")
            runner = _PauseFailRunner(
                results_by_key={
                    pause_cmd: [
                        CommandResult(
                            command=pause_cmd,
                            returncode=1,
                            stdout="",
                            stderr='level=error msg="timeout of 10s reached waiting for the cgroup to freeze"',
                        )
                    ]
                },
            )
            runtime = self._make_runtime(runner, root)

            with self.assertRaises(RuntimeError):
                runtime.pause(SandboxId("sbx-test"))

            # The failed pause must be followed by a defensive resume
            # so the kernel cgroup is explicitly thawed even if runc's
            # async freeze completed after its 10s give-up window.
            resume_cmd = ("runc", "--root", str(root / "state"), "resume", "sbx-test")
            self.assertIn(resume_cmd, runner.commands)
            self.assertEqual(runner.commands[-2:], [pause_cmd, resume_cmd])

    def test_pause_freeze_timeout_retries_defensive_resume_when_first_fails(self) -> None:
        """Heavy-load corollary of the freeze-timeout path: under load
        the defensive resume's *own* cgroup-freeze poll can time out
        too (build-cython-ext fault-35 in benchmark 20260430_123407).
        The cgroup.freeze=0 write itself succeeded — only runc's
        kernel-ack poll was slow — so a few seconds later the kernel
        has finished thawing and the next resume's poll completes
        immediately AND updates state.json from "paused" to "running".
        Without this retry, state.json stays "paused" forever and
        every subsequent `runc exec` is rejected.

        First defensive resume call returns the same freeze-timeout;
        second call (after retry delay) succeeds. Both must appear in
        the recorded command stream."""
        with tempfile.TemporaryDirectory(prefix="agent_cr_runc_pause_") as tmp:
            root = Path(tmp)
            pause_cmd = ("runc", "--root", str(root / "state"), "pause", "sbx-test")
            resume_cmd = ("runc", "--root", str(root / "state"), "resume", "sbx-test")
            runner = _PauseFailRunner(
                results_by_key={
                    pause_cmd: [
                        CommandResult(
                            command=pause_cmd,
                            returncode=1,
                            stdout="",
                            stderr='level=error msg="timeout of 10s reached waiting for the cgroup to freeze"',
                        )
                    ],
                    resume_cmd: [
                        CommandResult(
                            command=resume_cmd,
                            returncode=1,
                            stdout="",
                            stderr='level=error msg="timeout of 10s reached waiting for the cgroup to freeze"',
                        ),
                        # Second attempt succeeds — the kernel finished
                        # thawing during the retry delay.
                        CommandResult(command=resume_cmd, returncode=0, stdout="", stderr=""),
                    ],
                },
            )
            runtime = self._make_runtime(runner, root)

            # Patch out the sleep so the retry delay doesn't slow the
            # test suite. The real path waits 5s; here we just want
            # the retry sequence verified.
            from unittest.mock import patch

            with patch("agent_cr.runtime.runc.time.sleep"):
                with self.assertRaises(RuntimeError):
                    runtime.pause(SandboxId("sbx-test"))

            # Pause was attempted once; resume was attempted twice.
            self.assertEqual(runner.commands.count(pause_cmd), 1)
            self.assertEqual(runner.commands.count(resume_cmd), 2)

    def test_pause_container_not_running_does_not_attempt_resume(self) -> None:
        """When the pause failed because the container was already
        gone, there is nothing to thaw. The defensive resume is for
        the freeze-timeout path; on container-gone we just sync
        runtime state and re-raise."""
        with tempfile.TemporaryDirectory(prefix="agent_cr_runc_pause_") as tmp:
            root = Path(tmp)
            pause_cmd = ("runc", "--root", str(root / "state"), "pause", "sbx-test")
            runner = _PauseFailRunner(
                results_by_key={
                    pause_cmd: [
                        CommandResult(
                            command=pause_cmd,
                            returncode=1,
                            stdout="",
                            stderr='level=error msg="container not running"',
                        )
                    ]
                },
            )
            runtime = self._make_runtime(runner, root)

            with self.assertRaises(RuntimeError):
                runtime.pause(SandboxId("sbx-test"))

            resume_cmd = ("runc", "--root", str(root / "state"), "resume", "sbx-test")
            self.assertNotIn(resume_cmd, runner.commands)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import subprocess
import unittest
from unittest.mock import patch

from agent_cr.runtime.base import SubprocessCommandRunner


class RuntimeBaseTests(unittest.TestCase):
    def test_detached_commands_use_devnull_for_all_stdio(self) -> None:
        runner = SubprocessCommandRunner(timeout_seconds=1.0)

        with patch("agent_cr.runtime.base.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=["runc", "run", "-d", "sbx-test"],
                returncode=0,
                stdout=None,
                stderr=None,
            )

            result = runner.run(["runc", "run", "-d", "sbx-test"])

        self.assertEqual(result.returncode, 0)
        kwargs = run_mock.call_args.kwargs
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIs(kwargs["stdout"], subprocess.DEVNULL)
        self.assertIs(kwargs["stderr"], subprocess.DEVNULL)

    def test_attached_commands_capture_stdout_and_stderr(self) -> None:
        runner = SubprocessCommandRunner(timeout_seconds=1.0)

        with patch("agent_cr.runtime.base.subprocess.run") as run_mock:
            run_mock.return_value = subprocess.CompletedProcess(
                args=["runc", "state", "sbx-test"],
                returncode=0,
                stdout="ok\n",
                stderr="",
            )

            result = runner.run(["runc", "state", "sbx-test"])

        self.assertEqual(result.stdout, "ok\n")
        kwargs = run_mock.call_args.kwargs
        self.assertIsNone(kwargs["stdin"])
        self.assertIs(kwargs["stdout"], subprocess.PIPE)
        self.assertIs(kwargs["stderr"], subprocess.PIPE)


if __name__ == "__main__":
    unittest.main()

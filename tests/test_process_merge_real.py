"""Real-host end-to-end for C4.1 process-merge replay: a source with a
live background process adopts a fork's work by replaying its journaled
execs (auto strategy picks replay), the background process survives,
files land, and the nondeterministic command surfaces as exactly one
deviation. Plus the daemon/CLI passthrough. Self-skipping outside the
crab-dev VM."""
from __future__ import annotations

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox
from crab.cli import commands as cli_commands
from crab.daemon.server import DaemonServer


def _real_stack_available() -> bool:
    import os

    if os.geteuid() != 0:
        return False
    return all(shutil.which(tool) is not None for tool in ("docker", "runc", "criu", "zfs"))


class ProcessMergeRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_procmerge_e2e_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _engine(self) -> Engine:
        engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=self.root / "storage",
                runtime_root=self.root / "runtime",
            )
        )
        self.addCleanup(engine.stop)
        return engine

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def test_auto_replay_preserves_source_processes_and_adopts_work(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        # The source runs a background workload - the very thing a
        # promotion would kill, so auto must pick replay.
        self._run(
            sandbox,
            "mkdir -p /probe && sh -c 'nohup sleep 300 > /dev/null 2>&1 & echo $! > /probe/bg.pid'",
        )

        [fork] = sandbox.fork()
        self.addCleanup(fork.kill)
        self._run(fork, "echo alpha > /probe/a.txt")
        self._run(fork, "echo beta > /probe/b.txt")
        # Nondeterministic on purpose: $$ differs per shell, so the
        # replayed stdout cannot match the recorded digest.
        self._run(fork, "echo $$ > /probe/nondet.txt && cat /probe/nondet.txt")

        report = sandbox.merge_processes(fork)

        self.assertEqual(report.strategy, "replay")
        self.assertGreater(report.source_processes, 2)
        self.assertEqual(len(report.replayed), 3)
        self.assertEqual(report.deviations, 1)
        self.assertFalse(report.stopped_early)
        deviated = [entry for entry in report.replayed if entry.deviated]
        self.assertEqual(len(deviated), 1)
        self.assertIn("nondet", " ".join(deviated[0].argv))
        # The source's background process survived the whole merge.
        self.assertEqual(
            self._run(sandbox, "test -d /proc/$(cat /probe/bg.pid) && echo alive"), "alive"
        )
        # The fork's work re-happened on the source.
        self.assertEqual(self._run(sandbox, "cat /probe/a.txt"), "alpha")
        self.assertEqual(self._run(sandbox, "cat /probe/b.txt"), "beta")
        self.assertEqual(
            self._run(sandbox, "test -s /probe/nondet.txt && echo present"), "present"
        )
        # The fork stays alive (replay is read-only toward it).
        self.assertEqual(self._run(fork, "cat /probe/a.txt"), "alpha")
        markers = [
            record
            for record in sandbox.actions(kind="lifecycle")
            if record["payload"].get("event") == "process_replay"
        ]
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]["payload"]["metadata"]["deviations"], 1)

    def test_stop_on_deviation_halts_replay(self) -> None:
        engine = self._engine()
        sandbox = Sandbox(image=self._IMAGE, engine=engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, "mkdir -p /probe")

        [fork] = sandbox.fork()
        self.addCleanup(fork.kill)
        self._run(fork, "echo $$ > /probe/first-nondet.txt && cat /probe/first-nondet.txt")
        self._run(fork, "echo never > /probe/after.txt")

        report = sandbox.merge_processes(
            fork, strategy="replay", stop_on_deviation=True
        )

        self.assertTrue(report.stopped_early)
        self.assertEqual(len(report.replayed), 1)
        self.assertEqual(report.deviations, 1)
        self.assertEqual(
            self._run(sandbox, "test -e /probe/after.txt && echo yes || echo no"), "no"
        )


class ProcessMergeDaemonRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_procmerged_e2e_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.socket_path = root / "crab.sock"
        self.daemon = DaemonServer(
            engine_config=EngineConfig(
                runtime="runc",
                enable_sandbox_network=False,
                enable_interceptor=False,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            ),
            socket_path=self.socket_path,
        )
        self.daemon.start()
        self.addCleanup(self.daemon.stop)
        self.engine = Engine.connect(self.socket_path)

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def test_remote_replay_and_cli(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self._run(sandbox, "mkdir -p /probe")
        [fork] = sandbox.fork()
        self.addCleanup(fork.kill)
        self._run(fork, "echo remote > /probe/remote.txt")

        # SDK over RPC (explicit replay: the idle source would otherwise
        # auto-resolve to the C4.2 promote path).
        report = sandbox.merge_processes(fork, strategy="replay")
        self.assertEqual(report.strategy, "replay")
        self.assertEqual(report.deviations, 0)
        self.assertEqual(self._run(sandbox, "cat /probe/remote.txt"), "remote")

        # CLI round trip: the journal now holds both execs and both are
        # deterministic, so the replay is clean and exits 0.
        self._run(fork, "echo second > /probe/second.txt")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = cli_commands.main(
                [
                    "--socket", str(self.socket_path),
                    "sandbox", "merge-processes",
                    str(sandbox.sandbox_id), str(fork.sandbox_id),
                    "--strategy", "replay",
                ]
            )
        self.assertEqual(rc, 0, msg=stdout.getvalue())
        self.assertIn("strategy=replay replayed=2 deviations=0", stdout.getvalue())
        self.assertEqual(self._run(sandbox, "cat /probe/second.txt"), "second")


if __name__ == "__main__":
    unittest.main()

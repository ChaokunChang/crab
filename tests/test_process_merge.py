"""Unit tests for C4.1 process merge: the replay engine, strategy
resolution from the process census, CrabSystem orchestration and
guards, SDK plumbing, and the daemon route/shim/CLI. Host-runnable —
no runc."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from crab import (
    AdapterFileSystemCWorker,
    AdapterFileSystemRWorker,
    AdapterProcessCWorker,
    AdapterProcessRWorker,
    CRExecutor,
    CRScheduler,
    CrabSystem,
    DefaultCWorker,
    DefaultRWorker,
    EBPFSandboxInspector,
    ExecutorConfig,
    InMemorySchedulerStateStore,
    InMemoryTelemetrySink,
    LocalCheckpointManager,
    SandboxId,
    SchedulerConfig,
    StorageConfig,
)
from crab.daemon.server import _build_handler, _Routes, _TxnConflict
from crab.daemon.transport import DaemonClient, DaemonRequestError, serve_unix_socket
from crab.interceptor import SandboxResponseGateRegistry
from crab.journal import ActionJournal
from crab.models import ProcessMergeReport, ReplayEntry, SandboxExecResult
from crab.process_merge import (
    PROCESS_PROBE_ARGV,
    ProcessMergeConflict,
    replay_fork_execs,
)
from crab.remote_engine import RemoteEngine
from crab.sandbox import Sandbox
from crab.scheduler import FaultToleranceCheckpointingPolicy


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _record_exec(journal, sandbox_id, argv, *, returncode=0, stdout="out", capture=True):
    return journal.record_exec(
        SandboxId(str(sandbox_id)),
        argv=list(argv),
        cwd="/w",
        env={"K": "V"},
        user="root",
        timeout_s=30.0,
        capture_output=capture,
        returncode=returncode,
        duration_ms=1.0,
        stdout=stdout if capture else None,
        stderr="",
        started_at="2026-01-01T00:00:00+00:00",
        finished_at="2026-01-01T00:00:01+00:00",
    )


class _FakeRecord:
    def __init__(self, seq, payload):
        self.seq = seq
        self.payload = payload


class ReplayEngineTests(unittest.TestCase):
    def _record(self, seq, argv, *, returncode=0, stdout_sha=None, **extra):
        payload = {
            "argv": list(argv),
            "cwd": "/w",
            "env": {"K": "V"},
            "user": "root",
            "timeout_s": 30.0,
            "returncode": returncode,
            "stdout_sha256": stdout_sha,
        }
        payload.update(extra)
        return _FakeRecord(seq, payload)

    def test_verbatim_passthrough_and_clean_match(self) -> None:
        calls = []

        def exec_fn(argv, **kwargs):
            calls.append((tuple(argv), kwargs))
            return SandboxExecResult(args=tuple(argv), returncode=0, stdout="out")

        records = [self._record(3, ["echo", "hi"], stdout_sha=_sha("out"))]
        entries, stopped = replay_fork_execs(exec_fn, records)

        self.assertFalse(stopped)
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.origin_seq, 3)
        self.assertFalse(entry.deviated)
        self.assertTrue(entry.stdout_matched)
        argv, kwargs = calls[0]
        self.assertEqual(argv, ("echo", "hi"))
        self.assertEqual(
            kwargs,
            {
                "cwd": "/w",
                "env": {"K": "V"},
                "user": "root",
                "timeout_s": 30.0,
                "capture_output": True,
            },
        )

    def test_returncode_and_stdout_deviations(self) -> None:
        def exec_fn(argv, **kwargs):
            return SandboxExecResult(args=tuple(argv), returncode=1, stdout="other")

        records = [
            self._record(0, ["a"], returncode=0, stdout_sha=_sha("out")),  # rc + stdout
            self._record(1, ["b"], returncode=1, stdout_sha=_sha("other")),  # clean
            self._record(2, ["c"], returncode=1, stdout_sha=_sha("out")),  # stdout only
        ]
        entries, _ = replay_fork_execs(exec_fn, records)
        self.assertEqual([entry.deviated for entry in entries], [True, False, True])

    def test_uncaptured_records_have_no_stdout_verdict(self) -> None:
        def exec_fn(argv, **kwargs):
            self.assertTrue(kwargs["capture_output"])  # always captured on replay
            return SandboxExecResult(args=tuple(argv), returncode=0, stdout="whatever")

        records = [self._record(0, ["x"], returncode=0, stdout_sha=None)]
        entries, _ = replay_fork_execs(exec_fn, records)
        self.assertIsNone(entries[0].stdout_matched)
        self.assertFalse(entries[0].deviated)

    def test_stop_on_deviation_short_circuits(self) -> None:
        executed = []

        def exec_fn(argv, **kwargs):
            executed.append(tuple(argv))
            return SandboxExecResult(args=tuple(argv), returncode=7, stdout="")

        records = [self._record(0, ["a"]), self._record(1, ["b"])]
        entries, stopped = replay_fork_execs(exec_fn, records, stop_on_deviation=True)
        self.assertTrue(stopped)
        self.assertEqual(len(entries), 1)
        self.assertEqual(executed, [("a",)])


class _ScriptedExec:
    """runtime.exec stand-in: answers the census probe and scripted
    replay commands."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...], dict]] = []
        self.probe_stdout = "5\n"
        self.probe_returncode = 0
        self.results: dict[tuple[str, ...], tuple[int, str]] = {}

    def __call__(self, sandbox_id, argv, **kwargs):
        self.calls.append((str(sandbox_id), tuple(argv), kwargs))
        if tuple(argv) == tuple(PROCESS_PROBE_ARGV):
            return SandboxExecResult(
                args=tuple(argv), returncode=self.probe_returncode, stdout=self.probe_stdout
            )
        returncode, stdout = self.results.get(tuple(argv), (0, "out"))
        return SandboxExecResult(args=tuple(argv), returncode=returncode, stdout=stdout)


class SystemMergeProcessesTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_procmerge_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.exec_fake = _ScriptedExec()
        self.fake_runtime = SimpleNamespace(name="runc", exec=self.exec_fake)
        self.telemetry = InMemoryTelemetrySink()
        self.journal = ActionJournal(self.root / "storage" / "journal")
        storage = LocalCheckpointManager(
            StorageConfig(root_dir=self.root / "storage"),
            destroy_filesystem_ref=lambda fs_ref: None,
        )
        executor = CRExecutor(
            ExecutorConfig(max_workers=1),
            DefaultCWorker(
                AdapterProcessCWorker(self.fake_runtime),
                AdapterFileSystemCWorker(self.fake_runtime),
                storage,
                self.fake_runtime,
            ),
            DefaultRWorker(
                AdapterProcessRWorker(self.fake_runtime),
                AdapterFileSystemRWorker(self.fake_runtime),
                storage,
            ),
            self.telemetry,
        )
        self.addCleanup(executor.shutdown)
        scheduler_cfg = SchedulerConfig(
            min_checkpoint_interval_seconds=0.0,
            force_checkpoint_after_seconds=0.0,
            require_change_signal=True,
        )
        scheduler = CRScheduler(
            scheduler_cfg,
            EBPFSandboxInspector(),
            self.fake_runtime,
            InMemorySchedulerStateStore(),
            self.telemetry,
            FaultToleranceCheckpointingPolicy(scheduler_cfg),
        )
        self.system = CrabSystem(
            scheduler=scheduler,
            executor=executor,
            storage=storage,
            inspector=EBPFSandboxInspector(),
            runtime=self.fake_runtime,
            telemetry=self.telemetry,
            response_gate_registry=SandboxResponseGateRegistry(),
            journal=self.journal,
        )
        self.source = SandboxId("sbx-src")
        self.fork = SandboxId("sbx-src-fork-1")
        self.journal.record_lifecycle(
            self.fork,
            "fork_created",
            metadata={"source_sandbox_id": str(self.source), "checkpoint_id": "base-1"},
        )

    def test_auto_resolves_replay_and_runs_fork_history(self) -> None:
        _record_exec(self.journal, self.fork, ["echo", "one"], stdout="out")
        _record_exec(self.journal, self.fork, ["echo", "two"], returncode=2, stdout="dev")
        self.exec_fake.results[("echo", "two")] = (0, "dev")  # rc deviates

        report = self.system.merge_processes(self.source, self.fork)

        self.assertEqual(report.strategy, "replay")
        self.assertEqual(report.source_processes, 5)
        self.assertEqual(len(report.replayed), 2)
        self.assertEqual(report.deviations, 1)
        self.assertFalse(report.stopped_early)
        # probe + two replays, all on the source, verbatim kwargs.
        self.assertEqual(len(self.exec_fake.calls), 3)
        self.assertTrue(all(call[0] == str(self.source) for call in self.exec_fake.calls))
        self.assertEqual(self.exec_fake.calls[0][1], tuple(PROCESS_PROBE_ARGV))
        replay_call = self.exec_fake.calls[1]
        self.assertEqual(replay_call[1], ("echo", "one"))
        self.assertEqual(
            replay_call[2],
            {
                "cwd": "/w",
                "env": {"K": "V"},
                "user": "root",
                "timeout_s": 30.0,
                "capture_output": True,
            },
        )
        markers = [
            record.payload["metadata"]
            for record in self.journal.entries(self.source, kind="lifecycle")
            if record.payload.get("event") == "process_replay"
        ]
        self.assertEqual(len(markers), 1)
        self.assertEqual(markers[0]["replayed"], 2)
        self.assertEqual(markers[0]["deviations"], 1)
        events = [
            attrs for name, attrs in self.telemetry.events if name == "process_merge.completed"
        ]
        self.assertTrue(events and events[-1]["deviations"] == 1)
        round_trip = ProcessMergeReport.from_json(report.to_json())
        self.assertEqual(round_trip, report)

    def test_auto_resolves_promote_when_source_is_quiet(self) -> None:
        self.exec_fake.probe_stdout = "2\n"
        with self.assertRaises(NotImplementedError):
            self.system.merge_processes(self.source, self.fork)

    def test_explicit_promote_points_at_c42(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.system.merge_processes(self.source, self.fork, strategy="promote")

    def test_stop_on_deviation_reflected_in_report(self) -> None:
        _record_exec(self.journal, self.fork, ["boom"], returncode=0, stdout="out")
        _record_exec(self.journal, self.fork, ["after"], stdout="out")
        self.exec_fake.results[("boom",)] = (9, "out")

        report = self.system.merge_processes(
            self.source, self.fork, strategy="replay", stop_on_deviation=True
        )

        self.assertTrue(report.stopped_early)
        self.assertEqual(len(report.replayed), 1)
        # probe + the single deviating replay; "after" never ran.
        self.assertEqual(len(self.exec_fake.calls), 2)

    def test_guards(self) -> None:
        with self.assertRaises(ValueError):
            self.system.merge_processes(self.source, self.fork, strategy="migrate")
        with self.assertRaises(ValueError):
            self.system.merge_processes(self.source, self.source)
        stranger = SandboxId("sbx-stranger")
        self.journal.record_lifecycle(
            stranger,
            "fork_created",
            metadata={"source_sandbox_id": "sbx-other", "checkpoint_id": "b"},
        )
        with self.assertRaises(ValueError):
            self.system.merge_processes(self.source, stranger)
        with self.system._txn_lock:
            self.system._active_txns[self.source] = None
        self.addCleanup(lambda: self.system._active_txns.pop(self.source, None))
        with self.assertRaises(ProcessMergeConflict):
            self.system.merge_processes(self.source, self.fork)

    def test_probe_failures_surface(self) -> None:
        self.exec_fake.probe_returncode = 1
        with self.assertRaises(RuntimeError):
            self.system.merge_processes(self.source, self.fork)
        self.exec_fake.probe_returncode = 0
        self.exec_fake.probe_stdout = "not-a-number"
        with self.assertRaises(RuntimeError):
            self.system.merge_processes(self.source, self.fork)

    def test_journal_required(self) -> None:
        self.system.journal = None
        with self.assertRaises(RuntimeError):
            self.system.merge_processes(self.source, self.fork)


class SandboxPlumbingTests(unittest.TestCase):
    class _FakeSystem:
        def __init__(self) -> None:
            self.calls: list = []

        def merge_processes(self, source_id, fork_id, **kwargs):
            self.calls.append((str(source_id), str(fork_id), kwargs))
            return ProcessMergeReport(
                source_sandbox_id=source_id,
                fork_sandbox_id=fork_id,
                strategy="replay",
                source_processes=4,
            )

    class _FakeEngine:
        def __init__(self, system) -> None:
            self.system = system

        def _register_sandbox(self, sandbox) -> None:
            pass

    def test_plumbs_all_kwargs(self) -> None:
        system = self._FakeSystem()
        engine = self._FakeEngine(system)
        source = Sandbox.connect("sbx-src", engine=engine)
        fork = Sandbox.connect("sbx-fork", engine=engine)
        report = source.merge_processes(
            fork, strategy="replay", stop_on_deviation=True, lazy_pages=False, force=True
        )
        self.assertEqual(report.strategy, "replay")
        self.assertEqual(
            system.calls,
            [
                (
                    "sbx-src",
                    "sbx-fork",
                    {
                        "strategy": "replay",
                        "policy": "fail_fast",
                        "observations": "append",
                        "stop_on_deviation": True,
                        "lazy_pages": False,
                        "force": True,
                    },
                )
            ],
        )

    def test_bare_engine_raises(self) -> None:
        class _Bare:
            system = object()

            def _register_sandbox(self, sandbox) -> None:
                pass

        sandbox = Sandbox.connect("sbx-src", engine=_Bare())
        with self.assertRaises(NotImplementedError):
            sandbox.merge_processes("sbx-fork")


def _report(strategy: str = "replay") -> ProcessMergeReport:
    return ProcessMergeReport(
        source_sandbox_id=SandboxId("sbx-src"),
        fork_sandbox_id=SandboxId("sbx-fork"),
        strategy=strategy,
        source_processes=5,
        replayed=(
            ReplayEntry(
                origin_seq=1,
                argv=("echo", "one"),
                returncode=0,
                expected_returncode=0,
                stdout_matched=True,
            ),
            ReplayEntry(
                origin_seq=2,
                argv=("echo", "two"),
                returncode=1,
                expected_returncode=0,
                stdout_matched=False,
                deviated=True,
            ),
        ),
        deviations=1,
    )


class _FakeDaemonSystem:
    def __init__(self) -> None:
        self.calls: list = []
        self.error: Exception | None = None

    def merge_processes(self, source_id, fork_id, **kwargs):
        self.calls.append((str(source_id), str(fork_id), kwargs))
        if self.error is not None:
            raise self.error
        return _report()


class _FakeDaemon:
    def __init__(self, engine) -> None:
        self.engine = engine

    def require_engine(self):
        return self.engine

    def register_sandbox(self, sandbox_id) -> None:
        pass

    def unregister_sandbox(self, sandbox_id) -> None:
        pass


class DaemonRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = SimpleNamespace(system=_FakeDaemonSystem())
        self.routes = _Routes(_FakeDaemon(self.engine))

    def test_route_serializes_report_and_plumbs(self) -> None:
        response = self.routes.merge_processes_sandbox(
            {
                "fork_sandbox_id": "sbx-fork",
                "strategy": "replay",
                "stop_on_deviation": True,
                "lazy_pages": False,
            },
            sandbox_id="sbx-src",
        )
        self.assertTrue(response["ok"])
        self.assertEqual(response["report"]["deviations"], 1)
        _, _, kwargs = self.engine.system.calls[0]
        self.assertEqual(kwargs["strategy"], "replay")
        self.assertTrue(kwargs["stop_on_deviation"])
        self.assertFalse(kwargs["lazy_pages"])

    def test_route_error_mapping(self) -> None:
        from crab.daemon.server import _BadRequest

        with self.assertRaises(_BadRequest):
            self.routes.merge_processes_sandbox({}, sandbox_id="sbx-src")
        self.engine.system.error = ProcessMergeConflict("txn active")
        with self.assertRaises(_TxnConflict) as ctx:
            self.routes.merge_processes_sandbox(
                {"fork_sandbox_id": "sbx-fork"}, sandbox_id="sbx-src"
            )
        self.assertEqual(ctx.exception.error_type, "process_merge_conflict")
        self.engine.system.error = NotImplementedError("promote lands with C4.2")
        with self.assertRaises(_BadRequest):
            self.routes.merge_processes_sandbox(
                {"fork_sandbox_id": "sbx-fork"}, sandbox_id="sbx-src"
            )

    def test_dispatch_over_socket_with_conflict(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="crab_procmerged_")
        self.addCleanup(tmp.cleanup)
        socket_path = Path(tmp.name) / "crab.sock"
        server = serve_unix_socket(socket_path, _build_handler(_FakeDaemon(self.engine)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        client = DaemonClient(socket_path, timeout_seconds=10.0)

        response = client.post_json(
            "/sandboxes/sbx-src/processes/merge", {"fork_sandbox_id": "sbx-fork"}
        )
        self.assertEqual(response["report"]["strategy"], "replay")

        self.engine.system.error = ProcessMergeConflict("busy")
        with self.assertRaises(DaemonRequestError) as ctx:
            client.post_json(
                "/sandboxes/sbx-src/processes/merge", {"fork_sandbox_id": "sbx-fork"}
            )
        self.assertEqual(ctx.exception.status_code, 409)
        payload = json.loads(ctx.exception.body.decode("utf-8"))
        self.assertEqual(payload["error_type"], "process_merge_conflict")


class _FakeDaemonClient:
    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.responses: dict[str, object] = {}

    def post_json(self, path, payload=None, *, timeout_seconds=None):
        self.requests.append({"path": path, "payload": payload, "timeout": timeout_seconds})
        response = self.responses.get(path)
        if isinstance(response, Exception):
            raise response
        return response or {"ok": True}

    def get_json(self, path, *, timeout_seconds=None):
        return self.responses.get(path) or {"ok": True}


class ShimTests(unittest.TestCase):
    _INFO = {"runtime": "runc", "default_image": "ubuntu:22.04"}

    def _engine(self):
        client = _FakeDaemonClient()
        return RemoteEngine(client, info=self._INFO), client

    def test_payload_timeout_and_rehydration(self) -> None:
        engine, client = self._engine()
        client.responses["/sandboxes/sbx-src/processes/merge"] = {
            "ok": True,
            "report": _report().to_json(),
        }
        report = engine.system.merge_processes(
            SandboxId("sbx-src"), SandboxId("sbx-fork"), stop_on_deviation=True
        )
        self.assertIsInstance(report, ProcessMergeReport)
        self.assertIsInstance(report.replayed[0], ReplayEntry)
        self.assertEqual(report.deviations, 1)
        request = client.requests[0]
        self.assertEqual(request["timeout"], 600.0)
        self.assertEqual(
            request["payload"],
            {
                "fork_sandbox_id": "sbx-fork",
                "strategy": "auto",
                "policy": "fail_fast",
                "observations": "append",
                "stop_on_deviation": True,
                "lazy_pages": True,
                "force": False,
            },
        )

    def test_conflict_rehydrates_and_others_pass_through(self) -> None:
        engine, client = self._engine()
        body = json.dumps(
            {"ok": False, "error": "busy", "error_type": "process_merge_conflict"}
        ).encode("utf-8")
        client.responses["/sandboxes/sbx-src/processes/merge"] = DaemonRequestError(
            409, "/x", body
        )
        with self.assertRaises(ProcessMergeConflict):
            engine.system.merge_processes(SandboxId("sbx-src"), SandboxId("sbx-fork"))
        client.responses["/sandboxes/sbx-src/processes/merge"] = DaemonRequestError(
            500, "/x", b"{}"
        )
        with self.assertRaises(DaemonRequestError):
            engine.system.merge_processes(SandboxId("sbx-src"), SandboxId("sbx-fork"))


class CliTests(unittest.TestCase):
    def _run_cli(self, argv: list[str], responses: dict) -> tuple[int, str, list]:
        requests: list[dict] = []

        class _CliClient:
            def __init__(self, socket_path, *, timeout_seconds):
                requests.append({"socket": str(socket_path), "timeout": timeout_seconds})

            def post_json(self, path, payload=None, *, timeout_seconds=None):
                requests.append({"path": path, "payload": payload})
                return responses[path]

            def get_json(self, path, *, timeout_seconds=None):
                return responses[path]

        stdout = io.StringIO()
        from crab.cli import commands

        with mock.patch.object(commands, "DaemonClient", _CliClient):
            with contextlib.redirect_stdout(stdout):
                rc = commands.main(argv)
        return rc, stdout.getvalue(), requests

    def test_replay_summary_deviation_rows_and_exit_code(self) -> None:
        rc, out, requests = self._run_cli(
            [
                "sandbox", "merge-processes", "sbx-src", "sbx-fork",
                "--strategy", "replay", "--stop-on-deviation",
            ],
            {
                "/sandboxes/sbx-src/processes/merge": {
                    "ok": True,
                    "report": _report().to_json(),
                }
            },
        )
        self.assertEqual(rc, 1)  # deviations surface to scripts
        self.assertIn("strategy=replay replayed=2 deviations=1", out)
        self.assertIn("deviation\tseq=2\trc=1 (expected 0)\techo two", out)
        payload = requests[-1]["payload"]
        self.assertEqual(payload["strategy"], "replay")
        self.assertTrue(payload["stop_on_deviation"])
        self.assertTrue(payload["lazy_pages"])
        self.assertEqual(requests[0]["timeout"], 600.0)

    def test_clean_replay_exits_zero(self) -> None:
        clean = ProcessMergeReport(
            source_sandbox_id=SandboxId("sbx-src"),
            fork_sandbox_id=SandboxId("sbx-fork"),
            strategy="replay",
            source_processes=3,
        )
        rc, out, _ = self._run_cli(
            ["sandbox", "merge-processes", "sbx-src", "sbx-fork"],
            {"/sandboxes/sbx-src/processes/merge": {"ok": True, "report": clean.to_json()}},
        )
        self.assertEqual(rc, 0)
        self.assertIn("deviations=0", out)


if __name__ == "__main__":
    unittest.main()

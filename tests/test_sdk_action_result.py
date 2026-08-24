"""SDK unit tests for the rich return-value surface of
``commands.run()`` / ``commands.stream()`` (feat/sdk-improvements).

Covers the new ``ActionResult`` wrapper, the ``AsyncCheckpoint`` /
``AsyncChangeset`` background handles (with client-preallocated
``ckpt-*`` ids), the read-only inspector peek, sandbox-level
``auto_checkpoint``, and the ``ExecStream`` iterate-then-.result shape.

Host-runnable: the engine/runtime/system are fakes, so no runc/CRIU/zfs
and no daemon socket. The fakes mirror the real duck-typed contracts the
SDK depends on (``runtime.exec`` / ``runtime.stream_exec`` /
``system.checkpoint_once`` / ``system.fork_changeset`` /
``system.inspector.inspect``).
"""
from __future__ import annotations

import threading
import unittest
from types import SimpleNamespace

from crab.ids import CheckpointId, SandboxId
from crab.models import (
    ChangesetEntry,
    ChangesetResult,
    ExecDone,
    ExecEvent,
    JobStatus,
    SandboxExecResult,
    SandboxSnapshot,
    utc_now,
)
from crab.sandbox import (
    ActionResult,
    AsyncChangeset,
    AsyncCheckpoint,
    ExecStream,
    Sandbox,
)


# ---------------------------------------------------------------------------
# Fakes: a minimal engine/runtime/system that duck-types onto what the
# SDK's commands namespace and _build_action_result reach for.
# ---------------------------------------------------------------------------


class _FakeInspector:
    def __init__(self, *, filesystem_changed=False, process_changed=False) -> None:
        self.filesystem_changed = filesystem_changed
        self.process_changed = process_changed
        self.inspect_calls: list[SandboxId] = []

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        self.inspect_calls.append(sandbox_id)
        return SandboxSnapshot(
            sandbox_id=sandbox_id,
            runtime_name="docker",
            is_running=True,
            process_changed=self.process_changed,
            filesystem_changed=self.filesystem_changed,
            observed_at=utc_now(),
        )

    def upsert_snapshot(self, *_, **__) -> None:
        return None

    def mark_changed(self, *_, **__) -> None:
        return None


class _FakeSystem:
    def __init__(self, inspector: _FakeInspector | None = None) -> None:
        self.inspector = inspector or _FakeInspector()
        self.checkpoint_calls: list[dict] = []
        self.fork_calls: list[tuple[SandboxId, bool]] = []
        self.checkpoint_barrier: threading.Event | None = None

    def checkpoint_once(self, sandbox_id, *, leave_running=True, checkpoint_id=None):
        if self.checkpoint_barrier is not None:
            self.checkpoint_barrier.wait(timeout=5.0)
        self.checkpoint_calls.append(
            {
                "sandbox_id": sandbox_id,
                "leave_running": leave_running,
                "checkpoint_id": checkpoint_id,
            }
        )
        # The composite worker honours a client-supplied id; mirror that.
        resolved = checkpoint_id or "ckpt-server-minted"
        ckpt = CheckpointId(str(resolved))
        return SimpleNamespace(
            checkpoint_id=ckpt,
            manifest=SimpleNamespace(checkpoint_id=ckpt),
            status=JobStatus.SUCCEEDED,
            message="",
        )

    def fork_changeset(self, sandbox_id, *, force=False):
        self.fork_calls.append((sandbox_id, force))
        return ChangesetResult(
            sandbox_id=sandbox_id,
            base_checkpoint_id=CheckpointId("ckpt-forkpoint"),
            entries=(ChangesetEntry(path="/new.txt", change="added"),),
        )

    def changeset_since(self, sandbox_id, checkpoint_id, *, use_inspector_gate=True):
        return ChangesetResult(
            sandbox_id=sandbox_id,
            base_checkpoint_id=checkpoint_id,
            entries=(),
        )


class _FakeRuntime:
    name = "docker"

    def __init__(self) -> None:
        self.exec_calls: list[list[str]] = []

    def exec(self, sandbox_id, argv, *, cwd=None, env=None, user=None,
             timeout_s=None, capture_output=True):
        self.exec_calls.append(argv)
        return SandboxExecResult(
            args=tuple(argv),
            returncode=0,
            stdout="ok",
            stderr="",
        )

    def stream_exec(self, sandbox_id, argv, *, cwd=None, env=None, user=None,
                    timeout_s=None):
        yield ExecEvent(channel="stdout", text="line1\n")
        yield ExecEvent(channel="stdout", text="line2\n")
        yield ExecDone(returncode=0)


class _FakeEngine:
    def __init__(self, system: _FakeSystem, runtime: _FakeRuntime) -> None:
        self.system = system
        self.runtime = runtime

    def _register_sandbox(self, sandbox) -> None:
        pass


def _make_sandbox(*, auto_checkpoint=False, inspector=None):
    system = _FakeSystem(inspector=inspector)
    runtime = _FakeRuntime()
    engine = _FakeEngine(system, runtime)
    sbx = Sandbox(engine=engine, autostart=False, auto_checkpoint=auto_checkpoint)
    sbx._sandbox_id = SandboxId("sbx-test")
    return sbx, system, runtime


# ---------------------------------------------------------------------------
# ActionResult basics
# ---------------------------------------------------------------------------


class ActionResultBasicsTests(unittest.TestCase):
    def test_plain_run_returns_action_result(self) -> None:
        sbx, _system, runtime = _make_sandbox()
        result = sbx.commands.run("echo hi")
        self.assertIsInstance(result, ActionResult)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "ok")
        self.assertEqual(result.args, ("/bin/sh", "-c", "echo hi"))
        # No enrichment requested -> all optional fields None.
        self.assertIsNone(result.checkpoint)
        self.assertIsNone(result.changeset)
        self.assertIsNone(result.filesystem_changed)
        self.assertIsNone(result.process_changed)
        self.assertEqual(runtime.exec_calls, [["/bin/sh", "-c", "echo hi"]])

    def test_argv_bypasses_shell(self) -> None:
        sbx, _system, runtime = _make_sandbox()
        result = sbx.commands.run(argv=["ls", "-la"])
        self.assertEqual(result.args, ("ls", "-la"))
        self.assertEqual(runtime.exec_calls, [["ls", "-la"]])

    def test_check_raises_on_nonzero(self) -> None:
        sbx, _system, runtime = _make_sandbox()

        def _boom(*_a, **_k):
            return SandboxExecResult(args=("false",), returncode=1, stdout="", stderr="nope")

        runtime.exec = _boom  # type: ignore[assignment]
        with self.assertRaises(RuntimeError):
            sbx.commands.run("false", check=True)


# ---------------------------------------------------------------------------
# checkpoint=True -> AsyncCheckpoint with pre-allocated id
# ---------------------------------------------------------------------------


class AsyncCheckpointTests(unittest.TestCase):
    def test_checkpoint_true_preallocates_id_and_waits(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        result = sbx.commands.run("make", checkpoint=True)
        self.assertIsInstance(result.checkpoint, AsyncCheckpoint)
        # id is available immediately, before the background job finishes.
        self.assertTrue(result.checkpoint.checkpoint_id.startswith("ckpt-"))
        returned_id = result.checkpoint.wait(timeout=5.0)
        self.assertEqual(returned_id, result.checkpoint.checkpoint_id)
        self.assertTrue(result.checkpoint.done)
        # The daemon received the client-preallocated id verbatim.
        self.assertEqual(len(system.checkpoint_calls), 1)
        self.assertEqual(
            system.checkpoint_calls[0]["checkpoint_id"],
            result.checkpoint.checkpoint_id,
        )

    def test_done_is_false_until_barrier_released(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        system.checkpoint_barrier = threading.Event()
        result = sbx.commands.run("make", checkpoint=True)
        handle = result.checkpoint
        assert handle is not None
        self.assertFalse(handle.done)
        with self.assertRaises(TimeoutError):
            handle.wait(timeout=0.1)
        system.checkpoint_barrier.set()
        handle.wait(timeout=5.0)
        self.assertTrue(handle.done)

    def test_checkpoint_failure_propagates_on_wait(self) -> None:
        sbx, system, _runtime = _make_sandbox()

        def _fail(*_a, **_k):
            raise RuntimeError("checkpoint blew up")

        system.checkpoint_once = _fail  # type: ignore[assignment]
        result = sbx.commands.run("make", checkpoint=True)
        assert result.checkpoint is not None
        with self.assertRaises(RuntimeError):
            result.checkpoint.wait(timeout=5.0)


class AutoCheckpointTests(unittest.TestCase):
    def test_sandbox_auto_checkpoint_triggers_without_flag(self) -> None:
        sbx, system, _runtime = _make_sandbox(auto_checkpoint=True)
        result = sbx.commands.run("echo hi")
        self.assertIsInstance(result.checkpoint, AsyncCheckpoint)
        result.checkpoint.wait(timeout=5.0)
        self.assertEqual(len(system.checkpoint_calls), 1)

    def test_auto_and_explicit_do_not_double_checkpoint(self) -> None:
        sbx, system, _runtime = _make_sandbox(auto_checkpoint=True)
        result = sbx.commands.run("echo hi", checkpoint=True)
        assert result.checkpoint is not None
        result.checkpoint.wait(timeout=5.0)
        # A single run performs a single checkpoint even when both the
        # sandbox-level flag and the per-call flag are set.
        self.assertEqual(len(system.checkpoint_calls), 1)

    def test_no_checkpoint_when_disabled(self) -> None:
        sbx, system, _runtime = _make_sandbox(auto_checkpoint=False)
        result = sbx.commands.run("echo hi")
        self.assertIsNone(result.checkpoint)
        self.assertEqual(system.checkpoint_calls, [])


# ---------------------------------------------------------------------------
# changeset=True -> AsyncChangeset (async default) / list (sync)
# ---------------------------------------------------------------------------


class ChangesetActionTests(unittest.TestCase):
    def test_changeset_async_default(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        result = sbx.commands.run("touch x", changeset=True)
        self.assertIsInstance(result.changeset, AsyncChangeset)
        entries = result.changeset.wait(timeout=5.0)
        self.assertEqual(entries, [{"path": "/new.txt", "change": "added"}])
        self.assertTrue(result.changeset.done)
        # Async changeset forces the backend diff (force=True).
        self.assertEqual(system.fork_calls, [(SandboxId("sbx-test"), True)])

    def test_changeset_sync_returns_list(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        result = sbx.commands.run("touch x", changeset=True, changeset_sync=True)
        self.assertIsInstance(result.changeset, list)
        self.assertEqual(result.changeset, [{"path": "/new.txt", "change": "added"}])
        self.assertEqual(system.fork_calls, [(SandboxId("sbx-test"), True)])

    def test_no_changeset_when_not_requested(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        result = sbx.commands.run("noop")
        self.assertIsNone(result.changeset)
        self.assertEqual(system.fork_calls, [])


# ---------------------------------------------------------------------------
# observe=True -> inspector peek (read-only, no reset)
# ---------------------------------------------------------------------------


class ObserveTests(unittest.TestCase):
    def test_observe_populates_flags(self) -> None:
        inspector = _FakeInspector(filesystem_changed=True, process_changed=False)
        sbx, _system, _runtime = _make_sandbox(inspector=inspector)
        result = sbx.commands.run("touch x", observe=True)
        self.assertTrue(result.filesystem_changed)
        self.assertFalse(result.process_changed)
        self.assertEqual(inspector.inspect_calls, [SandboxId("sbx-test")])

    def test_observe_absent_by_default(self) -> None:
        inspector = _FakeInspector(filesystem_changed=True)
        sbx, _system, _runtime = _make_sandbox(inspector=inspector)
        result = sbx.commands.run("touch x")
        self.assertIsNone(result.filesystem_changed)
        self.assertIsNone(result.process_changed)
        self.assertEqual(inspector.inspect_calls, [])


# ---------------------------------------------------------------------------
# ExecStream: iterate, then read .result
# ---------------------------------------------------------------------------


class ExecStreamTests(unittest.TestCase):
    def test_stream_yields_events_then_result(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        stream = sbx.commands.stream("make test", checkpoint=True, observe=True)
        self.assertIsInstance(stream, ExecStream)
        texts = []
        for event in stream:
            if isinstance(event, ExecEvent):
                texts.append(event.text)
        self.assertEqual(texts, ["line1\n", "line2\n"])
        result = stream.result
        self.assertIsInstance(result, ActionResult)
        self.assertEqual(result.returncode, 0)
        self.assertIsInstance(result.checkpoint, AsyncCheckpoint)
        result.checkpoint.wait(timeout=5.0)
        self.assertEqual(len(system.checkpoint_calls), 1)

    def test_result_before_consumption_raises(self) -> None:
        sbx, _system, _runtime = _make_sandbox()
        stream = sbx.commands.stream("make test")
        with self.assertRaises(RuntimeError):
            _ = stream.result

    def test_stream_changeset_sync(self) -> None:
        sbx, system, _runtime = _make_sandbox()
        stream = sbx.commands.stream("build", changeset=True, changeset_sync=True)
        list(stream)  # drain
        self.assertEqual(stream.result.changeset, [{"path": "/new.txt", "change": "added"}])
        self.assertEqual(system.fork_calls, [(SandboxId("sbx-test"), True)])


if __name__ == "__main__":
    unittest.main()

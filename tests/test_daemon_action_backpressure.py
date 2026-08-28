"""Per-sandbox checkpoint backpressure in the daemon action handler.

A new POST /sandboxes/{id}/action must wait for that sandbox's in-flight
background checkpoint (if any) to complete before it starts exec, so a slow
checkpoint delays — but never drops — the next request.
"""
from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

from crab.daemon import server as daemon_server
from crab.daemon.server import _Routes
from crab.models import FailureCode, JobStatus


class _FakeRuntime:
    def __init__(self, log: list) -> None:
        self._log = log

    def exec(self, sid, argv, *, cwd=None, env=None, user=None,
             timeout_s=None, capture_output=True):
        self._log.append(("exec", list(argv), time.monotonic()))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def stop(self, sid):
        self._log.append(("stop", str(sid), time.monotonic()))


class _FakeSystem:
    def __init__(self, log: list, ckpt_delay: float) -> None:
        self._log = log
        self._ckpt_delay = ckpt_delay
        # Observe path is unused here but action_sandbox may touch it.
        self.inspector = SimpleNamespace(
            inspect=lambda sid: SimpleNamespace(
                filesystem_changed=False, process_changed=False
            )
        )

    def checkpoint_once(self, sid, *, leave_running=True, checkpoint_id=None):
        # Simulate a slow checkpoint (e.g. large ZFS snapshot).
        time.sleep(self._ckpt_delay)
        self._log.append(("ckpt_done", str(checkpoint_id), time.monotonic()))
        return SimpleNamespace(checkpoint_id=checkpoint_id or "ckpt-auto")


class _FakeDaemon:
    def __init__(self, engine) -> None:
        self.engine = engine

    def require_engine(self):
        return self.engine


class CheckpointBackpressureTests(unittest.TestCase):
    def setUp(self) -> None:
        # Reset module-level singletons so tests don't contaminate each other.
        daemon_server._ckpt_backpressure = daemon_server._CheckpointBackpressure()
        daemon_server._sandbox_activity = daemon_server._SandboxActivityGate()
        daemon_server._job_store = daemon_server._JobStore()
        self.log: list = []

    def _make_routes(self, ckpt_delay: float) -> _Routes:
        engine = SimpleNamespace(
            runtime=_FakeRuntime(self.log),
            system=_FakeSystem(self.log, ckpt_delay),
        )
        return _Routes(_FakeDaemon(engine))

    def _wait_for_jobs_idle(self, sandbox_id: str, timeout: float = 5.0) -> None:
        """Block until the sandbox's background checkpoint has released."""
        ev = daemon_server._ckpt_backpressure._event_for(sandbox_id)
        self.assertTrue(ev.wait(timeout=timeout), "background checkpoint never finished")

    def _action(self, routes, sandbox_id, argv, *, checkpoint, checkpoint_id=None):
        body = {"exec": {"argv": argv}}
        if checkpoint:
            body["checkpoint"] = True
            if checkpoint_id is not None:
                body["checkpoint_id"] = checkpoint_id
        return routes.action_sandbox(body, sandbox_id=sandbox_id)

    def test_second_action_blocks_on_first_checkpoint(self) -> None:
        routes = self._make_routes(ckpt_delay=1.0)
        sbx = "sbx-bp-1"

        # First action returns fast: exec runs, checkpoint goes to background.
        t0 = time.monotonic()
        resp1 = self._action(routes, sbx, ["echo", "1"], checkpoint=True,
                              checkpoint_id="ckpt-1")
        first_elapsed = time.monotonic() - t0
        self.assertEqual(resp1["checkpoint_status"], "pending")
        self.assertLess(first_elapsed, 0.5,
                        "first action should not wait for its own checkpoint")

        # Second action must wait for ckpt-1 (1s) before its exec returns.
        t1 = time.monotonic()
        self._action(routes, sbx, ["echo", "2"], checkpoint=True,
                     checkpoint_id="ckpt-2")
        second_elapsed = time.monotonic() - t1
        self.assertGreaterEqual(
            second_elapsed, 0.8,
            f"second action should be blocked ~1s by backpressure, "
            f"took {second_elapsed:.3f}s",
        )
        self._wait_for_jobs_idle(sbx)

    def test_first_checkpoint_completes_before_second_exec(self) -> None:
        routes = self._make_routes(ckpt_delay=1.0)
        sbx = "sbx-bp-2"

        self._action(routes, sbx, ["echo", "1"], checkpoint=True,
                     checkpoint_id="ckpt-1")
        self._action(routes, sbx, ["echo", "2"], checkpoint=True,
                     checkpoint_id="ckpt-2")
        self._wait_for_jobs_idle(sbx)

        # Ordering: ckpt-1 completion must precede the second exec.
        ckpt1_done = next(ts for (kind, val, ts) in self.log
                          if kind == "ckpt_done" and val == "ckpt-1")
        second_exec = next(ts for (kind, val, ts) in self.log
                           if kind == "exec" and val == ["echo", "2"])
        self.assertLess(ckpt1_done, second_exec,
                        "first checkpoint must finish before the second exec starts")

    def test_backpressure_applies_even_without_new_checkpoint(self) -> None:
        # A follow-up action that does NOT request a checkpoint still waits
        # for the prior background checkpoint (the wait happens before exec).
        routes = self._make_routes(ckpt_delay=1.0)
        sbx = "sbx-bp-3"

        self._action(routes, sbx, ["echo", "1"], checkpoint=True,
                     checkpoint_id="ckpt-1")
        t1 = time.monotonic()
        resp2 = self._action(routes, sbx, ["echo", "2"], checkpoint=False)
        second_elapsed = time.monotonic() - t1
        self.assertNotIn("checkpoint_status", resp2)
        self.assertGreaterEqual(
            second_elapsed, 0.8,
            f"plain follow-up action should still wait on prior checkpoint, "
            f"took {second_elapsed:.3f}s",
        )
        self._wait_for_jobs_idle(sbx)

    def test_idle_sandbox_action_does_not_wait(self) -> None:
        # With no in-flight checkpoint, a single action returns quickly and
        # does not block on its own (fast) checkpoint.
        routes = self._make_routes(ckpt_delay=0.0)
        sbx = "sbx-bp-4"

        t0 = time.monotonic()
        resp = self._action(routes, sbx, ["echo", "1"], checkpoint=True,
                            checkpoint_id="ckpt-1")
        elapsed = time.monotonic() - t0
        self.assertEqual(resp["checkpoint_status"], "pending")
        self.assertLess(elapsed, 0.5, "idle action should not block")
        self._wait_for_jobs_idle(sbx)

        # A second action once idle also returns fast.
        t1 = time.monotonic()
        self._action(routes, sbx, ["echo", "2"], checkpoint=True,
                     checkpoint_id="ckpt-2")
        self.assertLess(time.monotonic() - t1, 0.5)
        self._wait_for_jobs_idle(sbx)

    def test_distinct_sandboxes_do_not_block_each_other(self) -> None:
        routes = self._make_routes(ckpt_delay=1.0)

        # Start a slow checkpoint on sandbox A.
        self._action(routes, "sbx-A", ["echo", "a"], checkpoint=True,
                     checkpoint_id="ckpt-A")
        # An action on sandbox B must not be blocked by A's checkpoint.
        t1 = time.monotonic()
        self._action(routes, "sbx-B", ["echo", "b"], checkpoint=True,
                     checkpoint_id="ckpt-B")
        self.assertLess(time.monotonic() - t1, 0.5,
                        "backpressure must be per-sandbox, not global")
        self._wait_for_jobs_idle("sbx-A")
        self._wait_for_jobs_idle("sbx-B")

    def test_stop_waits_for_running_command(self) -> None:
        routes = self._make_routes(ckpt_delay=0.0)
        sbx = "sbx-active-command"
        self.assertTrue(
            daemon_server._sandbox_activity.begin_command(sbx, timeout=1.0)
        )
        thread = threading.Thread(
            target=lambda: routes.stop_sandbox({}, sandbox_id=sbx)
        )
        thread.start()
        time.sleep(0.1)
        self.assertTrue(thread.is_alive())
        self.assertFalse(any(kind == "stop" for kind, _, _ in self.log))

        daemon_server._sandbox_activity.end_command(sbx)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(any(kind == "stop" for kind, _, _ in self.log))

    def test_stop_waits_for_async_checkpoint(self) -> None:
        routes = self._make_routes(ckpt_delay=0.4)
        sbx = "sbx-active-checkpoint"
        self._action(
            routes,
            sbx,
            ["echo", "checkpoint"],
            checkpoint=True,
            checkpoint_id="ckpt-active",
        )
        thread = threading.Thread(
            target=lambda: routes.stop_sandbox({}, sandbox_id=sbx)
        )
        thread.start()
        time.sleep(0.1)
        self.assertTrue(thread.is_alive())
        self.assertFalse(any(kind == "stop" for kind, _, _ in self.log))

        self._wait_for_jobs_idle(sbx)
        thread.join(timeout=2.0)
        self.assertFalse(thread.is_alive())
        checkpoint_done = next(
            ts for kind, value, ts in self.log
            if kind == "ckpt_done" and value == "ckpt-active"
        )
        stopped = next(ts for kind, _, ts in self.log if kind == "stop")
        self.assertLess(checkpoint_done, stopped)


class CheckpointStopTests(unittest.TestCase):
    def setUp(self) -> None:
        daemon_server._sandbox_activity = daemon_server._SandboxActivityGate()
        self.log: list = []

    def _routes(self, status: JobStatus) -> _Routes:
        runtime = _FakeRuntime(self.log)

        def checkpoint_once(sid, *, leave_running=True, checkpoint_id=None):
            self.assertTrue(leave_running)
            return SimpleNamespace(
                checkpoint_id=checkpoint_id or "ckpt-generated",
                status=status,
                failure_code=(
                    FailureCode.NONE
                    if status == JobStatus.SUCCEEDED
                    else FailureCode.RUNTIME_ERROR
                ),
                message=None if status == JobStatus.SUCCEEDED else "dump failed",
            )

        engine = SimpleNamespace(
            runtime=runtime,
            system=SimpleNamespace(checkpoint_once=checkpoint_once),
        )
        return _Routes(_FakeDaemon(engine))

    def test_success_returns_checkpoint_id_then_stops(self) -> None:
        response = self._routes(JobStatus.SUCCEEDED).checkpoint_stop_sandbox(
            {}, sandbox_id="sbx-checkpoint-stop"
        )
        self.assertTrue(response["ok"])
        self.assertTrue(response["stopped"])
        self.assertEqual(response["checkpoint_id"], "ckpt-generated")
        self.assertTrue(any(kind == "stop" for kind, _, _ in self.log))

    def test_failed_checkpoint_returns_id_without_stopping(self) -> None:
        response = self._routes(JobStatus.FAILED).checkpoint_stop_sandbox(
            {"checkpoint_id": "ckpt-requested"},
            sandbox_id="sbx-checkpoint-stop-failed",
        )
        self.assertFalse(response["ok"])
        self.assertFalse(response["stopped"])
        self.assertEqual(response["checkpoint_id"], "ckpt-requested")
        self.assertEqual(response["status"], "failed")
        self.assertFalse(any(kind == "stop" for kind, _, _ in self.log))


if __name__ == "__main__":
    unittest.main()

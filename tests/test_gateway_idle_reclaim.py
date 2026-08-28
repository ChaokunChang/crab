"""Unit tests for the gateway idle auto-reclaim feature (registry idle
columns/methods, timeout parsing, and the sweeper's decision + action
routing). Host-runnable — no daemon, no root."""
from __future__ import annotations

import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from crab.gateway.registry import GatewayRegistry
from crab.gateway.server import (
    DEFAULT_IDLE_ACTION,
    IDLE_ACTIONS,
    GatewayServer,
    _BadRequest,
    _GatewayRoutes,
    _parse_idle_timeout,
)


def _iso(seconds_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds_ago)).isoformat(
        timespec="seconds"
    )


class RegistryIdleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.registry = GatewayRegistry(Path(self._tmp.name) / "gateway.sqlite3")
        self.addCleanup(self.registry.close)
        self.tenant = self.registry.create_tenant("acme", {})

    def test_migration_adds_idle_columns(self) -> None:
        cols = {
            str(row["name"])
            for row in self.registry._conn.execute("PRAGMA table_info(sandboxes)")
        }
        for col in (
            "idle_timeout",
            "idle_action",
            "last_activity",
            "last_idle_action",
            "last_idle_status",
            "last_idle_checkpoint_id",
            "last_idle_reclaim_at",
            "last_idle_error",
        ):
            self.assertIn(col, cols)

    def test_set_idle_policy_and_list(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "stop")
        rows = self.registry.list_active_with_idle()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sandbox_id"], "sb-idle")
        self.assertEqual(rows[0]["idle_timeout"], 60.0)
        self.assertEqual(rows[0]["idle_action"], "stop")
        self.assertIsNotNone(rows[0]["last_activity"])

    def test_no_policy_means_not_listed(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-plain")
        self.assertEqual(self.registry.list_active_with_idle(), [])

    def test_disable_by_setting_none(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "stop")
        self.registry.set_idle_policy("sb-idle", None)
        self.assertEqual(self.registry.list_active_with_idle(), [])

    def test_set_idle_policy_none_action_preserves_existing(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "pause")
        # Extend the timer without specifying an action -> keep "pause".
        self.registry.set_idle_policy("sb-idle", 120.0, None)
        rows = self.registry.list_active_with_idle()
        self.assertEqual(rows[0]["idle_timeout"], 120.0)
        self.assertEqual(rows[0]["idle_action"], "pause")

    def test_touch_updates_activity_only_for_idle_sandboxes(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "stop")
        before = self.registry.list_active_with_idle()[0]["last_activity"]
        self.registry.touch("sb-idle")
        after = self.registry.list_active_with_idle()[0]["last_activity"]
        self.assertGreaterEqual(after, before)
        # A sandbox with no idle policy is left untouched (no error).
        self.registry.register_sandbox(self.tenant["id"], "sb-plain")
        self.registry.touch("sb-plain")

    def test_record_idle_reclaim_is_available_for_api_reads(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-idle")
        self.registry.set_idle_policy("sb-idle", 60.0, "checkpoint_stop")
        self.registry.record_idle_reclaim(
            "sb-idle",
            action="checkpoint_stop",
            status="succeeded",
            checkpoint_id="ckpt-idle-1",
        )
        row = self.registry.get_idle_policy("sb-idle")
        assert row is not None
        self.assertEqual(row["last_idle_action"], "checkpoint_stop")
        self.assertEqual(row["last_idle_status"], "succeeded")
        self.assertEqual(row["last_idle_checkpoint_id"], "ckpt-idle-1")
        self.assertIsNotNone(row["last_idle_reclaim_at"])


class ParseIdleTimeoutTests(unittest.TestCase):
    def test_numeric(self) -> None:
        self.assertEqual(_parse_idle_timeout(60), 60.0)
        self.assertEqual(_parse_idle_timeout(90.5), 90.5)

    def test_numeric_string(self) -> None:
        self.assertEqual(_parse_idle_timeout("120"), 120.0)

    def test_none_and_nonpositive_disable(self) -> None:
        self.assertIsNone(_parse_idle_timeout(None))
        self.assertIsNone(_parse_idle_timeout(0))
        self.assertIsNone(_parse_idle_timeout(-5))

    def test_malformed_is_bad_request(self) -> None:
        with self.assertRaises(_BadRequest):
            _parse_idle_timeout("not-a-number")


class IdleRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.registry = GatewayRegistry(Path(self._tmp.name) / "gateway.sqlite3")
        self.addCleanup(self.registry.close)
        self.tenant = self.registry.create_tenant("acme", {})["id"]
        self.registry.register_sandbox(self.tenant, "source")
        self.registry.set_idle_policy("source", 120.0, "pause")

    def _gateway(self, *, forks=None):
        return types.SimpleNamespace(
            registry=self.registry,
            require_owned=lambda tenant_id, sandbox_id: self.registry.get_sandbox(
                sandbox_id
            ),
            proxy=lambda *args: {
                "ok": True,
                "forks": list(forks or []),
            },
        )

    def test_get_idle_returns_last_checkpoint_id(self) -> None:
        self.registry.record_idle_reclaim(
            "source",
            action="checkpoint_stop",
            status="succeeded",
            checkpoint_id="ckpt-idle-api",
        )
        response = _GatewayRoutes(self._gateway()).get_idle(
            self.tenant, {}, sandbox_id="source"
        )
        self.assertEqual(response["idle_timeout"], 120.0)
        self.assertEqual(response["idle_action"], "pause")
        self.assertEqual(
            response["last_reclaim"]["checkpoint_id"], "ckpt-idle-api"
        )

    def test_fork_inherits_idle_policy_with_fresh_activity_clock(self) -> None:
        source_activity = self.registry.get_idle_policy("source")["last_activity"]
        routes = _GatewayRoutes(
            self._gateway(forks=[{"sandbox_id": "child"}])
        )
        routes.fork_sandbox(self.tenant, {"count": 1}, sandbox_id="source")
        child = self.registry.get_idle_policy("child")
        assert child is not None
        self.assertEqual(child["idle_timeout"], 120.0)
        self.assertEqual(child["idle_action"], "pause")
        self.assertGreaterEqual(child["last_activity"], source_activity)


class _FakeRegistry:
    def __init__(self, rows) -> None:
        self._rows = rows
        self.statuses: dict[str, str] = {}
        self.reclaims: list[dict] = []

    def list_active_with_idle(self):
        return self._rows

    def release_all_ports(self, sandbox_id):
        return []

    def set_status(self, sandbox_id, status):
        self.statuses[sandbox_id] = status

    def record_idle_reclaim(self, sandbox_id, **result):
        self.reclaims.append({"sandbox_id": sandbox_id, **result})


class _FakePortManager:
    def release_all(self, sandbox_id, ports):
        pass


def _make_fake_gateway(rows, *, checkpoint_result=None):
    gw = types.SimpleNamespace()
    gw.registry = _FakeRegistry(rows)
    gw.proxy_calls = []

    def proxy(method, path, body, timeout):
        gw.proxy_calls.append((method, path))
        if path.endswith("/checkpoint-stop"):
            return checkpoint_result or {
                "ok": True,
                "status": "succeeded",
                "checkpoint_id": "ckpt-idle-1",
                "stopped": True,
            }
        return {"ok": True}

    gw.proxy = proxy
    gw._port_manager = _FakePortManager()
    gw._apply_idle_action = GatewayServer._apply_idle_action.__get__(gw, GatewayServer)
    return gw


def _row(sid, timeout, action, last_activity):
    return {
        "sandbox_id": sid,
        "tenant_id": "t1",
        "status": "active",
        "created_at": "2020-01-01T00:00:00+00:00",
        "last_activity": last_activity,
        "idle_timeout": timeout,
        "idle_action": action,
    }


class IdleSweeperTests(unittest.TestCase):
    def test_reaps_running_idle_sandbox_with_stop(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "stop", _iso(3600))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertIn(("POST", "/sandboxes/sb-1/stop"), gw.proxy_calls)

    def test_skips_sandbox_within_idle_window(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 3600.0, "stop", _iso(10))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertEqual(gw.proxy_calls, [])

    def test_skips_sandbox_not_running(self) -> None:
        # Already paused/stopped -> not re-reaped.
        for status in ("paused", "stopped"):
            gw = _make_fake_gateway([_row("sb-1", 60.0, "stop", _iso(3600))])
            GatewayServer._reap_idle_sandboxes(gw, {"sb-1": status})
            self.assertEqual(gw.proxy_calls, [])

    def test_falls_back_to_created_at_without_last_activity(self) -> None:
        row = _row("sb-1", 60.0, "stop", None)
        gw = _make_fake_gateway([row])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertIn(("POST", "/sandboxes/sb-1/stop"), gw.proxy_calls)

    def test_default_action_is_stop(self) -> None:
        self.assertEqual(DEFAULT_IDLE_ACTION, "stop")

    def test_pause_action_routes_pause(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "pause", _iso(3600))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertIn(("POST", "/sandboxes/sb-1/pause"), gw.proxy_calls)

    def test_kill_action_routes_delete_and_marks_killed(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "kill", _iso(3600))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertIn(("DELETE", "/sandboxes/sb-1"), gw.proxy_calls)
        self.assertEqual(gw.registry.statuses.get("sb-1"), "killed")

    def test_checkpoint_stop_checkpoints_then_stops(self) -> None:
        gw = _make_fake_gateway([_row("sb-1", 60.0, "checkpoint_stop", _iso(3600))])
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertEqual(
            gw.proxy_calls,
            [("POST", "/sandboxes/sb-1/checkpoint-stop")],
        )
        self.assertEqual(gw.registry.reclaims[-1]["status"], "succeeded")
        self.assertEqual(
            gw.registry.reclaims[-1]["checkpoint_id"], "ckpt-idle-1"
        )

    def test_failed_checkpoint_stop_is_recorded_and_not_followed_by_stop(self) -> None:
        gw = _make_fake_gateway(
            [_row("sb-1", 60.0, "checkpoint_stop", _iso(3600))],
            checkpoint_result={
                "ok": False,
                "status": "failed",
                "checkpoint_id": "ckpt-failed-1",
                "stopped": False,
                "message": "checkpoint rejected",
            },
        )
        GatewayServer._reap_idle_sandboxes(gw, {"sb-1": "running"})
        self.assertEqual(
            gw.proxy_calls,
            [("POST", "/sandboxes/sb-1/checkpoint-stop")],
        )
        self.assertEqual(gw.registry.reclaims[-1]["status"], "failed")
        self.assertEqual(
            gw.registry.reclaims[-1]["checkpoint_id"], "ckpt-failed-1"
        )
        self.assertEqual(
            gw.registry.reclaims[-1]["error"], "checkpoint rejected"
        )

    def test_idle_actions_constant(self) -> None:
        self.assertEqual(
            set(IDLE_ACTIONS), {"pause", "stop", "checkpoint_stop", "kill"}
        )


if __name__ == "__main__":
    unittest.main()

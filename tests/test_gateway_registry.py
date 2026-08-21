"""Unit tests for the gateway registry (track S, S1): WAL mode, tenants,
hashed API keys, the two-phase create protocol, quota arithmetic, and
reconciliation helpers. Host-runnable — no daemon, no root."""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from crab.gateway.registry import (
    API_KEY_PREFIX,
    PENDING_ID_PREFIX,
    GatewayRegistry,
    QuotaExceeded,
    hash_api_key,
)


class RegistryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = Path(self._tmp.name) / "gateway.sqlite3"
        self.registry = GatewayRegistry(self.db_path)
        self.addCleanup(self.registry.close)


class WalAndSchemaTests(RegistryTestCase):
    def test_wal_mode_enabled(self) -> None:
        self.assertEqual(self.registry.journal_mode().lower(), "wal")

    def test_state_survives_reopen(self) -> None:
        tenant = self.registry.create_tenant("acme", {"max_sandboxes": 3})
        created = self.registry.create_api_key(tenant["id"])
        self.registry.register_sandbox(tenant["id"], "sb-1")
        self.registry.set_meta("daemon_boot_id", "1234")
        self.registry.close()

        reopened = GatewayRegistry(self.db_path)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.get_tenant(tenant["id"])["quotas"], {"max_sandboxes": 3})
        self.assertEqual(reopened.resolve_api_key(created["api_key"]), tenant["id"])
        self.assertEqual(reopened.get_sandbox("sb-1")["status"], "active")
        self.assertEqual(reopened.get_meta("daemon_boot_id"), "1234")


class TenantTests(RegistryTestCase):
    def test_create_and_list(self) -> None:
        a = self.registry.create_tenant("acme")
        b = self.registry.create_tenant("blit", {"max_sandboxes": 1})
        self.assertTrue(a["id"].startswith("tn_"))
        self.assertEqual(a["quotas"], {})
        names = [t["name"] for t in self.registry.list_tenants()]
        self.assertEqual(names, ["acme", "blit"])
        self.assertEqual(self.registry.get_tenant(b["id"])["quotas"], {"max_sandboxes": 1})

    def test_duplicate_name_rejected(self) -> None:
        self.registry.create_tenant("acme")
        with self.assertRaises(Exception):
            self.registry.create_tenant("acme")

    def test_set_quotas(self) -> None:
        tenant = self.registry.create_tenant("acme")
        updated = self.registry.set_quotas(tenant["id"], {"max_sandboxes": 7})
        self.assertEqual(updated["quotas"], {"max_sandboxes": 7})
        with self.assertRaises(KeyError):
            self.registry.set_quotas("tn_missing", {})


class ApiKeyTests(RegistryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant = self.registry.create_tenant("acme")

    def test_plaintext_shown_once_hash_stored(self) -> None:
        created = self.registry.create_api_key(self.tenant["id"])
        self.assertTrue(created["api_key"].startswith(API_KEY_PREFIX))
        self.assertEqual(created["key_sha256"], hash_api_key(created["api_key"]))
        # The plaintext never appears in the database file.
        raw = self.db_path.read_bytes()
        for sibling in (self.db_path.with_suffix(".sqlite3-wal"),):
            if sibling.exists():
                raw += sibling.read_bytes()
        self.assertNotIn(created["api_key"].encode(), raw)

    def test_resolve_and_revoke(self) -> None:
        created = self.registry.create_api_key(self.tenant["id"])
        self.assertEqual(self.registry.resolve_api_key(created["api_key"]), self.tenant["id"])
        self.assertIsNone(self.registry.resolve_api_key(API_KEY_PREFIX + "0" * 48))
        self.assertTrue(self.registry.revoke_api_key(created["api_key"]))
        self.assertIsNone(self.registry.resolve_api_key(created["api_key"]))

    def test_revoke_by_sha256(self) -> None:
        created = self.registry.create_api_key(self.tenant["id"])
        self.assertTrue(self.registry.revoke_api_key(created["key_sha256"]))
        self.assertIsNone(self.registry.resolve_api_key(created["api_key"]))

    def test_revoke_unknown_returns_false(self) -> None:
        self.assertFalse(self.registry.revoke_api_key("deadbeef"))

    def test_key_for_unknown_tenant_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.create_api_key("tn_missing")


class TwoPhaseCreateTests(RegistryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant = self.registry.create_tenant("acme", {"max_sandboxes": 2})

    def test_begin_complete(self) -> None:
        intent = self.registry.begin_create(self.tenant["id"], name="worker")
        self.assertTrue(intent.startswith(PENDING_ID_PREFIX))
        row = self.registry.get_sandbox(intent)
        self.assertEqual(row["status"], "pending")
        self.assertEqual(row["name"], "worker")

        self.registry.complete_create(intent, "sb-1")
        self.assertIsNone(self.registry.get_sandbox(intent))
        row = self.registry.get_sandbox("sb-1")
        self.assertEqual(row["status"], "active")
        self.assertEqual(row["tenant_id"], self.tenant["id"])

    def test_abort_reaps_pending_row(self) -> None:
        intent = self.registry.begin_create(self.tenant["id"])
        self.registry.abort_create(intent)
        self.assertIsNone(self.registry.get_sandbox(intent))
        self.assertEqual(self.registry.live_count(self.tenant["id"]), 0)

    def test_complete_unknown_intent_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.complete_create(PENDING_ID_PREFIX + "nope", "sb-1")

    def test_pending_rows_count_against_quota(self) -> None:
        self.registry.begin_create(self.tenant["id"])
        self.registry.begin_create(self.tenant["id"])
        with self.assertRaises(QuotaExceeded) as ctx:
            self.registry.begin_create(self.tenant["id"])
        self.assertEqual(ctx.exception.quota["max_sandboxes"], 2)
        self.assertEqual(ctx.exception.quota["live_sandboxes"], 2)

    def test_killed_and_lost_do_not_count(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1", status="killed")
        self.registry.register_sandbox(self.tenant["id"], "sb-2", status="lost")
        self.assertEqual(self.registry.live_count(self.tenant["id"]), 0)
        self.registry.begin_create(self.tenant["id"])  # quota 2, still room

    def test_ensure_capacity_for_fork(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1")
        self.registry.ensure_capacity(self.tenant["id"], additional=1)
        with self.assertRaises(QuotaExceeded):
            self.registry.ensure_capacity(self.tenant["id"], additional=2)

    def test_unknown_tenant_rejected(self) -> None:
        with self.assertRaises(KeyError):
            self.registry.begin_create("tn_missing")

    def test_concurrent_begin_create_respects_quota(self) -> None:
        tenant = self.registry.create_tenant("burst", {"max_sandboxes": 4})
        outcomes: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            try:
                self.registry.begin_create(tenant["id"])
                result = "ok"
            except QuotaExceeded:
                result = "quota"
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=worker) for _ in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(outcomes.count("ok"), 4)
        self.assertEqual(outcomes.count("quota"), 8)
        self.assertEqual(self.registry.live_count(tenant["id"]), 4)


class ReconciliationHelperTests(RegistryTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tenant = self.registry.create_tenant("acme")

    def test_mark_all_active_lost(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1")
        self.registry.register_sandbox(self.tenant["id"], "sb-2", status="killed")
        self.assertEqual(self.registry.mark_all_active_lost(), 1)
        self.assertEqual(self.registry.get_sandbox("sb-1")["status"], "lost")
        self.assertEqual(self.registry.get_sandbox("sb-2")["status"], "killed")

    def test_mark_missing_lost(self) -> None:
        self.registry.register_sandbox(self.tenant["id"], "sb-1")
        self.registry.register_sandbox(self.tenant["id"], "sb-2")
        missing = self.registry.mark_missing_lost({"sb-2"})
        self.assertEqual(missing, ["sb-1"])
        self.assertEqual(self.registry.get_sandbox("sb-1")["status"], "lost")
        self.assertEqual(self.registry.get_sandbox("sb-2")["status"], "active")

    def test_pending_rows_and_tracked_ids(self) -> None:
        intent = self.registry.begin_create(self.tenant["id"])
        self.registry.register_sandbox(self.tenant["id"], "sb-1")
        pending = self.registry.pending_rows()
        self.assertEqual([row["sandbox_id"] for row in pending], [intent])
        self.assertEqual(self.registry.all_tracked_ids(), {"sb-1"})

    def test_meta_roundtrip(self) -> None:
        self.assertIsNone(self.registry.get_meta("daemon_boot_id"))
        self.registry.set_meta("daemon_boot_id", "42")
        self.registry.set_meta("daemon_boot_id", "43")
        self.assertEqual(self.registry.get_meta("daemon_boot_id"), "43")


if __name__ == "__main__":
    unittest.main()

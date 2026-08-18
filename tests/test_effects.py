"""Unit tests for D3.1: the effect gate — policy decisions, the defer
allow-list, the queue, and the proxy integration (202/503 shapes).
Host-runnable; the proxy is driven over loopback."""
from __future__ import annotations

import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from crab.effects import (
    DECISION_DEFER,
    DECISION_PASS,
    DECISION_REJECT,
    DECISION_SEAL,
    DeferredRequest,
    EffectGate,
    EffectRule,
    build_deferred_request,
    read_remaining_body,
)
from crab.egress import CassetteRecorder, EgressProxyServer, EgressRule
from crab.cassettes import CassetteStore
from crab.http_wire import parse_head
from crab.ids import SandboxId
from crab.journal import ActionJournal
from crab.models import EgressFlow, EgressLedger


class RuleMatchingTests(unittest.TestCase):
    def test_globs_and_method(self) -> None:
        rule = EffectRule(
            host_glob="*.internal.example", method="POST", path_glob="/events*"
        )
        self.assertTrue(
            rule.matches(host="metrics.internal.example", method="post", path="/events/1")
        )
        self.assertFalse(
            rule.matches(host="metrics.internal.example", method="PUT", path="/events/1")
        )
        self.assertFalse(
            rule.matches(host="api.example.com", method="POST", path="/events/1")
        )
        self.assertFalse(
            rule.matches(host="metrics.internal.example", method="POST", path="/other")
        )

    def test_defaults_match_everything(self) -> None:
        rule = EffectRule()
        self.assertTrue(rule.matches(host="anything", method="DELETE", path="/x"))

    def test_from_json_and_defer_false(self) -> None:
        rule = EffectRule.from_json(
            {"host_glob": "*.x", "method": "post", "path_glob": "/a*", "defer": False}
        )
        self.assertFalse(rule.defer)
        self.assertEqual(rule.method, "post")
        self.assertTrue(rule.matches(host="h.x", method="POST", path="/ab"))


class PolicyDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = EffectGate()
        self.sandbox_id = "sbx-1"

    def _decide(self, **overrides) -> str:
        kwargs = {"host": "api.example.com", "method": "POST", "path": "/orders"}
        kwargs.update(overrides)
        decision, _ = self.gate.decide_write(self.sandbox_id, **kwargs)
        return decision

    def test_no_session_is_inert(self) -> None:
        self.assertIsNone(
            self.gate.decide_write(
                "sbx-none", host="api.example.com", method="POST", path="/x"
            )
        )
        self.assertIsNone(self.gate.decide_opaque("sbx-none"))

    def test_allow_passes(self) -> None:
        self.gate.begin(self.sandbox_id, policy="allow")
        self.assertEqual(self._decide(), DECISION_PASS)
        self.assertEqual(self.gate.counters(self.sandbox_id), (0, 0, 1, False))

    def test_reject_refuses(self) -> None:
        self.gate.begin(self.sandbox_id, policy="reject")
        self.assertEqual(self._decide(), DECISION_REJECT)
        self.assertEqual(self.gate.counters(self.sandbox_id), (0, 1, 0, False))

    def test_seal_passes_but_marks_the_txn(self) -> None:
        self.gate.begin(self.sandbox_id, policy="seal")
        self.assertFalse(self.gate.sealed(self.sandbox_id))
        self.assertEqual(self._decide(), DECISION_SEAL)
        self.assertTrue(self.gate.sealed(self.sandbox_id))

    def test_defer_needs_the_allow_list(self) -> None:
        # Empty list + defer: refused, never silently queued.
        self.gate.begin(self.sandbox_id, policy="defer")
        self.assertEqual(self._decide(), DECISION_REJECT)

        self.gate.begin(
            self.sandbox_id,
            policy="defer",
            rules=(EffectRule(host_glob="api.example.com", method="POST", path_glob="/orders*"),),
        )
        self.assertEqual(self._decide(), DECISION_DEFER)
        # A different endpoint still falls to on_unlisted.
        self.assertEqual(self._decide(path="/refunds"), DECISION_REJECT)

    def test_on_unlisted_allow(self) -> None:
        self.gate.begin(self.sandbox_id, policy="defer", on_unlisted="allow")
        self.assertEqual(self._decide(), DECISION_PASS)

    def test_rule_can_exempt_an_endpoint_from_deferral(self) -> None:
        self.gate.begin(
            self.sandbox_id,
            policy="defer",
            rules=(EffectRule(host_glob="*", defer=False),),
        )
        self.assertEqual(self._decide(), DECISION_PASS)

    def test_first_matching_rule_wins(self) -> None:
        self.gate.begin(
            self.sandbox_id,
            policy="defer",
            rules=(
                EffectRule(host_glob="*", defer=False),
                EffectRule(host_glob="api.example.com", defer=True),
            ),
        )
        self.assertEqual(self._decide(), DECISION_PASS)

    def test_opaque_behaviors(self) -> None:
        self.gate.begin(self.sandbox_id, policy="reject", opaque_effects="allow")
        decision, _ = self.gate.decide_opaque(self.sandbox_id)
        self.assertEqual(decision, DECISION_PASS)  # HTTPS keeps working

        self.gate.begin(self.sandbox_id, policy="reject", opaque_effects="reject")
        decision, _ = self.gate.decide_opaque(self.sandbox_id)
        self.assertEqual(decision, DECISION_REJECT)

        self.gate.begin(self.sandbox_id, policy="allow", opaque_effects="seal")
        decision, _ = self.gate.decide_opaque(self.sandbox_id)
        self.assertEqual(decision, DECISION_SEAL)
        self.assertTrue(self.gate.sealed(self.sandbox_id))

    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.gate.begin(self.sandbox_id, policy="hold")
        with self.assertRaises(ValueError):
            self.gate.begin(self.sandbox_id, policy="defer", on_unlisted="queue")
        with self.assertRaises(ValueError):
            self.gate.begin(self.sandbox_id, policy="defer", opaque_effects="defer")

    def test_session_lifecycle(self) -> None:
        session = self.gate.begin(self.sandbox_id, policy="reject", txn_id="txn-1")
        self.assertIs(self.gate.session_for(self.sandbox_id), session)
        self.assertEqual(session.txn_id, "txn-1")
        self.assertIs(self.gate.end(self.sandbox_id), session)
        self.assertIsNone(self.gate.session_for(self.sandbox_id))
        self.assertIsNone(self.gate.end(self.sandbox_id))  # idempotent


class QueueTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = EffectGate()
        self.gate.begin("sbx-1", policy="defer", rules=(EffectRule(),), txn_id="txn-9")

    def _request(self, path: str) -> DeferredRequest:
        return DeferredRequest(
            method="POST", host="api.example.com", port=80, path=path,
            headers=(("Content-Type", "application/json"),), body=b"{}",
            txn_id="txn-9",
        )

    def test_enqueue_and_drain_preserve_order(self) -> None:
        self.assertEqual(self.gate.enqueue("sbx-1", self._request("/a")), 1)
        self.assertEqual(self.gate.enqueue("sbx-1", self._request("/b")), 2)
        self.assertEqual(self.gate.counters("sbx-1")[0], 2)
        drained = self.gate.drain("sbx-1")
        self.assertEqual([entry.path for entry in drained], ["/a", "/b"])
        # Draining empties the queue (commit flushes, abort discards).
        self.assertEqual(self.gate.drain("sbx-1"), [])

    def test_enqueue_without_session_is_a_noop(self) -> None:
        self.assertEqual(self.gate.enqueue("sbx-other", self._request("/a")), 0)
        self.assertEqual(self.gate.drain("sbx-other"), [])

    def test_body_digest(self) -> None:
        request = self._request("/a")
        self.assertEqual(len(request.body_sha256), 64)


class ConcurrencyTests(unittest.TestCase):
    """One thread per connection in the proxy, so the queue and counters
    are touched concurrently — the D2.2 review's lesson applied up front."""

    def test_counters_and_queue_survive_parallel_writers(self) -> None:
        gate = EffectGate()
        gate.begin("sbx-1", policy="defer", rules=(EffectRule(),))
        rounds, workers = 150, 8

        def hammer() -> None:
            for index in range(rounds):
                gate.decide_write(
                    "sbx-1", host="api.example.com", method="POST", path=f"/p{index}"
                )
                gate.enqueue(
                    "sbx-1",
                    DeferredRequest(
                        method="POST", host="api.example.com", port=80,
                        path=f"/p{index}", headers=(), body=b"",
                    ),
                )

        threads = [threading.Thread(target=hammer) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        deferred, rejected, passed, sealed = gate.counters("sbx-1")
        self.assertEqual(deferred, rounds * workers)
        self.assertEqual((rejected, passed, sealed), (0, 0, False))
        self.assertEqual(len(gate.drain("sbx-1")), rounds * workers)


class BodyReadingTests(unittest.TestCase):
    """A deferred write must be queued whole: only the first peek reached
    the proxy, so a body that is still on the socket has to be drained
    before the request can be replayed at commit."""

    def _pair(self):
        left, right = socket.socketpair()
        self.addCleanup(left.close)
        self.addCleanup(right.close)
        return left, right

    def test_reads_the_rest_of_a_framed_body(self) -> None:
        head = parse_head(b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 10\r\n\r\nabc")
        left, right = self._pair()
        right.sendall(b"defghij")
        body, complete = read_remaining_body(left, head, head.rest, limit=1024)
        self.assertEqual(body, b"abcdefghij")
        self.assertTrue(complete)

    def test_already_complete_body_is_returned_as_is(self) -> None:
        head = parse_head(b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 3\r\n\r\nabc")
        left, _ = self._pair()
        body, complete = read_remaining_body(left, head, head.rest, limit=1024)
        self.assertEqual((body, complete), (b"abc", True))

    def test_oversized_body_is_incomplete(self) -> None:
        head = parse_head(b"POST /x HTTP/1.1\r\nHost: h\r\nContent-Length: 5000\r\n\r\nabc")
        left, _ = self._pair()
        body, complete = read_remaining_body(left, head, head.rest, limit=64)
        self.assertFalse(complete)

    def test_unframed_body_cannot_be_deferred(self) -> None:
        chunked = parse_head(
            b"POST /x HTTP/1.1\r\nHost: h\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n"
        )
        left, _ = self._pair()
        _, complete = read_remaining_body(left, chunked, chunked.rest, limit=1024)
        self.assertFalse(complete)
        # A bodyless request (no Content-Length, nothing buffered) is fine.
        bodyless = parse_head(b"DELETE /x HTTP/1.1\r\nHost: h\r\n\r\n")
        body, complete = read_remaining_body(left, bodyless, bodyless.rest, limit=1024)
        self.assertEqual((body, complete), (b"", True))

    def test_build_strips_credentials(self) -> None:
        head = parse_head(
            b"POST /x HTTP/1.1\r\nHost: h\r\nAuthorization: Bearer secret\r\n"
            b"Cookie: sid=1\r\nContent-Type: application/json\r\n\r\n"
        )
        request = build_deferred_request(
            parsed_head=head, body=b'{"a":1}', host="h", port=80, method="post",
            path="/x", txn_id="txn-1", enqueued_at="now",
        )
        names = {name.lower() for name, _ in request.headers}
        self.assertNotIn("authorization", names)
        self.assertNotIn("cookie", names)
        self.assertIn("content-type", names)
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.body, b'{"a":1}')


class _Upstream:
    def __init__(self, response: bytes = b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok"):
        self.response = response
        self.received: list[bytes] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                try:
                    self.received.append(conn.recv(8192))
                    conn.sendall(self.response)
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()


class ProxyEffectGateTests(unittest.TestCase):
    """End-to-end through the proxy: the decisive property is that a
    refused or deferred write never reaches the upstream."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_effects_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.sandbox_id = SandboxId("sbx-effects")
        self.upstream = _Upstream()
        self.addCleanup(self.upstream.close)
        self.gate = EffectGate()
        self.proxy = EgressProxyServer(
            journal=self.journal,
            sandbox_id_resolver=lambda peer: self.sandbox_id,
            host="127.0.0.1",
            port=0,
            head_timeout_seconds=1.0,
            effect_gate=self.gate,
        )
        self.proxy.start()
        self.addCleanup(self.proxy.stop)
        patcher = mock.patch(
            "crab.egress.original_destination",
            return_value=("127.0.0.1", self.upstream.port),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _talk(self, payload: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", self.proxy.port), timeout=5.0) as sock:
            sock.sendall(payload)
            chunks = []
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks)

    def _flows(self, *, expected: int = 1, timeout: float = 5.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        rows: list = []
        while time.monotonic() < deadline:
            rows = self.journal.entries(self.sandbox_id, kind="egress")
            if len(rows) >= expected:
                break
            time.sleep(0.05)
        return [row.payload for row in rows]

    _POST = (
        b"POST /orders HTTP/1.1\r\nHost: api.example.com\r\n"
        b"Content-Length: 7\r\n\r\n{\"a\":1}"
    )

    def test_reject_never_reaches_upstream(self) -> None:
        self.gate.begin(self.sandbox_id, policy="reject", txn_id="txn-1")
        response = self._talk(self._POST)
        self.assertIn(b"503", response)
        self.assertIn(b"X-Crab-Effect: rejected", response)
        self.assertEqual(self.upstream.received, [])
        [flow] = self._flows()
        self.assertEqual(flow["effect"], "rejected")

    def test_defer_queues_the_whole_request_and_answers_202(self) -> None:
        self.gate.begin(
            self.sandbox_id, policy="defer", rules=(EffectRule(),), txn_id="txn-2"
        )
        response = self._talk(self._POST)
        self.assertIn(b"202", response)
        self.assertIn(b"X-Crab-Effect: deferred", response)
        self.assertEqual(self.upstream.received, [])  # nothing left the host

        [flow] = self._flows()
        self.assertEqual(flow["effect"], "deferred")
        self.assertEqual(flow["effect_queue_position"], 1)
        [queued] = self.gate.drain(self.sandbox_id)
        self.assertEqual(queued.method, "POST")
        self.assertEqual(queued.path, "/orders")
        self.assertEqual(queued.body, b'{"a":1}')  # body captured whole
        self.assertEqual(queued.txn_id, "txn-2")

    def test_reads_are_never_gated(self) -> None:
        self.upstream.response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi"
        self.gate.begin(self.sandbox_id, policy="reject", txn_id="txn-3")
        response = self._talk(b"GET /things HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self.assertIn(b"200", response)
        self.assertEqual(len(self.upstream.received), 1)
        [flow] = self._flows()
        self.assertIsNone(flow.get("effect"))

    def test_seal_lets_the_write_out_and_marks_the_session(self) -> None:
        self.gate.begin(self.sandbox_id, policy="seal", txn_id="txn-4")
        response = self._talk(self._POST)
        self.assertIn(b"201", response)
        self.assertEqual(len(self.upstream.received), 1)
        self.assertTrue(self.gate.sealed(self.sandbox_id))
        [flow] = self._flows()
        self.assertEqual(flow["effect"], "sealed")

    def test_no_session_behaves_exactly_as_before(self) -> None:
        response = self._talk(self._POST)
        self.assertIn(b"201", response)
        self.assertEqual(len(self.upstream.received), 1)
        [flow] = self._flows()
        self.assertIsNone(flow.get("effect"))

    def test_unqueueable_body_is_refused_not_queued(self) -> None:
        self.gate.begin(
            self.sandbox_id, policy="defer", rules=(EffectRule(),), txn_id="txn-5"
        )
        self.proxy.max_deferred_body_bytes = 4
        response = self._talk(self._POST)  # 7-byte body vs a 4-byte cap
        self.assertIn(b"503", response)
        self.assertEqual(self.upstream.received, [])
        self.assertEqual(self.gate.drain(self.sandbox_id), [])
        [flow] = self._flows()
        self.assertEqual(flow["effect"], "rejected")
        self.assertEqual(flow["effect_reason"], "unqueueable")

    def test_opaque_flow_follows_opaque_effects(self) -> None:
        self.gate.begin(self.sandbox_id, policy="reject", opaque_effects="reject")
        response = self._talk(b"\x16\x03\x01\x00\x50 not really tls")
        self.assertIn(b"503", response)
        self.assertEqual(self.upstream.received, [])


class LedgerEffectFieldsTests(unittest.TestCase):
    def test_counters_and_round_trip(self) -> None:
        def flow(seq: int, effect: str | None) -> EgressFlow:
            return EgressFlow(
                seq=seq, host="api.example.com", dst_ip="10.0.0.1", dst_port=80,
                scheme="http", classification="mutating", method="POST", path="/w",
                effect=effect,
            )

        ledger = EgressLedger(
            sandbox_id=SandboxId("sbx-1"),
            flows=(flow(1, "deferred"), flow(2, "rejected"), flow(3, "flushed"), flow(4, None)),
        )
        self.assertEqual((ledger.deferred, ledger.rejected, ledger.flushed), (1, 1, 1))
        payload = ledger.to_json()
        self.assertEqual(payload["deferred"], 1)
        self.assertEqual(EgressLedger.from_json(payload), ledger)

    def test_pre_d3_rows_have_no_effect(self) -> None:
        flow = EgressFlow.from_json(
            {"seq": 1, "host": "h", "dst_ip": "1.2.3.4", "dst_port": 80,
             "scheme": "http", "classification": "mutating"}
        )
        self.assertIsNone(flow.effect)
        self.assertIsNone(flow.effect_status)


class ConfigTests(unittest.TestCase):
    def test_defaults_and_nested_parsing(self) -> None:
        from crab.engine import EngineConfig

        cfg = EngineConfig()
        self.assertEqual(cfg.effects_default_policy, "allow")
        self.assertEqual(cfg.effects_fork_policy, "reject")
        self.assertEqual(cfg.effects_on_unlisted, "reject")
        self.assertEqual(cfg.effects_opaque_effects, "allow")
        self.assertEqual(cfg.effects_rules, ())

        parsed = EngineConfig.from_mapping(
            {
                "effects": {
                    "default_policy": "defer",
                    "fork_policy": "allow",
                    "on_unlisted": "allow",
                    "opaque_effects": "seal",
                    "rules": [{"host_glob": "*.internal", "method": "POST"}],
                }
            }
        )
        self.assertEqual(parsed.effects_default_policy, "defer")
        self.assertEqual(parsed.effects_fork_policy, "allow")
        self.assertEqual(parsed.effects_on_unlisted, "allow")
        self.assertEqual(parsed.effects_opaque_effects, "seal")
        self.assertEqual(len(parsed.effects_rules), 1)


if __name__ == "__main__":
    unittest.main()

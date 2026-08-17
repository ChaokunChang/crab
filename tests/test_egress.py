"""Unit tests for D1.1 egress interception: sniffers, SO_ORIGINAL_DST
decoding, the proxy's splice/attribution/journal behavior (over
loopback sockets, no netfilter), the bridge redirect rule shape, and
the engine config gates. Host-runnable."""
from __future__ import annotations

import socket
import struct
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from crab.egress import (
    EgressFlowRecorder,
    EgressProxyServer,
    parse_original_dst,
    sniff_http_head,
    sniff_tls_sni,
)
from crab.ids import SandboxId
from crab.journal import ActionJournal


def _client_hello(server_name: bytes | None) -> bytes:
    """A minimal but structurally valid TLS 1.2 ClientHello."""
    extensions = b""
    if server_name is not None:
        name_entry = b"\x00" + struct.pack("!H", len(server_name)) + server_name
        sni_body = struct.pack("!H", len(name_entry)) + name_entry
        extensions = struct.pack("!HH", 0x0000, len(sni_body)) + sni_body
    body = (
        b"\x03\x03"  # client_version
        + b"\x11" * 32  # random
        + b"\x00"  # session id length
        + struct.pack("!H", 2) + b"\x13\x01"  # cipher suites
        + b"\x01\x00"  # compression methods
        + struct.pack("!H", len(extensions)) + extensions
    )
    handshake = b"\x01" + len(body).to_bytes(3, "big") + body
    return b"\x16\x03\x01" + struct.pack("!H", len(handshake)) + handshake


class SnifferTests(unittest.TestCase):
    def test_http_head(self) -> None:
        head = b"POST /v1/things?a=b HTTP/1.1\r\nHost: api.example.com\r\nX: y\r\n\r\n{}"
        self.assertEqual(
            sniff_http_head(head), ("POST", "/v1/things?a=b", "api.example.com")
        )

    def test_http_head_without_host(self) -> None:
        self.assertEqual(
            sniff_http_head(b"GET / HTTP/1.0\r\n\r\n"), ("GET", "/", "")
        )

    def test_http_head_rejects_non_http(self) -> None:
        self.assertIsNone(sniff_http_head(b"\x16\x03\x01\x00\x50 garbage\r\n"))
        self.assertIsNone(sniff_http_head(b"HELLO world\r\n"))
        self.assertIsNone(sniff_http_head(b"no newline yet"))
        # A valid-looking verb with a bogus protocol token is not HTTP.
        self.assertIsNone(sniff_http_head(b"GET / SSH-2.0\r\n\r\n"))

    def test_tls_sni(self) -> None:
        self.assertEqual(
            sniff_tls_sni(_client_hello(b"secure.example.com")), "secure.example.com"
        )

    def test_tls_without_sni_and_non_tls(self) -> None:
        self.assertIsNone(sniff_tls_sni(_client_hello(None)))
        self.assertIsNone(sniff_tls_sni(b"GET / HTTP/1.1\r\nHost: x\r\n\r\n"))
        self.assertIsNone(sniff_tls_sni(b"\x16\x03\x01"))  # truncated

    def test_original_dst_decoding(self) -> None:
        packed = struct.pack("!HH", socket.AF_INET, 8443) + socket.inet_aton("10.250.0.9") + b"\x00" * 8
        self.assertEqual(parse_original_dst(packed), ("10.250.0.9", 8443))
        with self.assertRaises(ValueError):
            parse_original_dst(b"\x02\x00")


class _EchoUpstream:
    """Trivial TCP echo server standing in for the "external world"."""

    def __init__(self, response: bytes = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi") -> None:
        self.response = response
        self.received: list[bytes] = []
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                try:
                    self.received.append(conn.recv(4096))
                    conn.sendall(self.response)
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()
        self._thread.join(timeout=2.0)


class ProxyFlowTests(unittest.TestCase):
    """Drives the proxy over loopback with SO_ORIGINAL_DST patched (no
    netfilter needed to prove the flow/record behavior)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_egress_")
        self.addCleanup(self._tmp.cleanup)
        self.journal = ActionJournal(Path(self._tmp.name) / "journal")
        self.sandbox_id = SandboxId("sbx-egress")
        self.upstream = _EchoUpstream()
        self.addCleanup(self.upstream.close)
        self.proxy = EgressProxyServer(
            journal=self.journal,
            sandbox_id_resolver=lambda peer: self.sandbox_id if peer == "127.0.0.1" else None,
            host="127.0.0.1",
            port=0,
            head_timeout_seconds=1.0,
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
        """Flows are recorded when the connection completes, which is just
        after the client sees EOF — poll instead of racing it."""
        deadline = time.monotonic() + timeout
        records: list = []
        while time.monotonic() < deadline:
            records = self.journal.entries(self.sandbox_id, kind="egress")
            if len(records) >= expected:
                break
            time.sleep(0.05)
        return [record.payload for record in records]

    def _no_flows(self, *, settle: float = 1.0) -> list[dict]:
        time.sleep(settle)
        return [
            record.payload
            for record in self.journal.entries(self.sandbox_id, kind="egress")
        ]

    def test_http_flow_recorded_and_spliced(self) -> None:
        response = self._talk(
            b"POST /v1/orders HTTP/1.1\r\nHost: api.example.com\r\nContent-Length: 2\r\n\r\nhi"
        )
        self.assertIn(b"200 OK", response)
        self.assertIn(b"/v1/orders", self.upstream.received[0])
        [flow] = self._flows()
        self.assertEqual(flow["scheme"], "http")
        self.assertEqual(flow["method"], "POST")
        self.assertEqual(flow["path"], "/v1/orders")
        self.assertEqual(flow["host"], "api.example.com")
        self.assertEqual(flow["dst_port"], self.upstream.port)
        self.assertEqual(flow["classification"], "unclassified")  # PR-D1.2 fills this
        self.assertGreater(flow["bytes_out"], 0)
        self.assertGreater(flow["bytes_in"], 0)

    def test_tls_flow_records_sni_without_decrypting(self) -> None:
        hello = _client_hello(b"secure.example.com")
        self._talk(hello)
        [flow] = self._flows()
        self.assertEqual(flow["scheme"], "tls")
        self.assertEqual(flow["host"], "secure.example.com")
        self.assertIsNone(flow["method"])
        # The ClientHello reached the upstream byte-identical.
        self.assertEqual(self.upstream.received[0][: len(hello)], hello)

    def test_opaque_flow_falls_back_to_ip(self) -> None:
        self._talk(b"\x00\x01\x02binary-protocol\n")
        [flow] = self._flows()
        self.assertEqual(flow["scheme"], "tcp")
        self.assertEqual(flow["host"], "127.0.0.1")
        self.assertIsNone(flow["method"])

    def test_unattributed_flow_is_not_recorded(self) -> None:
        self.proxy._sandbox_id_resolver = lambda peer: None
        self._talk(b"GET / HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self.assertEqual(self._no_flows(), [])

    def test_txn_stamp_rides_along(self) -> None:
        self.journal.set_active_txn(self.sandbox_id, "txn-1")
        self.addCleanup(self.journal.set_active_txn, self.sandbox_id, None)
        self._talk(b"GET /health HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self._flows()
        [record] = self.journal.entries(self.sandbox_id, kind="egress")
        self.assertEqual(record.txn_id, "txn-1")

    def test_non_redirected_connection_is_dropped(self) -> None:
        with mock.patch("crab.egress.original_destination", return_value=None):
            self._talk(b"GET / HTTP/1.1\r\nHost: api.example.com\r\n\r\n")
        self.assertEqual(self._no_flows(), [])
        self.assertEqual(self.upstream.received, [])

    def test_recorder_swallows_journal_failures(self) -> None:
        broken = mock.Mock()
        broken.record_egress.side_effect = RuntimeError("disk on fire")
        EgressFlowRecorder(broken).record(self.sandbox_id, {"host": "x"})
        EgressFlowRecorder(None).record(self.sandbox_id, {"host": "x"})


class RedirectRuleTests(unittest.TestCase):
    """The rule shape IS the "LLM interception unchanged" guarantee, so
    it is asserted literally."""

    def _manager(self):
        from integrations.sandboxes.runtime.network import BenchmarkNetworkManager

        manager = BenchmarkNetworkManager()
        manager._bridge_name = "acbdeadbeef"
        return manager

    def test_enable_adds_bridge_rule_excluding_the_host(self) -> None:
        manager = self._manager()
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            run.return_value = mock.Mock(returncode=1)  # -C says "absent"
            manager.enable_egress_redirect(19999)
        appended = [
            call.args[0] for call in run.call_args_list if "-A" in call.args[0]
        ]
        self.assertEqual(
            appended,
            [
                [
                    "iptables", "-t", "nat", "-A", "PREROUTING",
                    "-i", "acbdeadbeef", "-p", "tcp",
                    "!", "-d", manager.bridge_ip,
                    "-j", "REDIRECT", "--to-ports", "19999",
                ]
            ],
        )

    def test_enable_is_idempotent_and_checks_first(self) -> None:
        manager = self._manager()
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            run.return_value = mock.Mock(returncode=0)  # -C says "present"
            manager.enable_egress_redirect(19999)
            manager.enable_egress_redirect(19999)  # cached: no further calls
        self.assertEqual(
            [call.args[0][3] for call in run.call_args_list], ["-C"]
        )

    def test_disable_deletes_the_same_rule(self) -> None:
        manager = self._manager()
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            run.return_value = mock.Mock(returncode=1)
            manager.enable_egress_redirect(18888)
            run.reset_mock()
            manager.disable_egress_redirect()
        [deleted] = [call.args[0] for call in run.call_args_list]
        self.assertEqual(deleted[:5], ["iptables", "-t", "nat", "-D", "PREROUTING"])
        self.assertIn("18888", deleted)
        # Second disable is a no-op (nothing left to remove).
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            manager.disable_egress_redirect()
        run.assert_not_called()

    def test_enable_requires_the_bridge(self) -> None:
        from integrations.sandboxes.runtime.network import BenchmarkNetworkManager

        with self.assertRaises(RuntimeError):
            BenchmarkNetworkManager().enable_egress_redirect(1234)

    def test_cleanup_removes_the_redirect(self) -> None:
        manager = self._manager()
        with mock.patch(
            "integrations.sandboxes.runtime.network.subprocess.run"
        ) as run:
            run.return_value = mock.Mock(returncode=1)
            manager.enable_egress_redirect(17777)
            run.reset_mock()
            manager.cleanup()
        self.assertTrue(
            any(
                "REDIRECT" in call.args[0] and "-D" in call.args[0]
                for call in run.call_args_list
            ),
            "cleanup did not drop the redirect rule",
        )


class ConfigGateTests(unittest.TestCase):
    def test_defaults_off(self) -> None:
        from crab.engine import EngineConfig

        cfg = EngineConfig()
        self.assertFalse(cfg.enable_egress_proxy)
        self.assertEqual(cfg.egress_proxy_port, 0)

    def test_from_mapping_nested_and_flat(self) -> None:
        from crab.engine import EngineConfig

        nested = EngineConfig.from_mapping({"egress": {"enabled": True, "port": 5555}})
        self.assertTrue(nested.enable_egress_proxy)
        self.assertEqual(nested.egress_proxy_port, 5555)
        flat = EngineConfig.from_mapping(
            {"enable_egress_proxy": True, "egress_proxy_port": 6666}
        )
        self.assertTrue(flat.enable_egress_proxy)
        self.assertEqual(flat.egress_proxy_port, 6666)

    def test_requires_sandbox_network(self) -> None:
        from crab.engine import Engine, EngineConfig

        with tempfile.TemporaryDirectory(prefix="crab_egress_cfg_") as tmp:
            cfg = EngineConfig(
                runtime="in-memory",
                enable_interceptor=False,
                enable_sandbox_network=False,
                enable_egress_proxy=True,
                storage_root=Path(tmp) / "storage",
            )
            with self.assertRaises(RuntimeError) as ctx:
                Engine.start(cfg)
            self.assertIn("enable_egress_proxy requires", str(ctx.exception))

    def test_requires_the_action_journal(self) -> None:
        from crab.engine import Engine, EngineConfig

        with tempfile.TemporaryDirectory(prefix="crab_egress_cfg_") as tmp:
            cfg = EngineConfig(
                runtime="in-memory",
                enable_interceptor=False,
                enable_sandbox_network=True,
                enable_egress_proxy=True,
                enable_action_journal=False,
                storage_root=Path(tmp) / "storage",
            )
            with self.assertRaises(RuntimeError) as ctx:
                Engine.start(cfg)
            # Without the ledger's store the proxy would forward everything
            # and record nothing — indistinguishable from working.
            self.assertIn("enable_action_journal", str(ctx.exception))

    def test_default_bind_is_loopback_not_wildcard(self) -> None:
        from crab.egress import EgressProxyServer

        proxy = EgressProxyServer()
        self.addCleanup(proxy.stop)
        self.assertEqual(proxy.server_address[0], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()

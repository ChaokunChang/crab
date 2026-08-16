"""Real-host end-to-end for D1.1 egress interception: the repository's
first ``enable_sandbox_network=True`` test. A sandbox on the bridge
network reaches a host-side "external" service; every TCP flow is
redirected into the egress proxy and lands in the sandbox's journal
with the host/method/scheme the proxy could observe. Self-skipping
outside the crab-dev VM."""
from __future__ import annotations

import http.server
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from crab import Engine, EngineConfig, Sandbox


def _real_stack_available() -> bool:
    if os.geteuid() != 0:
        return False
    tools = ("docker", "runc", "criu", "zfs", "iptables", "ip")
    return all(shutil.which(tool) is not None for tool in tools)


def _host_lan_ip() -> str | None:
    """An address of this host that a sandbox on the bridge can reach
    which is NOT the bridge IP itself (host-bound traffic is excluded
    from the redirect on purpose)."""
    result = subprocess.run(
        ["ip", "-4", "-o", "addr", "show", "scope", "global"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[1].startswith(("acb", "lo", "vh")):
            continue
        return fields[3].split("/")[0]
    return None


class _ExternalHTTPServer:
    """Stands in for the outside world: bound on all interfaces so the
    sandbox reaches it through the bridge (not via the host address the
    redirect rule excludes)."""

    class _Handler(http.server.BaseHTTPRequestHandler):
        def _respond(self) -> None:
            body = b"external-ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        do_GET = _respond
        do_POST = _respond

        def log_message(self, *args) -> None:  # silence
            pass

    def __init__(self) -> None:
        self._server = http.server.ThreadingHTTPServer(("0.0.0.0", 0), self._Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)


class _RawTCPServer:
    """Opaque (non-HTTP, non-TLS) listener for the raw-flow case."""

    def __init__(self) -> None:
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", 0))
        self._sock.listen(4)
        self.port = self._sock.getsockname()[1]
        self.received: list[bytes] = []
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
                    self.received.append(conn.recv(256))
                    conn.sendall(b"raw-ok\n")
                except OSError:
                    pass

    def close(self) -> None:
        self._sock.close()
        self._thread.join(timeout=2.0)


class EgressRealTests(unittest.TestCase):
    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self.host_ip = _host_lan_ip()
        if self.host_ip is None:
            self.skipTest("no global-scope host address for the external service")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_egress_e2e_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.external = _ExternalHTTPServer()
        self.addCleanup(self.external.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=False,
                enable_egress_proxy=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _run(self, sandbox: Sandbox, script: str) -> str:
        result = sandbox.commands.run(script)
        self.assertEqual(result.returncode, 0, msg=f"command failed: {script!r}: {result.stderr}")
        return result.stdout.strip()

    def _flows(self, sandbox: Sandbox, *, expected: int, timeout: float = 20.0) -> list[dict]:
        deadline = time.monotonic() + timeout
        rows: list[dict] = []
        while time.monotonic() < deadline:
            rows = [row["payload"] for row in sandbox.actions(kind="egress")]
            if len(rows) >= expected:
                return rows
            time.sleep(0.25)
        self.fail(f"expected >={expected} egress flows, saw {len(rows)}: {rows}")

    def test_all_sandbox_egress_lands_in_the_ledger(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        # Egress interception only works when the sandbox sits in the
        # bridge netns (the REDIRECT hook point) — assert it up front so a
        # host-networked sandbox can never masquerade as a passing test.
        self.assertIsNotNone(
            self.engine._network_manager.lease_for(sandbox.sandbox_id),
            "sandbox has no bridge network lease; egress would bypass the proxy",
        )
        base = f"http://{self.host_ip}:{self.external.port}"

        # 1. plain HTTP GET + POST through the redirect
        self.assertIn(
            "external-ok",
            self._run(sandbox, f"python3 -c \"import urllib.request;print(urllib.request.urlopen('{base}/read').read().decode())\""),
        )
        self._run(
            sandbox,
            f"python3 -c \"import urllib.request;urllib.request.urlopen(urllib.request.Request('{base}/write', data=b'x', method='POST')).read()\"",
        )
        flows = self._flows(sandbox, expected=2)
        by_path = {flow.get("path"): flow for flow in flows if flow.get("path")}
        self.assertIn("/read", by_path, msg=str(flows))
        self.assertIn("/write", by_path, msg=str(flows))
        self.assertEqual(by_path["/read"]["method"], "GET")
        self.assertEqual(by_path["/write"]["method"], "POST")
        for path in ("/read", "/write"):
            flow = by_path[path]
            self.assertEqual(flow["scheme"], "http")
            self.assertEqual(flow["dst_ip"], self.host_ip)
            self.assertEqual(flow["dst_port"], self.external.port)
            self.assertIn(self.host_ip, flow["host"])  # Host header carries ip:port
            self.assertGreater(flow["bytes_out"], 0)
            self.assertGreater(flow["bytes_in"], 0)
            self.assertEqual(flow["classification"], "unclassified")  # PR-D1.2

    def test_tls_sni_and_raw_tcp_are_recorded(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        raw = _RawTCPServer()
        self.addCleanup(raw.close)

        # TLS: the handshake fails (no server), but the ClientHello's SNI
        # is what the ledger needs — no decryption involved.
        sandbox.commands.run(
            "python3 -c \""
            "import socket,ssl\n"
            "ctx=ssl.create_default_context()\n"
            "ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE\n"
            f"s=socket.create_connection(('{self.host_ip}',{self.external.port}),timeout=5)\n"
            "try:\n"
            "    ctx.wrap_socket(s,server_hostname='secure.example.com').close()\n"
            "except Exception: pass\""
        )
        self._run(
            sandbox,
            f"python3 -c \"import socket;s=socket.create_connection(('{self.host_ip}',{raw.port}),timeout=5);s.sendall(b'\\x01\\x02opaque');print(s.recv(16).decode().strip());s.close()\"",
        )

        flows = self._flows(sandbox, expected=2)
        tls = [flow for flow in flows if flow["scheme"] == "tls"]
        opaque = [flow for flow in flows if flow["dst_port"] == raw.port]
        self.assertTrue(tls, msg=f"no TLS flow recorded: {flows}")
        self.assertEqual(tls[0]["host"], "secure.example.com")
        self.assertIsNone(tls[0]["method"])
        self.assertTrue(opaque, msg=f"no raw TCP flow recorded: {flows}")
        self.assertEqual(opaque[0]["scheme"], "tcp")
        self.assertEqual(opaque[0]["host"], self.host_ip)
        self.assertTrue(raw.received and raw.received[0].endswith(b"opaque"))

    def test_txn_correlation(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        base = f"http://{self.host_ip}:{self.external.port}"
        txn = sandbox.begin("egress-txn")
        txn.exec(
            f"python3 -c \"import urllib.request;urllib.request.urlopen(urllib.request.Request('{base}/in-txn', data=b'x', method='POST')).read()\""
        )
        self._flows(sandbox, expected=1)
        txn.commit()
        rows = [row for row in sandbox.actions(kind="egress") if row["payload"].get("path") == "/in-txn"]
        self.assertTrue(rows, "in-txn flow missing from the ledger")
        self.assertEqual(rows[0]["txn_id"], txn.txn_id)

    def test_redirect_rule_is_scoped_and_removed(self) -> None:
        rules = subprocess.run(
            ["iptables", "-t", "nat", "-S", "PREROUTING"],
            capture_output=True, text=True, check=True,
        ).stdout
        redirects = [line for line in rules.splitlines() if "REDIRECT" in line]
        self.assertTrue(redirects, "no REDIRECT rule installed")
        # Host-bound traffic must be excluded, or the LLM interceptor path
        # would be swallowed by the proxy.
        self.assertTrue(
            all("! -d " in line for line in redirects),
            f"redirect rule is not host-scoped: {redirects}",
        )
        self.engine.stop()
        after = subprocess.run(
            ["iptables", "-t", "nat", "-S", "PREROUTING"],
            capture_output=True, text=True, check=True,
        ).stdout
        self.assertNotIn("REDIRECT", after, "redirect rule leaked after engine stop")


class EgressLlmUnchangedRealTests(unittest.TestCase):
    """Exit criterion: with the egress proxy on, the LLM interception
    path behaves exactly as before (host-bound traffic is excluded from
    the redirect)."""

    _IMAGE = "python:3.11-slim"

    def setUp(self) -> None:
        if not _real_stack_available():
            self.skipTest("docker/runc/criu/zfs/iptables/root not available")
        self._tmp = tempfile.TemporaryDirectory(prefix="crab_egress_llm_")
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.upstream_hits: list[str] = []
        self.upstream = self._start_upstream()
        self.addCleanup(self.upstream.close)
        self.engine = Engine.start(
            EngineConfig(
                runtime="runc",
                enable_sandbox_network=True,
                enable_interceptor=True,
                enable_egress_proxy=True,
                storage_root=root / "storage",
                runtime_root=root / "runtime",
            )
        )
        self.addCleanup(self.engine.stop)

    def _start_upstream(self):
        hits = self.upstream_hits

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                hits.append(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(length)
                body = b'{"id":"cmpl-1","choices":[{"message":{"content":"pong"}}]}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args) -> None:
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()

        class _Holder:
            def __init__(self, srv):
                self._srv = srv
                self.port = srv.server_address[1]

            def close(self):
                self._srv.shutdown()
                self._srv.server_close()

        return _Holder(server)

    def test_llm_request_still_reaches_the_interceptor(self) -> None:
        sandbox = Sandbox(image=self._IMAGE, engine=self.engine)
        self.addCleanup(sandbox.kill)
        self.assertIsNotNone(
            self.engine._network_manager.lease_for(sandbox.sandbox_id),
            "sandbox has no bridge lease; the ledger check would be vacuous",
        )
        self.engine.register_upstream(
            sandbox.sandbox_id, f"http://127.0.0.1:{self.upstream.port}"
        )
        interceptor_url = self.engine.interceptor_base_url
        self.assertIsNotNone(interceptor_url)
        result = sandbox.commands.run(
            "python3 -c \"import json,urllib.request;"
            f"req=urllib.request.Request('{interceptor_url}/v1/chat/completions',"
            "data=json.dumps({'model':'m','messages':[{'role':'user','content':'ping'}]}).encode(),"
            "headers={'Content-Type':'application/json'});"
            "print(urllib.request.urlopen(req, timeout=30).read().decode())\""
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("pong", result.stdout)
        self.assertEqual(self.upstream_hits, ["/v1/chat/completions"])
        # The interceptor path is host-bound, so it must NOT be in the
        # egress ledger (the redirect rule excludes it).
        time.sleep(1.0)
        ledger_hosts = [row["payload"]["host"] for row in sandbox.actions(kind="egress")]
        self.assertFalse(
            any("127.0.0.1" in host for host in ledger_hosts),
            f"interceptor traffic was redirected into the proxy: {ledger_hosts}",
        )


if __name__ == "__main__":
    unittest.main()

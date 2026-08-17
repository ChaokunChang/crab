"""Transparent egress interception (roadmap D1).

All sandbox TCP egress is redirected (iptables REDIRECT on the bridge)
into ``EgressProxyServer``, which recovers the original destination via
``SO_ORIGINAL_DST``, sniffs just enough of the head to identify the
flow (HTTP request line + Host, or a TLS ClientHello's SNI), splices
the bytes both ways untouched, and appends one ``kind="egress"``
journal record per connection.

Deliberate limits (see .cache/tasks/egress-ledger.md decision log):
no TLS MITM in v1 — the SNI gives host-level identification, bodies
are D2's charter; host-bound traffic is never redirected, so the LLM
interceptor path stays byte-identical; TCP only.
"""
from __future__ import annotations

import logging
import socket
import socketserver
import struct
import threading
import time
from typing import Callable

from .ids import SandboxId

logger = logging.getLogger(__name__)

# Linux: getsockopt(SOL_IP, SO_ORIGINAL_DST) yields the pre-REDIRECT
# destination as a sockaddr_in.
SO_ORIGINAL_DST = 80

_HEAD_PEEK_BYTES = 2048
_SPLICE_CHUNK = 65536
_HTTP_METHODS = frozenset(
    {"GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE", "CONNECT"}
)

SandboxIdResolver = Callable[[str], "SandboxId | None"]


def parse_original_dst(packed: bytes) -> tuple[str, int]:
    """Decode a ``sockaddr_in`` from SO_ORIGINAL_DST into (ip, port)."""
    if len(packed) < 8:
        raise ValueError(f"short sockaddr_in: {len(packed)} bytes")
    port, = struct.unpack("!H", packed[2:4])
    return socket.inet_ntoa(packed[4:8]), int(port)


def original_destination(sock: socket.socket) -> tuple[str, int] | None:
    """The connection's pre-redirect destination, or None when the
    socket was not redirected (direct connect, or a platform without
    SO_ORIGINAL_DST)."""
    try:
        packed = sock.getsockopt(socket.SOL_IP, SO_ORIGINAL_DST, 16)
    except OSError:
        return None
    try:
        return parse_original_dst(packed)
    except ValueError:
        return None


def sniff_http_head(head: bytes) -> tuple[str, str, str] | None:
    """``(method, path, host)`` for a plaintext HTTP request head, or
    None when the bytes are not HTTP. ``host`` is "" when the request
    carries no Host header (HTTP/1.0)."""
    newline = head.find(b"\r\n")
    if newline < 0:
        return None
    parts = head[:newline].decode("latin-1").split(" ")
    if len(parts) != 3 or not parts[2].startswith("HTTP/"):
        return None
    method, path = parts[0], parts[1]
    if method not in _HTTP_METHODS:
        return None
    host = ""
    for raw_line in head[newline + 2:].split(b"\r\n"):
        if not raw_line:
            break
        name, _, value = raw_line.partition(b":")
        if name.strip().lower() == b"host":
            host = value.strip().decode("latin-1")
            break
    return method, path, host


def sniff_tls_sni(head: bytes) -> str | None:
    """The SNI host from a TLS ClientHello, or None when the bytes are
    not a ClientHello or carry no SNI extension. Parses only — nothing
    is decrypted or rewritten."""
    # TLS record: type(1) version(2) length(2); handshake: type(1) len(3)
    if len(head) < 45 or head[0] != 0x16 or head[5] != 0x01:
        return None
    try:
        pos = 9 + 2 + 32  # handshake header + client_version + random
        session_len = head[pos]
        pos += 1 + session_len
        cipher_len, = struct.unpack("!H", head[pos:pos + 2])
        pos += 2 + cipher_len
        compression_len = head[pos]
        pos += 1 + compression_len
        if pos + 2 > len(head):
            return None
        extensions_len, = struct.unpack("!H", head[pos:pos + 2])
        pos += 2
        end = min(pos + extensions_len, len(head))
        while pos + 4 <= end:
            ext_type, ext_len = struct.unpack("!HH", head[pos:pos + 4])
            pos += 4
            if ext_type == 0x0000:  # server_name
                # list_len(2) name_type(1) name_len(2) name
                if pos + 5 > len(head):
                    return None
                name_len, = struct.unpack("!H", head[pos + 3:pos + 5])
                name = head[pos + 5:pos + 5 + name_len]
                # SNI is an A-label (ASCII); IDN hosts arrive punycoded and
                # stay that way — the ledger records host identity, not a
                # display form (the idna codec rejects error handlers).
                return name.decode("ascii", errors="replace").lower() if name else None
            pos += ext_len
    except (IndexError, struct.error):
        return None
    return None


class EgressFlowRecorder:
    """Journal-backed sink for completed flows (the effect ledger's only
    store — same ruling as C3: the journal is the durable, ordered,
    txn-stamped account). Recording failures never break the flow."""

    def __init__(self, journal) -> None:
        self._journal = journal

    def record(self, sandbox_id: SandboxId, payload: dict) -> None:
        if self._journal is None:
            return
        try:
            self._journal.record_egress(sandbox_id, payload=payload)
        except Exception:
            logger.debug("Failed to record egress flow", exc_info=True)


class _EgressHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:  # noqa: C901 - straight-line proxy flow
        server: "EgressProxyServer" = self.server  # type: ignore[assignment]
        client = self.request
        peer_ip = self.client_address[0]
        destination = original_destination(client)
        if destination is None:
            # Not redirected: nothing to proxy to. Reject rather than
            # guess a destination (a direct hit on the proxy port).
            logger.debug("Dropping non-redirected connection from %s", peer_ip)
            return
        dst_ip, dst_port = destination
        sandbox_id = server.resolve_sandbox_id(peer_ip)
        started = time.monotonic()
        head = b""
        scheme = "tcp"
        method: str | None = None
        path: str | None = None
        host = ""
        bytes_out = 0
        bytes_in = 0
        upstream: socket.socket | None = None
        try:
            client.settimeout(server.head_timeout_seconds)
            try:
                head = client.recv(_HEAD_PEEK_BYTES)
            except (socket.timeout, OSError):
                head = b""
            client.settimeout(None)
            http = sniff_http_head(head)
            if http is not None:
                scheme = "http"
                method, path, host = http
            else:
                sni = sniff_tls_sni(head)
                if sni is not None:
                    scheme = "tls"
                    host = sni
            upstream = socket.create_connection(
                (dst_ip, dst_port), timeout=server.connect_timeout_seconds
            )
            upstream.settimeout(None)
            if head:
                upstream.sendall(head)
                bytes_out += len(head)
            out_extra, in_extra = _splice(client, upstream)
            bytes_out += out_extra
            bytes_in += in_extra
        except OSError as exc:
            logger.debug(
                "Egress flow to %s:%s failed: %s", dst_ip, dst_port, exc, exc_info=True
            )
        finally:
            for sock in (upstream, client):
                if sock is not None:
                    try:
                        sock.close()
                    except OSError:
                        pass
            if sandbox_id is None:
                # Unattributable flow (no lease for this peer): counted in
                # the log, not in any sandbox's ledger.
                logger.debug(
                    "Unattributed egress flow peer=%s dst=%s:%s", peer_ip, dst_ip, dst_port
                )
                return
            payload = {
                "host": host or dst_ip,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "scheme": scheme,
                "method": method,
                "path": path,
                "bytes_out": bytes_out,
                "bytes_in": bytes_in,
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                # PR-D1.2 fills this in; the field exists from day one so
                # stored rows never need a schema migration.
                "classification": "unclassified",
            }
            server.recorder.record(sandbox_id, payload)


def _splice(left: socket.socket, right: socket.socket) -> tuple[int, int]:
    """Pump bytes both ways until either side closes. Returns
    ``(left->right, right->left)`` byte counts."""
    counters = [0, 0]

    def pump(src: socket.socket, dst: socket.socket, slot: int) -> None:
        try:
            while True:
                chunk = src.recv(_SPLICE_CHUNK)
                if not chunk:
                    break
                dst.sendall(chunk)
                counters[slot] += len(chunk)
        except OSError:
            pass
        finally:
            try:
                dst.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    upstream_pump = threading.Thread(
        target=pump, args=(left, right, 0), daemon=True, name="crab-egress-out"
    )
    upstream_pump.start()
    pump(right, left, 1)
    upstream_pump.join(timeout=5.0)
    return counters[0], counters[1]


class EgressProxyServer(socketserver.ThreadingTCPServer):
    """Host-side transparent proxy for redirected sandbox egress."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        *,
        journal=None,
        sandbox_id_resolver: SandboxIdResolver | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        head_timeout_seconds: float = 2.0,
        connect_timeout_seconds: float = 10.0,
    ) -> None:
        """``host`` should be the bridge's own address: REDIRECT rewrites
        each flow's destination to it, so binding there receives all
        redirected traffic without exposing the proxy elsewhere."""
        super().__init__((host, port), _EgressHandler)
        self.recorder = EgressFlowRecorder(journal)
        self._sandbox_id_resolver = sandbox_id_resolver
        self.head_timeout_seconds = head_timeout_seconds
        self.connect_timeout_seconds = connect_timeout_seconds
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    def resolve_sandbox_id(self, peer_ip: str) -> SandboxId | None:
        if self._sandbox_id_resolver is None:
            return None
        try:
            return self._sandbox_id_resolver(peer_ip)
        except Exception:
            logger.debug("Egress sandbox attribution failed peer=%s", peer_ip, exc_info=True)
            return None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self.serve_forever, daemon=True, name="crab-egress-proxy"
        )
        self._thread.start()
        logger.info("Egress proxy listening on port %d", self.port)

    def stop(self) -> None:
        # shutdown() waits for the serve_forever loop to acknowledge, so it
        # would hang forever on a server that was never started.
        if self._thread is not None:
            self.shutdown()
        self.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


__all__ = [
    "SO_ORIGINAL_DST",
    "EgressFlowRecorder",
    "EgressProxyServer",
    "original_destination",
    "parse_original_dst",
    "sniff_http_head",
    "sniff_tls_sni",
]

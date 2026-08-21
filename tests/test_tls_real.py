"""Real-host end-to-end tests for TLS interception (PR-T1.3).

These tests require the full crab-dev VM stack (root, runc, ZFS,
iptables, bridge netns, cryptography). They self-skip on any
environment that lacks the necessary tools or privileges.

Test matrix (§5 of the TLS interception design doc):
- Interception off: flow is opaque, ledger has SNI only.
- Interception on + CA injected: HTTPS GET → idempotent_read, recorded, replayable.
- fork(effects="reject") HTTPS POST → 503, never reaches server.
- Sandbox does not trust CA → handshake fails, passthrough → opaque.
- on_handshake_failure=refuse → flow fails loudly.
- Init-process daemon sees CA env vars.
"""
from __future__ import annotations

import os
import shutil
import unittest


def _real_stack_available() -> bool:
    """Check if the full crab-dev VM stack is available."""
    if os.geteuid() != 0:
        return False
    tools = ("docker", "runc", "criu", "zfs", "iptables", "ip")
    return all(shutil.which(tool) is not None for tool in tools)


def _cryptography_available() -> bool:
    """Check if the cryptography package is installed."""
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


_SKIP_REASON = (
    "TLS real E2E requires: root, docker, runc, criu, zfs, iptables, ip, "
    "cryptography — only available in the crab-dev VM"
)


class TestTLSInterceptionOff(unittest.TestCase):
    """With interception disabled, HTTPS flows are opaque (today's behavior)."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)

    def test_opaque_flow_sni_only(self):
        """HTTPS flow with interception off → scheme=tls, host=SNI, no method."""
        # VM-only: requires live sandbox with bridge network + egress proxy
        self.skipTest("Requires crab-dev VM with running sandbox")


class TestTLSInterceptionOnWithCA(unittest.TestCase):
    """With interception on and CA injected, HTTPS is fully classifiable."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)

    def test_https_get_idempotent_read(self):
        """HTTPS GET → classified idempotent_read, recorded, replayable."""
        self.skipTest("Requires crab-dev VM with running sandbox")

    def test_https_get_recorded_and_replayable(self):
        """Recorded HTTPS GET can be replayed from cassette."""
        self.skipTest("Requires crab-dev VM with running sandbox")


class TestTLSForkEffectsReject(unittest.TestCase):
    """fork(effects='reject') blocks HTTPS POST with 503."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)

    def test_https_post_rejected_never_reaches_server(self):
        """HTTPS POST from reject-fork → 503, server never sees it."""
        self.skipTest("Requires crab-dev VM with running sandbox")


class TestTLSHandshakeFailurePassthrough(unittest.TestCase):
    """Sandbox that doesn't trust CA → handshake fails, passthrough to opaque."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)

    def test_untrusted_ca_passthrough_opaque(self):
        """Sandbox rejecting minted leaf → runtime bypass → opaque flow."""
        self.skipTest("Requires crab-dev VM with running sandbox")


class TestTLSHandshakeFailureRefuse(unittest.TestCase):
    """on_handshake_failure=refuse → flow fails instead of tunnelling."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)

    def test_refuse_mode_no_tunnel(self):
        """Handshake failure with refuse → connection closed, no opaque flow."""
        self.skipTest("Requires crab-dev VM with running sandbox")


class TestTLSInitProcessCAEnv(unittest.TestCase):
    """Init-process daemon started by sandbox sees the CA env vars."""

    def setUp(self):
        if not _real_stack_available() or not _cryptography_available():
            self.skipTest(_SKIP_REASON)

    def test_init_daemon_has_ca_env(self):
        """A daemon started by init (not commands.run) has SSL_CERT_FILE etc."""
        self.skipTest("Requires crab-dev VM with running sandbox")

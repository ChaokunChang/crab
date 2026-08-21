"""TLS CA store and leaf certificate minting for egress interception.

This module is **not** imported by any resident code path. It is loaded
lazily only when TLS interception is enabled at runtime, so deployments
that never enable ``crab[tls]`` need not install ``cryptography``.

Public API
----------
- ``CAStore(directory)`` — load-or-generate a self-signed CA.
- ``LeafMinter(ca_store)`` — mint short-lived leaf certs keyed by host (SNI).
"""

from __future__ import annotations

import datetime
import ipaddress
import os
import stat
import threading
from pathlib import Path
from typing import Tuple

# Lazy import: cryptography is an optional dependency (crab[tls]).
# Importing this module without the package installed will raise
# ImportError at class instantiation, not at module import time.
try:
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    _HAS_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    _HAS_CRYPTOGRAPHY = False

__all__ = ["CAStore", "LeafMinter"]

_CA_KEY_BITS = 4096
_LEAF_KEY_BITS = 2048
_CA_VALIDITY_DAYS = 3650  # ~10 years
_LEAF_VALIDITY_HOURS = 24


def _require_cryptography() -> None:
    if not _HAS_CRYPTOGRAPHY:
        raise ImportError(
            "TLS interception requires the 'cryptography' package. "
            "Install it with: pip install 'crab[tls]'"
        )


def _is_ip(host: str) -> bool:
    """Return True if *host* is an IP address literal."""
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


class CAStore:
    """Persistent CA key-pair store.

    On first use, generates a self-signed CA certificate and RSA private key
    in *directory*.  Subsequent instantiations with the same directory load the
    existing material.  The private key file is always chmod 0600.

    Parameters
    ----------
    directory : str | Path
        Directory to store ``ca.crt`` and ``ca.key``.
    """

    def __init__(self, directory: str | Path) -> None:
        _require_cryptography()
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cert_path = self._dir / "ca.crt"
        self._key_path = self._dir / "ca.key"
        self._cert: x509.Certificate
        self._key: rsa.RSAPrivateKey
        self._cert, self._key = self._load_or_generate()

    @property
    def cert(self) -> "x509.Certificate":
        """The CA certificate."""
        return self._cert

    @property
    def key(self) -> "rsa.RSAPrivateKey":
        """The CA private key."""
        return self._key

    @property
    def cert_path(self) -> Path:
        """Path to the PEM-encoded CA certificate file."""
        return self._cert_path

    @property
    def key_path(self) -> Path:
        """Path to the PEM-encoded CA private key file."""
        return self._key_path

    def cert_pem(self) -> bytes:
        """Return the CA certificate as PEM bytes."""
        return self._cert.public_bytes(serialization.Encoding.PEM)

    def _load_or_generate(
        self,
    ) -> Tuple["x509.Certificate", "rsa.RSAPrivateKey"]:
        if self._cert_path.exists() and self._key_path.exists():
            return self._load()
        return self._generate()

    def _load(self) -> Tuple["x509.Certificate", "rsa.RSAPrivateKey"]:
        cert = x509.load_pem_x509_certificate(
            self._cert_path.read_bytes()
        )
        key = serialization.load_pem_private_key(
            self._key_path.read_bytes(), password=None
        )
        # Enforce key permission even on load.
        self._secure_key_file()
        return cert, key  # type: ignore[return-value]

    def _generate(self) -> Tuple["x509.Certificate", "rsa.RSAPrivateKey"]:
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=_CA_KEY_BITS,
        )
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Crab TLS Interception"),
            x509.NameAttribute(NameOID.COMMON_NAME, "Crab Local CA"),
        ])
        now = datetime.datetime.now(datetime.timezone.utc)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(now + datetime.timedelta(days=_CA_VALIDITY_DAYS))
            .add_extension(
                x509.BasicConstraints(ca=True, path_length=0),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_cert_sign=True,
                    crl_sign=True,
                    content_commitment=False,
                    key_encipherment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(key, hashes.SHA256())
        )
        # Write key first (most sensitive).
        self._key_path.write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.TraditionalOpenSSL,
                serialization.NoEncryption(),
            )
        )
        self._secure_key_file()
        self._cert_path.write_bytes(
            cert.public_bytes(serialization.Encoding.PEM)
        )
        return cert, key

    def _secure_key_file(self) -> None:
        """Ensure key file is mode 0600."""
        os.chmod(self._key_path, stat.S_IRUSR | stat.S_IWUSR)


class LeafMinter:
    """Mint short-lived leaf certificates on demand, cached by host.

    Leaf certificates are SAN-only: hostname hosts get a ``DNSName`` SAN,
    IP-literal hosts get an ``IPAddress`` SAN. Validity is 24 hours.
    Certificates are signed by the CA from *ca_store*.

    The minter is thread-safe; concurrent requests for the same host will
    produce the same cached certificate.

    Parameters
    ----------
    ca_store : CAStore
        The CA whose key signs each leaf.
    """

    def __init__(self, ca_store: CAStore) -> None:
        _require_cryptography()
        self._ca = ca_store
        self._cache: dict[str, Tuple[x509.Certificate, rsa.RSAPrivateKey]] = {}
        self._lock = threading.Lock()

    def get_or_mint(
        self, host: str
    ) -> Tuple["x509.Certificate", "rsa.RSAPrivateKey"]:
        """Return ``(leaf_cert, leaf_key)`` for *host*, minting if needed.

        Results are cached in memory keyed by host.
        """
        with self._lock:
            if host in self._cache:
                return self._cache[host]
        # Mint outside the lock (crypto is CPU-bound), then re-check.
        cert, key = self._mint(host)
        with self._lock:
            # Double-check: another thread may have minted while we were busy.
            if host not in self._cache:
                self._cache[host] = (cert, key)
            return self._cache[host]

    def get_cert_and_key_pem(self, host: str) -> Tuple[bytes, bytes]:
        """Return ``(cert_pem, key_pem)`` bytes suitable for ssl.SSLContext.

        Example::

            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            cert_pem, key_pem = minter.get_cert_and_key_pem(host)
            ctx.load_cert_chain(certfile=..., keyfile=...)

        For in-memory loading without temp files, write to a
        ``tempfile.NamedTemporaryFile`` or use ``ctx.load_cert_chain``
        with file objects (Python 3.12+).
        """
        cert, key = self.get_or_mint(host)
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
        return cert_pem, key_pem

    def _mint(
        self, host: str
    ) -> Tuple["x509.Certificate", "rsa.RSAPrivateKey"]:
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=_LEAF_KEY_BITS,
        )
        now = datetime.datetime.now(datetime.timezone.utc)

        # Build SAN: DNS for hostnames, IP for IP-literals.
        if _is_ip(host):
            san = x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.ip_address(host))
            ])
        else:
            san = x509.SubjectAlternativeName([x509.DNSName(host)])

        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, host),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(self._ca.cert.subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now)
            .not_valid_after(
                now + datetime.timedelta(hours=_LEAF_VALIDITY_HOURS)
            )
            .add_extension(san, critical=False)
            .add_extension(
                x509.BasicConstraints(ca=False, path_length=None),
                critical=True,
            )
            .add_extension(
                x509.KeyUsage(
                    digital_signature=True,
                    key_encipherment=True,
                    content_commitment=False,
                    data_encipherment=False,
                    key_agreement=False,
                    key_cert_sign=False,
                    crl_sign=False,
                    encipher_only=False,
                    decipher_only=False,
                ),
                critical=True,
            )
            .sign(self._ca.key, hashes.SHA256())
        )
        return cert, key

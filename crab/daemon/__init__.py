"""Crab daemon — the long-running host service that owns the Engine.

Operators start the daemon once per host (`python -m crab.daemon`) and
the SDK connects to it through `Engine.connect(...)`. The daemon is the
sole owner of runc state, ZFS datasets, the host inspector, the LLM
interceptor and forwarder, and the network bridge — running two daemons
or mixing a daemon with an in-process engine on the same host is not
supported (they would race on the same runtime paths).

Public surface:
  - `DaemonServer` (server.py): wraps an in-process Engine and exposes it
    over HTTP-over-Unix-socket.
  - `DaemonClient` (transport.py): used by both the SDK proxy and the
    `crab` CLI to talk to the daemon.
  - `default_socket_path()` (transport.py): the conventional path the
    daemon listens on (`$XDG_RUNTIME_DIR/crab/crab.sock` for users,
    `/run/crab/crab.sock` for root).
"""
from __future__ import annotations

from .transport import DaemonClient, DaemonRequestError, default_socket_path

__all__ = [
    "DaemonClient",
    "DaemonRequestError",
    "default_socket_path",
]

"""`crab` CLI — operator front-end for the Crab daemon.

The CLI talks to the daemon's Unix-socket HTTP API via
`crab.daemon.DaemonClient`. The same wire format is shared with the
SDK proxy (`crab.remote_engine.RemoteEngine`), so anything the CLI
can do, the SDK can do too — and vice versa — without duplicating code.

Subcommand groups in this v1 cut:
  - `crab daemon start|stop|status`
  - `crab info`
  - `crab sandbox ls|rm`
  - `crab sandbox exec <id> -- <argv...>`

Use `crab --help` to discover.
"""
from .commands import main

__all__ = ["main"]

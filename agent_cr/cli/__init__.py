"""`agentcr` CLI — operator front-end for the Agent-CR daemon.

The CLI talks to the daemon's Unix-socket HTTP API via
`agent_cr.daemon.DaemonClient`. The same wire format is shared with the
SDK proxy (`agent_cr.remote_engine.RemoteEngine`), so anything the CLI
can do, the SDK can do too — and vice versa — without duplicating code.

Subcommand groups in this v1 cut:
  - `agentcr daemon start|stop|status`
  - `agentcr info`
  - `agentcr sandbox ls|rm`
  - `agentcr sandbox exec <id> -- <argv...>`

Use `agentcr --help` to discover.
"""
from .commands import main

__all__ = ["main"]

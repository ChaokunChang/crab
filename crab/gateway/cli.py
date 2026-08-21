"""crab-gateway CLI — `serve` plus the local-only admin plane.

`serve` runs the gateway process. The admin subcommands (`tenants`,
`keys`, `quotas`) talk to a *running* gateway over its own Unix socket
(design doc §4 S1: the admin surface is served locally, never over the
TCP listener), reusing `DaemonClient` — the admin plane speaks the same
HTTP-over-Unix-socket JSON the daemon does.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from ..daemon.transport import DaemonClient, DaemonRequestError
from .server import (
    DEFAULT_BIND_HOST,
    DEFAULT_BIND_PORT,
    GatewayServer,
    default_admin_socket_path,
    default_data_dir,
)

logger = logging.getLogger(__name__)


def _admin_client(args: argparse.Namespace) -> DaemonClient:
    socket_path = args.admin_socket or default_admin_socket_path()
    return DaemonClient(socket_path)


def _print_json(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _run_admin(args: argparse.Namespace, method: str, path: str, body: dict[str, Any] | None) -> int:
    client = _admin_client(args)
    try:
        if method == "GET":
            result = client.get_json(path)
        else:
            result = client.post_json(path, body)
    except FileNotFoundError:
        print(
            f"crab-gateway not reachable at {client.socket_path}; "
            "is the gateway running (`crab-gateway serve`)?",
            file=sys.stderr,
        )
        return 1
    except DaemonRequestError as exc:
        print(f"crab-gateway admin request failed: {exc}", file=sys.stderr)
        return 1
    _print_json(result)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
        force=True,
    )
    host, _, port_raw = args.bind.rpartition(":")
    if not host or not port_raw.isdigit():
        print(f"invalid --bind (expected HOST:PORT): {args.bind}", file=sys.stderr)
        return 2
    gateway = GatewayServer(
        data_dir=args.data_dir,
        daemon_socket=args.daemon_socket,
        host=host,
        port=int(port_raw),
        admin_socket_path=args.admin_socket,
    )
    try:
        gateway.start()
    except Exception:
        logger.exception("gateway failed to start")
        return 1
    gateway.serve_forever()
    return 0


def _cmd_tenants_create(args: argparse.Namespace) -> int:
    quotas: dict[str, Any] = {}
    if args.max_sandboxes is not None:
        quotas["max_sandboxes"] = args.max_sandboxes
    body: dict[str, Any] = {"name": args.name}
    if quotas:
        body["quotas"] = quotas
    return _run_admin(args, "POST", "/admin/tenants", body)


def _cmd_tenants_list(args: argparse.Namespace) -> int:
    return _run_admin(args, "GET", "/admin/tenants", None)


def _cmd_keys_create(args: argparse.Namespace) -> int:
    return _run_admin(args, "POST", "/admin/keys", {"tenant_id": args.tenant})


def _cmd_keys_revoke(args: argparse.Namespace) -> int:
    return _run_admin(args, "POST", "/admin/keys/revoke", {"key": args.key})


def _cmd_quotas_set(args: argparse.Namespace) -> int:
    quotas: dict[str, Any] = {}
    if args.max_sandboxes is not None:
        quotas["max_sandboxes"] = args.max_sandboxes
    return _run_admin(args, "POST", "/admin/quotas", {"tenant_id": args.tenant, "quotas": quotas})


def _add_admin_socket_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--admin-socket",
        type=Path,
        default=None,
        help="Gateway admin Unix socket path (default: $CRAB_GATEWAY_SOCKET "
        "or the conventional runtime-dir location).",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="crab-gateway",
        description="Multi-tenant service gateway in front of a crab daemon.",
    )
    sub = parser.add_subparsers(dest="command")

    serve_p = sub.add_parser("serve", help="Run the gateway (default when no subcommand).")
    serve_p.add_argument(
        "--bind",
        default=f"{DEFAULT_BIND_HOST}:{DEFAULT_BIND_PORT}",
        help="HOST:PORT for the public HTTP listener (default loopback; "
        "TLS termination belongs to a reverse proxy).",
    )
    serve_p.add_argument(
        "--daemon-socket",
        type=Path,
        default=None,
        help="Crab daemon Unix socket to front (default: $CRAB_DAEMON_SOCKET "
        "or the daemon's conventional path).",
    )
    serve_p.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=f"Registry/state directory (default: {default_data_dir()}).",
    )
    _add_admin_socket_arg(serve_p)
    serve_p.add_argument(
        "--log-level",
        default=os.environ.get("CRAB_GATEWAY_LOG_LEVEL", "INFO"),
        help="Log level (DEBUG/INFO/WARNING/ERROR).",
    )
    serve_p.set_defaults(fn=_cmd_serve)

    tenants_p = sub.add_parser("tenants", help="Tenant administration (local-only).")
    tenants_sub = tenants_p.add_subparsers(dest="tenants_command", required=True)
    tenants_create = tenants_sub.add_parser("create", help="Create a tenant.")
    tenants_create.add_argument("name", help="Human-readable unique tenant name.")
    tenants_create.add_argument(
        "--max-sandboxes",
        type=int,
        default=None,
        help="Quota: maximum live sandboxes (unset = unlimited).",
    )
    _add_admin_socket_arg(tenants_create)
    tenants_create.set_defaults(fn=_cmd_tenants_create)
    tenants_list = tenants_sub.add_parser("list", help="List tenants.")
    _add_admin_socket_arg(tenants_list)
    tenants_list.set_defaults(fn=_cmd_tenants_list)

    keys_p = sub.add_parser("keys", help="API key administration (local-only).")
    keys_sub = keys_p.add_subparsers(dest="keys_command", required=True)
    keys_create = keys_sub.add_parser(
        "create", help="Mint a key; the plaintext is shown once, only here."
    )
    keys_create.add_argument("--tenant", required=True, help="Tenant id the key belongs to.")
    _add_admin_socket_arg(keys_create)
    keys_create.set_defaults(fn=_cmd_keys_create)
    keys_revoke = keys_sub.add_parser("revoke", help="Revoke a key.")
    keys_revoke.add_argument("key", help="Plaintext key (crab_sk_...) or its sha256 digest.")
    _add_admin_socket_arg(keys_revoke)
    keys_revoke.set_defaults(fn=_cmd_keys_revoke)

    quotas_p = sub.add_parser("quotas", help="Quota administration (local-only).")
    quotas_sub = quotas_p.add_subparsers(dest="quotas_command", required=True)
    quotas_set = quotas_sub.add_parser("set", help="Replace a tenant's quotas.")
    quotas_set.add_argument("--tenant", required=True, help="Tenant id.")
    quotas_set.add_argument(
        "--max-sandboxes",
        type=int,
        default=None,
        help="Quota: maximum live sandboxes (omit to clear the cap).",
    )
    _add_admin_socket_arg(quotas_set)
    quotas_set.set_defaults(fn=_cmd_quotas_set)

    args = parser.parse_args(argv)
    if getattr(args, "fn", None) is None:
        # Bare `python -m crab.gateway` runs the server, like the daemon.
        args = parser.parse_args(["serve", *(argv or sys.argv[1:])])
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import ipaddress
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
import uuid

from agent_cr import SandboxId

logger = logging.getLogger(__name__)


def parse_ipv4_route_networks(raw_routes: str) -> list[ipaddress.IPv4Network]:
    networks: list[ipaddress.IPv4Network] = []
    for raw_line in raw_routes.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        destination = line.split()[0]
        if destination == "default":
            continue
        try:
            network = ipaddress.ip_network(destination, strict=False)
        except ValueError:
            continue
        if isinstance(network, ipaddress.IPv4Network) and network.prefixlen < 32:
            networks.append(network)
    return networks


def select_benchmark_network(*, existing_routes: str, candidate_pool: str = "10.250.0.0/16") -> tuple[str, str]:
    pool = ipaddress.ip_network(candidate_pool, strict=False)
    if not isinstance(pool, ipaddress.IPv4Network):
        raise ValueError(f"candidate pool must be IPv4, got {candidate_pool}")
    if pool.prefixlen > 24:
        raise ValueError(f"candidate pool must be at most /24, got {candidate_pool}")
    existing_networks = parse_ipv4_route_networks(existing_routes)
    candidates = [pool] if pool.prefixlen == 24 else list(pool.subnets(new_prefix=24))
    for network in candidates:
        if any(network.overlaps(existing) for existing in existing_networks):
            continue
        return str(next(network.hosts())), str(network)
    raise RuntimeError(f"unable to find an available benchmark /24 inside {candidate_pool}")


@dataclass(frozen=True)
class BenchmarkNetworkLease:
    sandbox_id: SandboxId
    namespace_name: str
    namespace_path: Path
    host_veth_name: str
    guest_veth_name: str
    guest_ip: str


class BenchmarkNetworkManager:
    def __init__(self) -> None:
        self._bridge_name: str | None = None
        self._bridge_ip = "10.250.0.1"
        self._network_cidr = "10.250.0.0/24"
        self._ip_cursor = 2
        self._leases: dict[SandboxId, BenchmarkNetworkLease] = {}
        self._ip_to_sandbox: dict[str, SandboxId] = {}
        self._lock = threading.Lock()

    @property
    def bridge_ip(self) -> str:
        return self._bridge_ip

    @property
    def network_cidr(self) -> str:
        return self._network_cidr

    def configure(self) -> None:
        configured_cidr = os.environ.get("AGENT_CR_BENCHMARK_NETWORK_CIDR", "").strip()
        if configured_cidr:
            network = ipaddress.ip_network(configured_cidr, strict=False)
            if not isinstance(network, ipaddress.IPv4Network):
                raise ValueError(f"benchmark network must be IPv4, got {configured_cidr}")
            if network.prefixlen != 24:
                raise ValueError(f"benchmark network must be a /24, got {configured_cidr}")
            self._network_cidr = str(network)
            self._bridge_ip = str(next(network.hosts()))
            return
        route_result = subprocess.run(
            ["ip", "-4", "route", "show", "table", "main"],
            check=True,
            capture_output=True,
            text=True,
        )
        self._bridge_ip, self._network_cidr = select_benchmark_network(existing_routes=route_result.stdout)

    def ensure_bridge(self) -> None:
        with self._lock:
            if self._bridge_name is not None:
                return
            bridge_name = f"acb{uuid.uuid4().hex[:8]}"
            subprocess.run(["ip", "link", "add", bridge_name, "type", "bridge"], check=True)
            subprocess.run(["ip", "addr", "add", f"{self._bridge_ip}/24", "dev", bridge_name], check=True)
            subprocess.run(["ip", "link", "set", bridge_name, "up"], check=True)
            self._bridge_name = bridge_name
            logger.info("Created benchmark bridge name=%s bridge_ip=%s", bridge_name, self._bridge_ip)

    def allocate_lease(self, sandbox_id: SandboxId) -> BenchmarkNetworkLease:
        self.ensure_bridge()
        with self._lock:
            assert self._bridge_name is not None
            network = ipaddress.ip_network(self._network_cidr)
            if self._ip_cursor >= network.num_addresses - 1:
                raise RuntimeError("benchmark network exhausted guest IP capacity")
            guest_ip = str(network[self._ip_cursor])
            self._ip_cursor += 1
            suffix = uuid.uuid4().hex[:8]
            namespace_name = f"ts-{suffix}"
            host_veth_name = f"vh{suffix[:6]}"
            guest_veth_name = f"vg{suffix[:6]}"
            subprocess.run(["ip", "netns", "add", namespace_name], check=True)
            subprocess.run(
                ["ip", "link", "add", host_veth_name, "type", "veth", "peer", "name", guest_veth_name],
                check=True,
            )
            subprocess.run(["ip", "link", "set", host_veth_name, "master", self._bridge_name], check=True)
            subprocess.run(["ip", "link", "set", host_veth_name, "up"], check=True)
            subprocess.run(["ip", "link", "set", guest_veth_name, "netns", namespace_name], check=True)
            subprocess.run(["ip", "netns", "exec", namespace_name, "ip", "link", "set", "lo", "up"], check=True)
            subprocess.run(
                ["ip", "netns", "exec", namespace_name, "ip", "link", "set", guest_veth_name, "name", "eth0"],
                check=True,
            )
            subprocess.run(
                ["ip", "netns", "exec", namespace_name, "ip", "addr", "add", f"{guest_ip}/24", "dev", "eth0"],
                check=True,
            )
            subprocess.run(["ip", "netns", "exec", namespace_name, "ip", "link", "set", "eth0", "up"], check=True)
            subprocess.run(
                ["ip", "netns", "exec", namespace_name, "ip", "route", "replace", "default", "via", self._bridge_ip],
                check=True,
            )
            lease = BenchmarkNetworkLease(
                sandbox_id=sandbox_id,
                namespace_name=namespace_name,
                namespace_path=Path("/var/run/netns") / namespace_name,
                host_veth_name=host_veth_name,
                guest_veth_name=guest_veth_name,
                guest_ip=guest_ip,
            )
            self._leases[sandbox_id] = lease
            logger.info(
                "Allocated benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
                sandbox_id,
                guest_ip,
                namespace_name,
            )
            return lease

    def register_guest_ip(self, guest_ip: str, sandbox_id: SandboxId) -> None:
        with self._lock:
            self._ip_to_sandbox[guest_ip] = sandbox_id

    def resolve_sandbox_id(self, client_host: str | None) -> SandboxId | None:
        if client_host is None:
            return None
        with self._lock:
            return self._ip_to_sandbox.get(client_host)

    def lease_for(self, sandbox_id: SandboxId) -> BenchmarkNetworkLease | None:
        with self._lock:
            return self._leases.get(sandbox_id)

    def release_lease(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            lease = self._leases.pop(sandbox_id, None)
            if lease is not None:
                self._ip_to_sandbox.pop(lease.guest_ip, None)
        if lease is None:
            return
        subprocess.run(["ip", "netns", "del", lease.namespace_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ip", "link", "delete", lease.host_veth_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info(
            "Released benchmark network lease sandbox=%s guest_ip=%s namespace=%s",
            sandbox_id,
            lease.guest_ip,
            lease.namespace_name,
        )

    def cleanup(self) -> None:
        with self._lock:
            sandbox_ids = list(self._leases)
            bridge_name = self._bridge_name
        for sandbox_id in sandbox_ids:
            self.release_lease(sandbox_id)
        if bridge_name is not None:
            subprocess.run(["ip", "link", "delete", bridge_name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with self._lock:
                if self._bridge_name == bridge_name:
                    self._bridge_name = None

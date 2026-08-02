from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import socket
from typing import Callable

from integrations.sandboxes.runtime.bundle import SandboxResourceLimits, write_bundle_config
from integrations.sandboxes.runtime.image import ImageRuntimeDefaults


@dataclass(frozen=True)
class PreparedBundleLaunch:
    bundle_dir: Path
    work_dir_host_path: Path | None
    status_host: str
    status_port: int
    llm_base_url: str


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def prepare_bundle_launch(
    *,
    bundle_root: Path,
    sandbox_name: str,
    provider: str,
    pool_name: str,
    interceptor_host: str,
    interceptor_port: int,
    status_host: str,
    status_port: int | None = None,
    work_dir_host_path: Path | None = None,
    network_namespace_path: Path | None = None,
    image_defaults: ImageRuntimeDefaults | None = None,
    image_rootfs_dir: Path | None = None,
    bundle_spec_writer: Callable[[Path], None] | None = None,
    resource_limits: SandboxResourceLimits | None = None,
) -> PreparedBundleLaunch:
    resolved_status_port = find_free_port() if status_port is None else status_port
    bundle_dir = bundle_root / sandbox_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir, ignore_errors=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if bundle_spec_writer is None:
        raise ValueError("bundle_spec_writer is required")
    bundle_spec_writer(bundle_dir)
    llm_base_url = f"http://{interceptor_host}:{interceptor_port}/v1"
    write_bundle_config(
        bundle_dir=bundle_dir,
        llm_base_url=llm_base_url,
        provider=provider,
        sandbox_name=sandbox_name,
        status_port=resolved_status_port,
        cgroup_path=f"crab-bench/{pool_name}/{sandbox_name}",
        work_dir_host_path=work_dir_host_path,
        network_namespace_path=network_namespace_path,
        image_defaults=image_defaults,
        image_rootfs_dir=image_rootfs_dir,
        resource_limits=resource_limits,
    )
    return PreparedBundleLaunch(
        bundle_dir=bundle_dir,
        work_dir_host_path=work_dir_host_path,
        status_host=status_host,
        status_port=resolved_status_port,
        llm_base_url=llm_base_url,
    )

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from integrations.sandboxes.runtime.image import ImageRuntimeDefaults


def parse_env_assignments(values: Iterable[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for entry in values:
        key, sep, value = str(entry).partition("=")
        if not sep:
            raise ValueError(f"invalid environment assignment: {entry!r}")
        parsed[key] = value
    return parsed


def merge_environment_defaults(defaults: Iterable[str], overrides: Iterable[str]) -> list[str]:
    merged = parse_env_assignments(defaults)
    merged.update(parse_env_assignments(overrides))
    return [f"{key}={value}" for key, value in merged.items()]


def parse_passwd_file(path: Path) -> tuple[dict[str, tuple[int, int]], dict[int, tuple[int, str]]]:
    by_name: dict[str, tuple[int, int]] = {}
    by_uid: dict[int, tuple[int, str]] = {}
    if not path.is_file():
        return by_name, by_uid
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        parts = raw_line.split(":")
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            uid = int(parts[2])
            gid = int(parts[3])
        except ValueError:
            continue
        by_name[name] = (uid, gid)
        by_uid[uid] = (gid, name)
    return by_name, by_uid


def parse_group_file(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    by_name: dict[str, int] = {}
    by_gid: dict[int, str] = {}
    if not path.is_file():
        return by_name, by_gid
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        parts = raw_line.split(":")
        if len(parts) < 3:
            continue
        name = parts[0]
        try:
            gid = int(parts[2])
        except ValueError:
            continue
        by_name[name] = gid
        by_gid[gid] = name
    return by_name, by_gid


def resolve_process_user_from_rootfs(*, rootfs_dir: Path, user_spec: str) -> dict[str, object]:
    value = user_spec.strip()
    if not value:
        raise ValueError("compose/image user must not be empty")
    passwd_by_name, passwd_by_uid = parse_passwd_file(rootfs_dir / "etc" / "passwd")
    group_by_name, _ = parse_group_file(rootfs_dir / "etc" / "group")

    user_part, sep, group_part = value.partition(":")
    if not user_part:
        raise ValueError(f"unsupported user specification: {user_spec!r}")

    uid: int
    gid: int | None = None
    if user_part.isdigit():
        uid = int(user_part)
        if uid in passwd_by_uid:
            gid = passwd_by_uid[uid][0]
    else:
        if user_part not in passwd_by_name:
            raise ValueError(f"user {user_part!r} not found in rootfs {rootfs_dir}")
        uid, gid = passwd_by_name[user_part]

    if sep:
        if not group_part:
            raise ValueError(f"unsupported user specification: {user_spec!r}")
        if group_part.isdigit():
            gid = int(group_part)
        else:
            if group_part not in group_by_name:
                raise ValueError(f"group {group_part!r} not found in rootfs {rootfs_dir}")
            gid = group_by_name[group_part]

    if gid is None:
        gid = 0
    return {"uid": uid, "gid": gid}


def write_bundle_config(
    *,
    bundle_dir: Path,
    llm_base_url: str,
    provider: str,
    sandbox_name: str,
    status_port: int,
    cgroup_path: str,
    work_dir_host_path: Path | None = None,
    network_namespace_path: Path | None = None,
    image_defaults: ImageRuntimeDefaults | None = None,
    image_rootfs_dir: Path | None = None,
) -> None:
    config_path = bundle_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    linux_cfg = cfg.get("linux", {})
    namespaces = []
    network_namespace_found = False
    for namespace in linux_cfg.get("namespaces", []):
        ns_type = namespace.get("type")
        if ns_type == "cgroup":
            continue
        if ns_type == "network":
            network_namespace_found = True
            if network_namespace_path is None:
                continue
            namespace = {**namespace, "path": str(network_namespace_path)}
        namespaces.append(namespace)
    if network_namespace_path is not None and not network_namespace_found:
        namespaces.append({"type": "network", "path": str(network_namespace_path)})
    linux_cfg["namespaces"] = namespaces
    linux_cfg["cgroupsPath"] = cgroup_path
    linux_cfg.pop("seccomp", None)
    cfg["linux"] = linux_cfg
    mounts = [mount for mount in cfg.get("mounts", []) if mount.get("destination") != "/work"]
    if work_dir_host_path is not None:
        work_dir_host_path.mkdir(parents=True, exist_ok=True)
        mounts.append(
            {
                "destination": "/work",
                "source": str(work_dir_host_path),
                "type": "bind",
                "options": ["rbind", "rw"],
            }
        )
    cfg["mounts"] = mounts
    cfg["process"]["terminal"] = False
    cfg["process"]["cwd"] = image_defaults.working_dir if image_defaults and image_defaults.working_dir else "/work"
    cfg["process"]["args"] = [
        "/bin/sh",
        "-lc",
        f"exec /usr/local/bin/agent-cli run --provider {provider} >/dev/null 2>/dev/null",
    ]
    runtime_env = [
        "PATH=/usr/local/bin:/usr/bin:/bin",
        "PYTHONUNBUFFERED=1",
        f"AGENT_CR_LLM_BASE_URL={llm_base_url}",
        f"STATUS_PORT={status_port}",
        "POLL_INTERVAL_S=0.2",
        "AGENT_WORK_DIR=/work",
        f"AGENT_SANDBOX_ID={sandbox_name}",
        f"AGENT_PROVIDER={provider}",
    ]
    cfg["process"]["env"] = merge_environment_defaults(
        image_defaults.environment if image_defaults is not None else (),
        runtime_env,
    )
    if image_defaults is not None and image_defaults.user is not None:
        if image_rootfs_dir is None:
            raise ValueError("image_rootfs_dir is required when image defaults specify a user")
        cfg["process"]["user"] = resolve_process_user_from_rootfs(
            rootfs_dir=image_rootfs_dir,
            user_spec=image_defaults.user,
        )
    cfg["root"]["path"] = "rootfs"
    cfg["root"]["readonly"] = False
    config_path.write_text(json.dumps(cfg, indent=2))

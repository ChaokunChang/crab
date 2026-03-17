from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from integrations.sandboxes.runtime.bundle import merge_environment_defaults, resolve_process_user_from_rootfs
from integrations.sandboxes.runtime.image import ImageRuntimeDefaults, docker_tag_component, export_image_rootfs, inspect_image_runtime_defaults

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")
_UNSUPPORTED_COMPOSE_FEATURES = {"depends_on", "profiles", "networks", "configs", "secrets", "healthcheck"}


@dataclass(frozen=True)
class ComposeSandboxTranslation:
    runtime_launch_metadata: dict[str, object]
    compose_launch_metadata: dict[str, object]


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"invalid env file line in {path}: {raw_line!r}")
        env[key.strip()] = value.strip()
    return env


def interpolate_compose_value(value: Any, env: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(lambda match: env.get(match.group(1), match.group(2) or ""), value)
    if isinstance(value, list):
        return [interpolate_compose_value(item, env) for item in value]
    if isinstance(value, dict):
        return {str(key): interpolate_compose_value(item, env) for key, item in value.items()}
    return value


def load_compose_service(
    *,
    compose_file: Path,
    env_file: Path | None = None,
    extra_env: dict[str, str] | None = None,
    service_name: str | None = None,
) -> tuple[str, dict[str, object]]:
    compose_env = dict(os.environ)
    if env_file is not None:
        compose_env.update(parse_env_file(env_file))
    if extra_env is not None:
        compose_env.update({str(key): str(value) for key, value in extra_env.items()})
    payload = yaml.safe_load(compose_file.read_text(encoding="utf-8")) or {}
    payload = interpolate_compose_value(payload, compose_env)
    services = payload.get("services")
    if not isinstance(services, dict) or not services:
        raise ValueError(f"compose file {compose_file} does not define any services")
    if service_name is None:
        if len(services) != 1:
            raise ValueError(f"compose file {compose_file} contains multiple services; specify service_name")
        service_name = next(iter(services))
    if service_name not in services:
        raise ValueError(f"compose service {service_name!r} not found in {compose_file}")
    service = services[service_name]
    if not isinstance(service, dict):
        raise ValueError(f"compose service {service_name!r} in {compose_file} must be an object")
    found_unsupported = sorted(key for key in _UNSUPPORTED_COMPOSE_FEATURES if key in service)
    if found_unsupported:
        raise ValueError(f"unsupported compose features for benchmark translation: {found_unsupported}")
    return service_name, service


def compose_build_tag(
    *,
    compose_file: Path,
    service_name: str,
    build_spec: str | dict[str, object],
) -> str:
    fingerprint = hashlib.sha256()
    fingerprint.update(str(compose_file.resolve()).encode("utf-8"))
    fingerprint.update(service_name.encode("utf-8"))
    fingerprint.update(json.dumps(build_spec, sort_keys=True).encode("utf-8"))
    service_component = docker_tag_component(service_name)
    return f"agent-cr-compose-{service_component}:{fingerprint.hexdigest()[:12]}"


def resolve_compose_image_ref(
    *,
    compose_file: Path,
    service_name: str,
    service: dict[str, object],
    compose_image_tags: set[str] | None = None,
) -> str:
    image_ref = service.get("image")
    build_spec = service.get("build")
    if isinstance(image_ref, str) and image_ref and build_spec is None:
        return image_ref
    if build_spec is None:
        raise ValueError(f"compose service {service_name} requires image or build")
    tag = (
        str(image_ref)
        if isinstance(image_ref, str) and image_ref
        else compose_build_tag(
            compose_file=compose_file,
            service_name=service_name,
            build_spec=build_spec,
        )
    )
    if compose_image_tags is not None:
        compose_image_tags.add(tag)
    build_context = compose_file.parent
    dockerfile = None
    build_args: list[str] = []
    if isinstance(build_spec, str):
        build_context = (compose_file.parent / build_spec).resolve()
    elif isinstance(build_spec, dict):
        context_value = build_spec.get("context", ".")
        build_context = (compose_file.parent / str(context_value)).resolve()
        dockerfile_value = build_spec.get("dockerfile")
        if dockerfile_value is not None:
            dockerfile = (build_context / str(dockerfile_value)).resolve()
        args_value = build_spec.get("args", {})
        if isinstance(args_value, dict):
            for key, value in sorted(args_value.items()):
                build_args.extend(["--build-arg", f"{key}={value}"])
    else:
        raise ValueError(f"unsupported compose build definition for service {service_name}: {build_spec!r}")
    command = ["docker", "build", "-t", tag]
    if dockerfile is not None:
        command.extend(["-f", str(dockerfile)])
    command.extend(build_args)
    command.append(str(build_context))
    subprocess.run(command, check=True)
    return tag


def compose_environment(service: dict[str, object], *, image_defaults: ImageRuntimeDefaults) -> list[str]:
    environment = service.get("environment", {})
    if isinstance(environment, dict):
        items = environment.items()
    elif isinstance(environment, list):
        items = []
        for item in environment:
            key, sep, value = str(item).partition("=")
            items.append((key, value if sep else os.environ.get(key, "")))
    else:
        raise ValueError(f"unsupported compose environment: {environment!r}")
    return merge_environment_defaults(
        image_defaults.environment,
        [f"{key}={value}" for key, value in items],
    )


def compose_working_dir(service: dict[str, object], *, image_defaults: ImageRuntimeDefaults) -> str:
    working_dir = service.get("working_dir")
    if working_dir is None:
        return image_defaults.working_dir or "/work"
    return str(working_dir)


def compose_process_args(service: dict[str, object], *, image_defaults: ImageRuntimeDefaults) -> list[str]:
    entrypoint = service.get("entrypoint")
    command = service.get("command")
    segments: list[str] = []
    for value in (entrypoint, command):
        if value is None:
            continue
        if isinstance(value, list):
            segments.extend(str(item) for item in value)
        elif isinstance(value, str):
            segments.extend(["/bin/sh", "-lc", value])
        else:
            raise ValueError(f"unsupported compose command/entrypoint value: {value!r}")
    if segments:
        return segments
    defaults = list(image_defaults.entrypoint) + list(image_defaults.command)
    if defaults:
        return defaults
    return ["/bin/sh"]


def compose_process_user(
    service: dict[str, object],
    *,
    rootfs_dir: Path,
    image_defaults: ImageRuntimeDefaults,
) -> dict[str, object] | None:
    user_value = service.get("user", image_defaults.user)
    if user_value is None:
        return None
    if isinstance(user_value, int):
        user_value = str(user_value)
    if not isinstance(user_value, str):
        raise ValueError(f"unsupported compose user value: {user_value!r}")
    return resolve_process_user_from_rootfs(rootfs_dir=rootfs_dir, user_spec=user_value)


def resolve_compose_bind_source(source: str, *, compose_file: Path) -> Path:
    path = Path(source).expanduser()
    if not path.is_absolute():
        path = compose_file.parent / path
    return path.resolve()


def compose_mounts(service: dict[str, object], *, compose_file: Path) -> list[dict[str, object]]:
    mounts: list[dict[str, object]] = []
    for item in service.get("volumes", []):
        if isinstance(item, str):
            parts = item.split(":")
            if len(parts) < 2:
                raise ValueError(f"unsupported compose volume syntax: {item!r}")
            source = resolve_compose_bind_source(parts[0], compose_file=compose_file)
            destination = parts[1]
            options = ["rbind", "rw"]
            if len(parts) > 2 and parts[2] == "ro":
                options = ["rbind", "ro"]
            mounts.append(
                {
                    "destination": destination,
                    "source": str(source),
                    "type": "bind",
                    "options": options,
                }
            )
            continue
        if isinstance(item, dict) and item.get("type", "bind") == "bind":
            source = resolve_compose_bind_source(str(item["source"]), compose_file=compose_file)
            read_only = bool(item.get("read_only", False))
            mounts.append(
                {
                    "destination": str(item["target"]),
                    "source": str(source),
                    "type": "bind",
                    "options": ["rbind", "ro" if read_only else "rw"],
                }
            )
            continue
        raise ValueError(f"unsupported compose volume definition: {item!r}")
    return mounts


def translate_compose_service(
    *,
    compose_file: Path,
    service_name: str,
    service: dict[str, object],
    bundle_dir: Path,
    sandbox_id: str,
    work_dir_host_path: Path | None,
    compose_image_root: Path,
    compose_image_tags: set[str] | None = None,
) -> ComposeSandboxTranslation:
    bundle_config = bundle_dir / "config.json"
    config = json.loads(bundle_config.read_text(encoding="utf-8"))
    image_ref = resolve_compose_image_ref(
        compose_file=compose_file,
        service_name=service_name,
        service=service,
        compose_image_tags=compose_image_tags,
    )
    image_defaults = inspect_image_runtime_defaults(tag=image_ref)
    rootfs_dir = export_image_rootfs(
        tag=image_ref,
        output_dir=compose_image_root / sandbox_id,
    )
    config["process"]["cwd"] = compose_working_dir(service, image_defaults=image_defaults)
    config["process"]["terminal"] = bool(service.get("tty", False))
    config["process"]["env"] = compose_environment(service, image_defaults=image_defaults)
    config["process"]["args"] = compose_process_args(service, image_defaults=image_defaults)
    process_user = compose_process_user(service, rootfs_dir=rootfs_dir, image_defaults=image_defaults)
    if process_user is not None:
        config["process"]["user"] = process_user
    mounts = [mount for mount in config.get("mounts", []) if mount.get("destination") != "/work"]
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
    mounts.extend(compose_mounts(service, compose_file=compose_file))
    config["mounts"] = mounts
    bundle_config.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return ComposeSandboxTranslation(
        compose_launch_metadata={
            "service_name": service_name,
            "image_ref": image_ref,
            "ports": list(service.get("ports", [])),
        },
        runtime_launch_metadata={
            "sandbox_id": sandbox_id,
            "bundle_path": str(bundle_dir),
            "work_dir_host_path": None if work_dir_host_path is None else str(work_dir_host_path),
            "rootfs_init_dirs": [
                "work",
                "tmp",
                "proc",
                "dev",
                "dev/pts",
                "dev/shm",
                "dev/mqueue",
                "sys",
                "run",
                "var",
            ],
            "rootfs_copy_paths": [{"source": str(rootfs_dir), "destination": "/"}],
            "compose_service_name": service_name,
            "compose_ports": list(service.get("ports", [])),
        },
    )

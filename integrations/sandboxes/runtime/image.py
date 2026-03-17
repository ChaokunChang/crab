from __future__ import annotations

import fcntl
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRuntimeDefaults:
    environment: tuple[str, ...] = ()
    working_dir: str | None = None
    user: str | None = None
    entrypoint: tuple[str, ...] = ()
    command: tuple[str, ...] = ()


def docker_tag_component(raw: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", raw.lower()).strip("-.")
    return normalized or "image"


def inspect_image_runtime_defaults(*, tag: str) -> ImageRuntimeDefaults:
    raw_output = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{json .Config}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config = json.loads(raw_output) if raw_output else {}
    if not isinstance(config, dict):
        raise ValueError(f"unexpected docker image config for {tag}: {config!r}")

    def _string_list(key: str) -> tuple[str, ...]:
        value = config.get(key)
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ValueError(f"unsupported docker image config {key} for {tag}: {value!r}")
        return tuple(str(item) for item in value)

    working_dir = config.get("WorkingDir")
    if working_dir is not None and not isinstance(working_dir, str):
        raise ValueError(f"unsupported docker image config WorkingDir for {tag}: {working_dir!r}")
    user = config.get("User")
    if user is not None and not isinstance(user, str):
        raise ValueError(f"unsupported docker image config User for {tag}: {user!r}")

    return ImageRuntimeDefaults(
        environment=_string_list("Env"),
        working_dir=working_dir or None,
        user=user or None,
        entrypoint=_string_list("Entrypoint"),
        command=_string_list("Cmd"),
    )


def build_image(*, tag: str, build_context: Path, dockerfile_path: Path) -> None:
    subprocess.run(
        ["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(build_context)],
        check=True,
    )


def export_image_rootfs(*, tag: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    container_id = (
        subprocess.run(
            ["docker", "create", tag],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
    )
    rootfs_dir = output_dir / "rootfs"
    lock_path = output_dir / ".export.lock"
    staging_dir = Path(tempfile.mkdtemp(prefix="rootfs-export-", dir=output_dir))
    tar_path = staging_dir / "rootfs.tar"
    staging_rootfs_dir = staging_dir / "rootfs"
    try:
        with tar_path.open("wb") as fh:
            subprocess.run(["docker", "export", container_id], check=True, stdout=fh)
        staging_rootfs_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(staging_rootfs_dir)
        with lock_path.open("w", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            backup_dir = output_dir / "rootfs.previous"
            try:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                if rootfs_dir.exists():
                    rootfs_dir.replace(backup_dir)
                staging_rootfs_dir.replace(rootfs_dir)
            except Exception:
                if backup_dir.exists() and not rootfs_dir.exists():
                    backup_dir.replace(rootfs_dir)
                raise
            finally:
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    finally:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(staging_dir, ignore_errors=True)
    return rootfs_dir

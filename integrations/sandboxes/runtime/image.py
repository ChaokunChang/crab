from __future__ import annotations

import copy
import fcntl
import json
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from crab.contracts import TelemetrySink


def _telemetry_sink(telemetry: TelemetrySink | None) -> TelemetrySink:
    if telemetry is not None:
        return telemetry
    # Keep this module importable while ``crab`` is still initializing. The
    # built-in iFlow agent imports the image helpers from crab.__init__.
    from crab.telemetry import NoopTelemetrySink

    return NoopTelemetrySink()


@dataclass(frozen=True)
class ImageRuntimeDefaults:
    environment: tuple[str, ...] = ()
    working_dir: str | None = None
    user: str | None = None
    entrypoint: tuple[str, ...] = ()
    command: tuple[str, ...] = ()


def container_rootfs_tar_filter(member: tarfile.TarInfo, destination: str | Path) -> tarfile.TarInfo | None:
    """Reject path escapes while preserving container rootfs metadata.

    Docker root filesystems legitimately contain absolute link targets because
    those links are interpreted inside the container root. Python 3.14's
    default ``data`` filter rejects them and also strips ownership, sticky
    bits, and group/world write bits. Those metadata are part of a container
    image's runtime semantics (for example, ``/tmp`` must remain ``01777``).
    Rewrite absolute link targets inside the rootfs, then use Python's ``tar``
    filter to reject absolute/member traversal without altering OCI metadata.
    """

    if member.issym() and posixpath.isabs(member.linkname):
        member = copy.copy(member)
        link_target = member.linkname.lstrip("/")
        link_parent = posixpath.dirname(member.name.lstrip("/")) or "."
        member.linkname = posixpath.relpath(link_target, start=link_parent)
    elif member.islnk() and posixpath.isabs(member.linkname):
        member = copy.copy(member)
        member.linkname = member.linkname.lstrip("/")
    filtered = tarfile.tar_filter(member, destination)
    if filtered is None:
        return None
    filtered = copy.copy(filtered)
    filtered.mode = member.mode
    filtered.uid = member.uid
    filtered.gid = member.gid
    filtered.uname = member.uname
    filtered.gname = member.gname
    return filtered


def docker_tag_component(raw: str) -> str:
    normalized = re.sub(r"[^a-z0-9_.-]+", "-", raw.lower()).strip("-.")
    return normalized or "image"


def image_exists(*, tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def inspect_image_id(*, tag: str, telemetry: TelemetrySink | None = None) -> str:
    sink = _telemetry_sink(telemetry)
    started = time.perf_counter()
    raw_output = subprocess.run(
        ["docker", "image", "inspect", tag, "--format", "{{.Id}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    duration_ms = (time.perf_counter() - started) * 1000.0
    image_id = raw_output.replace("sha256:", "")
    sink.emit_metric(
        "image.inspect_ms",
        duration_ms,
        {"tag": tag, "image_id": image_id},
    )
    sink.emit_event("image.inspect", {"tag": tag, "image_id": image_id})
    return image_id


def inspect_image_runtime_defaults(
    *,
    tag: str,
    cache_root: Path | None = None,
    telemetry: TelemetrySink | None = None,
) -> ImageRuntimeDefaults:
    sink = _telemetry_sink(telemetry)
    image_id = inspect_image_id(tag=tag, telemetry=sink)
    cache_dir = None if cache_root is None else cache_root / image_id
    cache_path = None if cache_dir is None else cache_dir / "runtime_defaults.json"
    if cache_path is not None and cache_path.is_file():
        sink.emit_event("image.defaults_cache_hit", {"tag": tag, "image_id": image_id, "path": str(cache_path)})
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return ImageRuntimeDefaults(
            environment=tuple(payload.get("environment", [])),
            working_dir=payload.get("working_dir"),
            user=payload.get("user"),
            entrypoint=tuple(payload.get("entrypoint", [])),
            command=tuple(payload.get("command", [])),
        )

    started = time.perf_counter()
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

    defaults = ImageRuntimeDefaults(
        environment=_string_list("Env"),
        working_dir=working_dir or None,
        user=user or None,
        entrypoint=_string_list("Entrypoint"),
        command=_string_list("Cmd"),
    )
    if cache_path is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(asdict(defaults), sort_keys=True, indent=2), encoding="utf-8")
    duration_ms = (time.perf_counter() - started) * 1000.0
    sink.emit_metric(
        "image.inspect_defaults_ms",
        duration_ms,
        {"tag": tag, "image_id": image_id, "cache_hit": False},
    )
    sink.emit_event("image.defaults_cache_miss", {"tag": tag, "image_id": image_id})
    return defaults


def build_image(
    *,
    tag: str,
    build_context: Path,
    dockerfile_path: Path,
    telemetry: TelemetrySink | None = None,
    skip_if_exists: bool = True,
) -> None:
    sink = _telemetry_sink(telemetry)
    if skip_if_exists and image_exists(tag=tag):
        sink.emit_event("image.build_cache_hit", {"tag": tag, "build_context": str(build_context)})
        sink.emit_metric("image.build_ms", 0.0, {"tag": tag, "cache_hit": True})
        return
    started = time.perf_counter()
    subprocess.run(
        ["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(build_context)],
        check=True,
    )
    duration_ms = (time.perf_counter() - started) * 1000.0
    sink.emit_metric("image.build_ms", duration_ms, {"tag": tag, "cache_hit": False})
    sink.emit_event("image.build", {"tag": tag, "build_context": str(build_context), "dockerfile_path": str(dockerfile_path)})


def export_image_rootfs(
    *,
    tag: str,
    output_dir: Path,
    cache_root: Path | None = None,
    telemetry: TelemetrySink | None = None,
) -> Path:
    sink = _telemetry_sink(telemetry)
    image_id = inspect_image_id(tag=tag, telemetry=sink)
    resolved_output_dir = output_dir if cache_root is None else cache_root / image_id
    rootfs_dir = resolved_output_dir / "rootfs"
    lock_path = resolved_output_dir / ".export.lock"
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    lock_wait_started = time.perf_counter()
    with lock_path.open("w", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        lock_wait_ms = (time.perf_counter() - lock_wait_started) * 1000.0
        sink.emit_metric("image.cache_lock_wait_ms", lock_wait_ms, {"tag": tag, "image_id": image_id})
        if rootfs_dir.is_dir() and any(rootfs_dir.iterdir()):
            sink.emit_event("image.export_cache_hit", {"tag": tag, "image_id": image_id, "rootfs_dir": str(rootfs_dir)})
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            return rootfs_dir

        sink.emit_event("image.export_cache_miss", {"tag": tag, "image_id": image_id, "rootfs_dir": str(rootfs_dir)})
        container_id = (
            subprocess.run(
                ["docker", "create", tag],
                check=True,
                capture_output=True,
                text=True,
            )
            .stdout.strip()
        )
        staging_dir = Path(tempfile.mkdtemp(prefix="rootfs-export-", dir=resolved_output_dir))
        tar_path = staging_dir / "rootfs.tar"
        staging_rootfs_dir = staging_dir / "rootfs"
        started = time.perf_counter()
        try:
            with tar_path.open("wb") as fh:
                subprocess.run(["docker", "export", container_id], check=True, stdout=fh)
            staging_rootfs_dir.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tar_path) as tf:
                tf.extractall(staging_rootfs_dir, filter=container_rootfs_tar_filter)
            backup_dir = resolved_output_dir / "rootfs.previous"
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
        finally:
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            shutil.rmtree(staging_dir, ignore_errors=True)
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        duration_ms = (time.perf_counter() - started) * 1000.0
        sink.emit_metric("image.export_ms", duration_ms, {"tag": tag, "image_id": image_id, "cache_hit": False})
    return rootfs_dir

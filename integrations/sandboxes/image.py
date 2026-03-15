from __future__ import annotations

import fcntl
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


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

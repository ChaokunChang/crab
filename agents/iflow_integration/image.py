from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DOCKERFILE_PATH = PACKAGE_ROOT / "Dockerfile"


def build_image(*, tag: str) -> None:
    subprocess.run(
        ["docker", "build", "-t", tag, "-f", str(DOCKERFILE_PATH), str(PACKAGE_ROOT)],
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
    tar_path = output_dir / "rootfs.tar"
    rootfs_dir = output_dir / "rootfs"
    try:
        with tar_path.open("wb") as fh:
            subprocess.run(["docker", "export", container_id], check=True, stdout=fh)
        if rootfs_dir.exists():
            shutil.rmtree(rootfs_dir)
        rootfs_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path) as tf:
            tf.extractall(rootfs_dir)
    finally:
        subprocess.run(["docker", "rm", "-f", container_id], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return rootfs_dir

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DOCKERFILE_PATH = PACKAGE_ROOT / "Dockerfile"

__all__ = ["DOCKERFILE_PATH", "PACKAGE_ROOT"]

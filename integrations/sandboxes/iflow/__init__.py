from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
DOCKERFILE_PATH = PACKAGE_ROOT / "Dockerfile"

from .harness import (
    PreparedIFlowRuntime,
    PreparedIFlowState,
    cache_dir_from_env,
    ensure_cache_files,
    prepare_iflow_runtime,
    prepare_iflow_state,
    required_cache_paths,
)

__all__ = [
    "DOCKERFILE_PATH",
    "PACKAGE_ROOT",
    "PreparedIFlowRuntime",
    "PreparedIFlowState",
    "cache_dir_from_env",
    "ensure_cache_files",
    "prepare_iflow_runtime",
    "prepare_iflow_state",
    "required_cache_paths",
]

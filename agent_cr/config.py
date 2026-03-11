from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SchedulerConfig:
    max_pending_jobs: int = 10000

    def __post_init__(self) -> None:
        if self.max_pending_jobs < 1:
            raise ValueError("max_pending_jobs must be >= 1")


@dataclass(frozen=True)
class ExecutorConfig:
    max_workers: int = 4
    max_checkpoint_queue_size: int = 10000
    max_retries: int = 0
    retry_backoff_seconds: float = 0.05

    def __post_init__(self) -> None:
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.max_checkpoint_queue_size < 1:
            raise ValueError("max_checkpoint_queue_size must be >= 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be >= 0")


@dataclass(frozen=True)
class StorageConfig:
    root_dir: Path
    manifests_dirname: str = "manifests"
    artifacts_dirname: str = "artifacts"

    def __post_init__(self) -> None:
        if not self.manifests_dirname:
            raise ValueError("manifests_dirname must be non-empty")
        if not self.artifacts_dirname:
            raise ValueError("artifacts_dirname must be non-empty")


@dataclass(frozen=True)
class PolicyConfig:
    min_checkpoint_interval_seconds: float = 30.0
    force_checkpoint_after_seconds: float = 600.0
    require_change_signal: bool = True
    prefer_checkpoint_during_llm_request: bool = True
    require_llm_request_for_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.min_checkpoint_interval_seconds < 0:
            raise ValueError("min_checkpoint_interval_seconds must be >= 0")
        if self.force_checkpoint_after_seconds < 0:
            raise ValueError("force_checkpoint_after_seconds must be >= 0")


@dataclass(frozen=True)
class TelemetryConfig:
    enabled: bool = True

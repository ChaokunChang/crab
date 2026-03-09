from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4


@dataclass(frozen=True, order=True)
class SandboxId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("sandbox id must be non-empty")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def new(cls, prefix: str = "sbx") -> "SandboxId":
        return cls(f"{prefix}-{uuid4().hex[:12]}")


@dataclass(frozen=True, order=True)
class CheckpointId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("checkpoint id must be non-empty")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def new(cls, prefix: str = "ckpt") -> "CheckpointId":
        return cls(f"{prefix}-{uuid4().hex[:12]}")


@dataclass(frozen=True, order=True)
class JobId:
    value: str

    def __post_init__(self) -> None:
        if not self.value:
            raise ValueError("job id must be non-empty")

    def __str__(self) -> str:
        return self.value

    @classmethod
    def new(cls, prefix: str = "job") -> "JobId":
        return cls(f"{prefix}-{uuid4().hex[:12]}")

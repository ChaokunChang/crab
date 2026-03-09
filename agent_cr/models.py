from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .ids import CheckpointId, JobId, SandboxId


MANIFEST_SCHEMA_VERSION = "v1"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc).isoformat()


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw)


class JobType(str, Enum):
    CHECKPOINT = "checkpoint"
    RESTORE = "restore"


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ArtifactKind(str, Enum):
    PROCESS = "process"
    FILESYSTEM = "filesystem"
    METADATA = "metadata"


class FailureCode(str, Enum):
    NONE = "none"
    NOT_IMPLEMENTED = "not_implemented"
    RUNTIME_ERROR = "runtime_error"
    STORAGE_ERROR = "storage_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RuntimeCapabilities:
    supports_process_checkpoint: bool
    supports_filesystem_checkpoint: bool
    supports_incremental_filesystem: bool = False
    supports_custom_checkpoint_dir: bool = False


@dataclass(frozen=True)
class DryRunStatus:
    executed: bool
    reason: str
    planned_command: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactPayload:
    kind: ArtifactKind
    name: str
    data: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactReference:
    kind: ArtifactKind
    name: str
    relative_path: str
    size_bytes: int
    sha256: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ArtifactReference":
        return cls(
            kind=ArtifactKind(raw["kind"]),
            name=str(raw["name"]),
            relative_path=str(raw["relative_path"]),
            size_bytes=int(raw["size_bytes"]),
            sha256=str(raw["sha256"]),
            metadata=dict(raw.get("metadata", {})),
        )


@dataclass(frozen=True)
class WorkerStepResult:
    success: bool
    artifacts: list[ArtifactPayload] = field(default_factory=list)
    dry_run_status: DryRunStatus = field(
        default_factory=lambda: DryRunStatus(executed=False, reason="unknown")
    )
    failure_code: FailureCode = FailureCode.NONE
    message: str | None = None


@dataclass(frozen=True)
class CheckpointManifest:
    schema_version: str
    checkpoint_id: CheckpointId
    sandbox_id: SandboxId
    created_at: datetime
    runtime_name: str
    runtime_version: str | None
    process_artifacts: list[ArtifactReference]
    filesystem_artifacts: list[ArtifactReference]
    metadata: dict[str, Any] = field(default_factory=dict)
    integrity: dict[str, str] = field(default_factory=dict)

    def validate_schema(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema: {self.schema_version} "
                f"(expected {MANIFEST_SCHEMA_VERSION})"
            )

    def with_integrity(self) -> "CheckpointManifest":
        payload_hash = self.compute_manifest_hash()
        return replace(self, integrity={"manifest_sha256": payload_hash})

    def validate_integrity(self) -> None:
        expected = self.integrity.get("manifest_sha256")
        if not expected:
            raise ValueError("manifest integrity missing manifest_sha256")
        actual = self.compute_manifest_hash()
        if expected != actual:
            raise ValueError("manifest integrity hash mismatch")

    def compute_manifest_hash(self) -> str:
        canonical = json.dumps(
            {
                "schema_version": self.schema_version,
                "checkpoint_id": str(self.checkpoint_id),
                "sandbox_id": str(self.sandbox_id),
                "created_at": _isoformat(self.created_at),
                "runtime_name": self.runtime_name,
                "runtime_version": self.runtime_version,
                "process_artifacts": [a.to_dict() for a in self.process_artifacts],
                "filesystem_artifacts": [a.to_dict() for a in self.filesystem_artifacts],
                "metadata": self.metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": str(self.checkpoint_id),
            "sandbox_id": str(self.sandbox_id),
            "created_at": _isoformat(self.created_at),
            "runtime_name": self.runtime_name,
            "runtime_version": self.runtime_version,
            "process_artifacts": [a.to_dict() for a in self.process_artifacts],
            "filesystem_artifacts": [a.to_dict() for a in self.filesystem_artifacts],
            "metadata": self.metadata,
            "integrity": self.integrity,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CheckpointManifest":
        manifest = cls(
            schema_version=str(raw["schema_version"]),
            checkpoint_id=CheckpointId(str(raw["checkpoint_id"])),
            sandbox_id=SandboxId(str(raw["sandbox_id"])),
            created_at=_parse_ts(str(raw["created_at"])),
            runtime_name=str(raw["runtime_name"]),
            runtime_version=(None if raw.get("runtime_version") is None else str(raw.get("runtime_version"))),
            process_artifacts=[
                ArtifactReference.from_dict(x) for x in raw.get("process_artifacts", [])
            ],
            filesystem_artifacts=[
                ArtifactReference.from_dict(x) for x in raw.get("filesystem_artifacts", [])
            ],
            metadata=dict(raw.get("metadata", {})),
            integrity=dict(raw.get("integrity", {})),
        )
        manifest.validate_schema()
        manifest.validate_integrity()
        return manifest


@dataclass(frozen=True)
class CheckpointJob:
    job_id: JobId
    sandbox_id: SandboxId
    requested_at: datetime
    reason: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RestoreJob:
    job_id: JobId
    sandbox_id: SandboxId
    checkpoint_id: CheckpointId
    requested_at: datetime
    reason: str = "manual"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointResult:
    job_id: JobId
    sandbox_id: SandboxId
    checkpoint_id: CheckpointId
    status: JobStatus
    started_at: datetime
    finished_at: datetime
    manifest: CheckpointManifest | None
    failure_code: FailureCode = FailureCode.NONE
    message: str | None = None
    dry_run_statuses: tuple[DryRunStatus, ...] = ()


@dataclass(frozen=True)
class RestoreResult:
    job_id: JobId
    sandbox_id: SandboxId
    checkpoint_id: CheckpointId
    status: JobStatus
    started_at: datetime
    finished_at: datetime
    failure_code: FailureCode = FailureCode.NONE
    message: str | None = None
    dry_run_statuses: tuple[DryRunStatus, ...] = ()


@dataclass(frozen=True)
class JobRecord:
    job_id: JobId
    job_type: JobType
    sandbox_id: SandboxId
    checkpoint_id: CheckpointId | None
    status: JobStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    failure_code: FailureCode = FailureCode.NONE
    message: str | None = None


@dataclass(frozen=True)
class SandboxSnapshot:
    sandbox_id: SandboxId
    runtime_name: str
    is_running: bool
    process_changed: bool
    filesystem_changed: bool
    observed_at: datetime
    last_checkpoint_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScheduleDecision:
    should_checkpoint: bool
    reason: str
    policy_name: str
    next_earliest_checkpoint_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    sandbox_id: SandboxId
    started_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SandboxDescription:
    sandbox_id: SandboxId
    runtime_name: str
    status: str
    metadata: dict[str, Any] = field(default_factory=dict)

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from ..config import StorageConfig
from ..contracts import CheckpointManager
from ..ids import CheckpointId, SandboxId
from ..models import ArtifactPayload, ArtifactReference, CheckpointManifest

logger = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    if not value:
        raise ValueError("artifact name must be non-empty")
    return value.replace("/", "_")


class LocalCheckpointManager(CheckpointManager):
    def __init__(self, config: StorageConfig):
        self._config = config
        self._root = Path(config.root_dir)
        self._manifests_root = self._root / config.manifests_dirname
        self._artifacts_root = self._root / config.artifacts_dirname
        self._manifests_root.mkdir(parents=True, exist_ok=True)
        self._artifacts_root.mkdir(parents=True, exist_ok=True)

    def _manifest_path(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> Path:
        return self._manifests_root / str(sandbox_id) / f"{checkpoint_id}.json"

    def _artifact_path(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        kind: str,
        name: str,
    ) -> Path:
        return (
            self._artifacts_root
            / str(sandbox_id)
            / str(checkpoint_id)
            / kind
            / _safe_name(name)
        )

    def put_manifest(self, manifest: CheckpointManifest) -> None:
        manifest.validate_schema()
        manifest.validate_integrity()
        path = self._manifest_path(manifest.sandbox_id, manifest.checkpoint_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(manifest.to_dict(), sort_keys=True, indent=2))
        tmp.replace(path)
        logger.debug(
            "Stored manifest for sandbox=%s checkpoint=%s at %s",
            manifest.sandbox_id,
            manifest.checkpoint_id,
            path,
        )

    def get_manifest(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> CheckpointManifest:
        path = self._manifest_path(sandbox_id, checkpoint_id)
        if not path.exists():
            raise FileNotFoundError(f"manifest not found: {path}")
        raw = json.loads(path.read_text())
        logger.debug("Loaded manifest for sandbox=%s checkpoint=%s from %s", sandbox_id, checkpoint_id, path)
        return CheckpointManifest.from_dict(raw)

    def put_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        artifact: ArtifactPayload,
    ) -> ArtifactReference:
        path = self._artifact_path(
            sandbox_id,
            checkpoint_id,
            artifact.kind.value,
            artifact.name,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(artifact.data)
        tmp.replace(path)

        digest = hashlib.sha256(artifact.data).hexdigest()
        size = len(artifact.data)
        rel = str(path.relative_to(self._root))
        reference = ArtifactReference(
            kind=artifact.kind,
            name=artifact.name,
            relative_path=rel,
            size_bytes=size,
            sha256=digest,
            metadata=dict(artifact.metadata),
        )
        logger.debug(
            "Stored artifact %s for sandbox=%s checkpoint=%s at %s (%d bytes)",
            artifact.name,
            sandbox_id,
            checkpoint_id,
            path,
            size,
        )
        return reference

    def get_artifact(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> bytes:
        # `sandbox_id` and `checkpoint_id` are part of the API contract for callers;
        # they also make intent explicit when reading by reference.
        _ = (sandbox_id, checkpoint_id)
        path = self._root / reference.relative_path
        if not path.exists():
            raise FileNotFoundError(f"artifact not found: {path}")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != reference.sha256:
            raise ValueError(f"artifact digest mismatch for {reference.name}")
        if len(payload) != reference.size_bytes:
            raise ValueError(f"artifact size mismatch for {reference.name}")
        logger.debug(
            "Loaded artifact %s for sandbox=%s checkpoint=%s from %s",
            reference.name,
            sandbox_id,
            checkpoint_id,
            path,
        )
        return payload

    def list_checkpoints(self, sandbox_id: SandboxId) -> list[CheckpointId]:
        sandbox_dir = self._manifests_root / str(sandbox_id)
        if not sandbox_dir.exists():
            return []

        manifests: list[tuple[str, CheckpointId]] = []
        for path in sandbox_dir.glob("*.json"):
            raw = json.loads(path.read_text())
            created_at = str(raw.get("created_at", ""))
            ckpt = CheckpointId(str(raw["checkpoint_id"]))
            manifests.append((created_at, ckpt))
        manifests.sort(key=lambda x: x[0])
        logger.debug("Listed %d checkpoints for sandbox=%s", len(manifests), sandbox_id)
        return [x[1] for x in manifests]

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
import shutil

from ..config import StorageConfig
from ..contracts import CheckpointManager
from ..ids import CheckpointId, SandboxId
from ..models import ArtifactPayload, ArtifactReference, CheckpointManifest
from ..runtime import CommandRunner, SubprocessCommandRunner

logger = logging.getLogger(__name__)


def _safe_name(value: str) -> str:
    if not value:
        raise ValueError("artifact name must be non-empty")
    return value.replace("/", "_")


class LocalCheckpointManager(CheckpointManager):
    def __init__(
        self,
        config: StorageConfig,
        *,
        command_runner: CommandRunner | None = None,
        zfs_bin: str = "zfs",
    ):
        self._config = config
        self._root = Path(config.root_dir)
        self._manifests_root = self._root / config.manifests_dirname
        self._artifacts_root = self._root / config.artifacts_dirname
        self._runner = command_runner or SubprocessCommandRunner()
        self._zfs_bin = zfs_bin
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

    def delete_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        manifest: CheckpointManifest | None = None
        manifest_path = self._manifest_path(sandbox_id, checkpoint_id)
        if manifest_path.exists():
            manifest = self.get_manifest(sandbox_id, checkpoint_id)

        if manifest is not None:
            self._delete_runtime_references(manifest)

        artifact_dir = self._artifacts_root / str(sandbox_id) / str(checkpoint_id)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir, ignore_errors=True)
        if manifest_path.exists():
            manifest_path.unlink()
        self._prune_empty_parents(manifest_path.parent, stop=self._manifests_root)
        self._prune_empty_parents(artifact_dir.parent, stop=self._artifacts_root)
        logger.debug("Deleted checkpoint sandbox=%s checkpoint=%s", sandbox_id, checkpoint_id)

    def delete_all_checkpoints(self, sandbox_id: SandboxId) -> None:
        for checkpoint_id in list(self.list_checkpoints(sandbox_id)):
            self.delete_checkpoint(sandbox_id, checkpoint_id)

    def handle_checkpoint_complete(self, manifest: CheckpointManifest) -> None:
        _ = manifest

    def handle_restore_complete(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        _ = (sandbox_id, checkpoint_id)

    def _delete_runtime_references(self, manifest: CheckpointManifest) -> None:
        for reference in [*manifest.process_artifacts, *manifest.filesystem_artifacts]:
            payload = self._load_artifact_payload(manifest.sandbox_id, manifest.checkpoint_id, reference)
            if payload is None:
                continue
            if reference.kind.value == "process":
                self._delete_process_runtime_paths(payload)
            elif reference.kind.value == "filesystem":
                self._delete_filesystem_runtime_paths(payload)

    def _load_artifact_payload(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        reference: ArtifactReference,
    ) -> dict[str, object] | None:
        try:
            raw = self.get_artifact(sandbox_id, checkpoint_id, reference)
        except FileNotFoundError:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _delete_process_runtime_paths(self, payload: dict[str, object]) -> None:
        checkpoint_location = payload.get("process_checkpoint_location")
        if checkpoint_location:
            checkpoint_root = Path(str(checkpoint_location)).parent
            if checkpoint_root.exists():
                shutil.rmtree(checkpoint_root, ignore_errors=True)
        status = payload.get("status", {})
        if not isinstance(status, dict):
            return
        metadata = status.get("metadata", {})
        if not isinstance(metadata, dict):
            return
        image_path = metadata.get("image_path")
        if image_path:
            checkpoint_root = Path(str(image_path)).parent
            if checkpoint_root.exists():
                shutil.rmtree(checkpoint_root, ignore_errors=True)

    def _delete_filesystem_runtime_paths(self, payload: dict[str, object]) -> None:
        snapshot = None
        filesystem = payload.get("filesystem", {})
        if isinstance(filesystem, dict):
            snapshot = filesystem.get("snapshot")
        if snapshot is None:
            status = payload.get("status", {})
            if isinstance(status, dict):
                metadata = status.get("metadata", {})
                if isinstance(metadata, dict):
                    snapshot = metadata.get("snapshot")
        if snapshot is None:
            return
        result = self._runner.run([self._zfs_bin, "destroy", str(snapshot)])
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "dataset does not exist" in stderr or "snapshot does not exist" in stderr:
                return
            logger.warning(
                "Failed to destroy filesystem snapshot %s rc=%d stderr=%s",
                snapshot,
                result.returncode,
                stderr,
            )

    def _prune_empty_parents(self, start: Path, *, stop: Path) -> None:
        current = start
        while current != stop and current.exists():
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent

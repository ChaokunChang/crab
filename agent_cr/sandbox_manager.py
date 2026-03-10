from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

from .contracts import SandboxManager
from .ids import SandboxId
from .models import SandboxDescription
from .runtime.base import CommandRunner, SubprocessCommandRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuncSandboxManagerPaths:
    state_root: Path = Path("/run/agent-cr/runc")
    bundle_root: Path = Path("/var/lib/agent-cr/bundles")
    metadata_root: Path = Path("/var/lib/agent-cr/sandbox-metadata")
    zfs_dataset_prefix: str = "agentcr/sandboxes"


class InMemorySandboxManager(SandboxManager):
    def __init__(self) -> None:
        self._lock = Lock()
        self._items: dict[SandboxId, SandboxDescription] = {}

    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        with self._lock:
            sandbox_id = SandboxId.new()
            self._items[sandbox_id] = SandboxDescription(
                sandbox_id=sandbox_id,
                runtime_name=runtime_name,
                status="running",
                metadata=dict(metadata or {}),
            )
        logger.info("Launched in-memory sandbox %s with runtime=%s", sandbox_id, runtime_name)
        return sandbox_id

    def stop(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            cur = self._items[sandbox_id]
            self._items[sandbox_id] = replace(cur, status="stopped")
        logger.info("Stopped in-memory sandbox %s", sandbox_id)

    def delete(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            self._items.pop(sandbox_id)
        logger.info("Deleted in-memory sandbox %s", sandbox_id)

    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        with self._lock:
            return self._items[sandbox_id]


class RuncSandboxManager(SandboxManager):
    def __init__(
        self,
        *,
        paths: RuncSandboxManagerPaths | None = None,
        command_runner: CommandRunner | None = None,
        runtime_bin: str = "runc",
        zfs_bin: str = "zfs",
    ) -> None:
        self._paths = paths or RuncSandboxManagerPaths()
        self._runner = command_runner or SubprocessCommandRunner()
        self._runtime_bin = runtime_bin
        self._zfs_bin = zfs_bin
        self._lock = Lock()
        self._items: dict[SandboxId, SandboxDescription] = {}
        self._paths.metadata_root.mkdir(parents=True, exist_ok=True)

    def launch(self, runtime_name: str, metadata: dict[str, object] | None = None) -> SandboxId:
        if runtime_name != "runc":
            raise ValueError(f"unsupported runtime for real sandbox manager: {runtime_name}")

        sandbox_id = SandboxId(str((metadata or {}).get("sandbox_id", SandboxId.new())))
        md = dict(metadata or {})
        bundle_path = Path(str(md["bundle_path"])) if "bundle_path" in md else self._paths.bundle_root / str(sandbox_id)
        rootfs_path = bundle_path / "rootfs"
        dataset = str(md.get("zfs_dataset", f"{self._paths.zfs_dataset_prefix}/{sandbox_id}"))
        logger.info(
            "Launching runc sandbox %s with bundle=%s dataset=%s",
            sandbox_id,
            bundle_path,
            dataset,
        )

        bundle_path.mkdir(parents=True, exist_ok=True)
        rootfs_path.mkdir(parents=True, exist_ok=True)
        self._run([self._zfs_bin, "create", "-o", f"mountpoint={rootfs_path}", dataset])
        for rel in md.get("rootfs_init_dirs", []):
            (rootfs_path / str(rel)).mkdir(parents=True, exist_ok=True)
        for item in md.get("rootfs_copy_paths", []):
            source = Path(str(item["source"]))
            destination = rootfs_path / str(item["destination"]).lstrip("/")
            if source.is_dir():
                shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination, follow_symlinks=True)
        self._run([self._runtime_bin, "--root", str(self._paths.state_root), "run", "-d", "--bundle", str(bundle_path), str(sandbox_id)])

        description = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name=runtime_name,
            status="running",
            metadata={**md, "bundle_path": str(bundle_path), "rootfs_path": str(rootfs_path), "zfs_dataset": dataset},
        )
        with self._lock:
            self._items[sandbox_id] = description
        self._persist(description)
        logger.info("Sandbox %s is running with rootfs=%s", sandbox_id, rootfs_path)
        return sandbox_id

    def stop(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        logger.info("Stopping sandbox %s", sandbox_id)
        self._run([self._runtime_bin, "--root", str(self._paths.state_root), "kill", str(sandbox_id), "TERM"])
        updated = replace(description, status="stopped")
        with self._lock:
            self._items[sandbox_id] = updated
        self._persist(updated)
        logger.info("Sandbox %s stopped", sandbox_id)

    def delete(self, sandbox_id: SandboxId) -> None:
        description = self.describe(sandbox_id)
        logger.info("Deleting sandbox %s", sandbox_id)
        self._run([self._runtime_bin, "--root", str(self._paths.state_root), "delete", "-f", str(sandbox_id)])
        dataset = str(description.metadata.get("zfs_dataset", ""))
        if dataset:
            self._run([self._zfs_bin, "destroy", "-r", dataset])
        metadata_path = self._metadata_path(sandbox_id)
        if metadata_path.exists():
            metadata_path.unlink()
        with self._lock:
            self._items.pop(sandbox_id, None)
        logger.info("Sandbox %s deleted", sandbox_id)

    def describe(self, sandbox_id: SandboxId) -> SandboxDescription:
        with self._lock:
            current = self._items.get(sandbox_id)
        if current is not None:
            return current
        path = self._metadata_path(sandbox_id)
        if not path.exists():
            raise KeyError(sandbox_id)
        raw = json.loads(path.read_text())
        description = SandboxDescription(
            sandbox_id=sandbox_id,
            runtime_name=str(raw["runtime_name"]),
            status=str(raw["status"]),
            metadata=dict(raw.get("metadata", {})),
        )
        with self._lock:
            self._items[sandbox_id] = description
        logger.debug("Loaded sandbox %s description from %s", sandbox_id, path)
        return description

    def _persist(self, description: SandboxDescription) -> None:
        path = self._metadata_path(description.sandbox_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "sandbox_id": str(description.sandbox_id),
                    "runtime_name": description.runtime_name,
                    "status": description.status,
                    "metadata": description.metadata,
                },
                sort_keys=True,
                indent=2,
            )
        )
        tmp.replace(path)
        logger.debug("Persisted sandbox %s metadata to %s", description.sandbox_id, path)

    def _metadata_path(self, sandbox_id: SandboxId) -> Path:
        return self._paths.metadata_root / f"{sandbox_id}.json"

    def _run(self, command: list[str]) -> None:
        logger.debug("Running sandbox manager command: %s", " ".join(command))
        result = self._runner.run(command)
        if result.returncode != 0:
            logger.error(
                "Sandbox manager command failed rc=%d command=%s stderr=%s",
                result.returncode,
                " ".join(command),
                result.stderr.strip(),
            )
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(command)}"
                f"\nstderr: {result.stderr.strip()}"
            )
        logger.debug("Sandbox manager command completed: %s", " ".join(command))

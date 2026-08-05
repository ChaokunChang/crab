from __future__ import annotations

import logging
import uuid
from dataclasses import replace
from pathlib import Path

from ..ids import CheckpointId, SandboxId
from ..models import RuntimeOperationStatus
from .fs_provider import (
    DatasetResolver,
    FilesystemProvider,
    FsCommandExecutor,
    FsStatusExecutor,
    RootfsResolver,
)

logger = logging.getLogger(__name__)

_FS_REF_PREFIX = "btrfs:"
_TRASH_INFIX = ".trash-"


class BtrfsProvider(FilesystemProvider):
    """Btrfs-backed filesystem provider.

    Semantic mapping from the ZFS reference implementation:

    - A sandbox "dataset" is a btrfs subvolume living at its dataset path
      under ``btrfs_root`` (``<root>/sandboxes/<sandbox_id>``). Btrfs has
      no ``mountpoint=`` property, so the rootfs the runtime sees is a
      bind mount of the subvolume — uniform for every sandbox, mirroring
      ZFS mountpoint semantics 1:1.
    - Checkpoints are read-only subvolume snapshots stored flat next to
      the live subvolume as ``<dataset>@<checkpoint_id>`` (same naming
      shape as ZFS snapshots).
    - ``restore_filesystem`` is emulated (btrfs has no in-place
      rollback): unmount the rootfs bind, rename the live subvolume to a
      trash name, take a writable snapshot of the checkpoint back onto
      the live path, re-bind, and best-effort delete the trash. Restore
      only runs while the container is stopped, so the swap is safe.
      Unlike ``zfs rollback -r``, restoring does NOT destroy snapshots
      taken after the restore point: btrfs snapshots are independent
      subvolumes, so later checkpoints stay restorable and retention
      keeps governing their lifetime (accepted design decision; the
      divergence is disk-usage-only because restore selection is
      manifest-driven).
    - ``promote`` is a no-op: btrfs snapshots share extents without a
      clone-origin dependency, so destroying a fork's source never
      invalidates the fork.
    - Per-snapshot byte stats need qgroups, which carry real overhead;
      they are off by default and stats degrade to unknown (telemetry
      tolerates missing byte counts).
    """

    def __init__(
        self,
        *,
        btrfs_root: Path,
        runtime_name: str,
        run_command: FsCommandExecutor,
        run_status: FsStatusExecutor,
        dataset_resolver: DatasetResolver,
        rootfs_resolver: RootfsResolver,
        btrfs_bin: str = "btrfs",
        qgroups_enabled: bool = False,
    ) -> None:
        self._root = Path(btrfs_root)
        self._runtime_name = runtime_name
        self._run_command = run_command
        self._run_status = run_status
        self._dataset_resolver = dataset_resolver
        self._rootfs_resolver = rootfs_resolver
        self._btrfs_bin = btrfs_bin
        self._qgroups_enabled = qgroups_enabled

    @property
    def name(self) -> str:
        return "btrfs"

    # ------------------------------------------------------------------
    # Naming / layout
    # ------------------------------------------------------------------

    def default_dataset_name(self, sandbox_id: SandboxId) -> str:
        return str(self._root / "sandboxes" / str(sandbox_id))

    def _safe_prefix(self) -> str:
        return "".join(
            char if char.isalnum() or char in {"-", "_", "."} else "_"
            for char in str(self._root)
        )

    def shared_rootfs_details(self, key: str, *, persist_across_runs: bool) -> tuple[str, Path]:
        scope = "persistent" if persist_across_runs else "run"
        dataset = str(self._root / "shared" / scope / key)
        # The subvolume path doubles as the materialization mountpoint:
        # unlike ZFS there is no separate mount to arrange for the cache.
        return dataset, Path(dataset)

    def shared_rootfs_lock_path(self, key: str, *, persist_across_runs: bool) -> Path:
        scope = "persistent" if persist_across_runs else "run"
        return Path("/tmp/crab-rootfs-cache-locks") / self._safe_prefix() / scope / f"{key}.lock"

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def _subvolume_exists(self, path: str, *, operation: str) -> bool:
        result = self._run_command(
            [self._btrfs_bin, "subvolume", "show", path],
            operation=operation,
            check=False,
            metadata={"subvolume": path},
        )
        return result.returncode == 0

    def object_exists(self, name: str) -> bool:
        return self._subvolume_exists(name, operation="sandbox.btrfs_show")

    def dataset_exists(self, dataset: str) -> bool:
        return self._subvolume_exists(dataset, operation="sandbox.btrfs_exists")

    def snapshot_exists(self, snapshot: str) -> bool:
        return self._subvolume_exists(snapshot, operation="sandbox.btrfs_snapshot_exists")

    def _query_snapshot_sizes(
        self,
        snapshot: str,
        *,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> tuple[int | None, int | None]:
        if not self._qgroups_enabled:
            return None, None
        result = self._run_command(
            [self._btrfs_bin, "qgroup", "show", "--raw", "-f", snapshot],
            operation="sandbox.btrfs_snapshot_stats",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            check=False,
            metadata={"snapshot": snapshot},
        )
        if result.returncode != 0:
            return None, None
        # Last data row: `<qgroupid> <referenced> <exclusive> [path]`.
        # `exclusive` approximates ZFS `written`, `referenced` maps to `used`.
        for line in reversed(result.stdout.splitlines()):
            parts = line.split()
            if len(parts) >= 3 and "/" in parts[0]:
                try:
                    referenced = int(parts[1])
                    exclusive = int(parts[2])
                except ValueError:
                    continue
                return exclusive, referenced
        return None, None

    # ------------------------------------------------------------------
    # Bind-mount helpers
    # ------------------------------------------------------------------

    def _bind_rootfs(self, dataset: str, mountpoint: Path, *, sandbox_id: SandboxId | None) -> None:
        if str(mountpoint) == dataset:
            return
        mountpoint.mkdir(parents=True, exist_ok=True)
        self._run_command(
            ["mount", "--bind", dataset, str(mountpoint)],
            operation="sandbox.btrfs_bind_rootfs",
            sandbox_id=sandbox_id,
            metadata={"subvolume": dataset, "mountpoint": str(mountpoint)},
        )

    def _unbind_rootfs(self, mountpoint: Path, *, sandbox_id: SandboxId | None) -> None:
        # Best-effort: not-mounted is the common case for fresh launches
        # and for datasets that were never bind-mounted.
        self._run_command(
            ["umount", str(mountpoint)],
            operation="sandbox.btrfs_unbind_rootfs",
            sandbox_id=sandbox_id,
            check=False,
            metadata={"mountpoint": str(mountpoint)},
        )

    # ------------------------------------------------------------------
    # Launch / shared-rootfs mutations
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        dataset: str,
        mountpoint: Path,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        Path(dataset).parent.mkdir(parents=True, exist_ok=True)
        self._run_command(
            [self._btrfs_bin, "subvolume", "create", dataset],
            operation=operation,
            sandbox_id=sandbox_id,
            metadata={"subvolume": dataset, "mountpoint": str(mountpoint)},
            timeout_seconds=timeout_seconds,
        )
        self._bind_rootfs(dataset, mountpoint, sandbox_id=sandbox_id)

    def create_snapshot(self, dataset: str, snapshot: str, *, operation: str) -> None:
        self._run_command(
            [self._btrfs_bin, "subvolume", "snapshot", "-r", dataset, snapshot],
            operation=operation,
            metadata={"subvolume": dataset, "snapshot": snapshot},
        )

    def clone_shared_base(
        self,
        shared_dataset: str,
        shared_snapshot: str,
        dataset: str,
        rootfs_path: Path,
        *,
        sandbox_id: SandboxId,
    ) -> None:
        Path(dataset).parent.mkdir(parents=True, exist_ok=True)
        self._run_command(
            [self._btrfs_bin, "subvolume", "snapshot", shared_snapshot, dataset],
            operation="sandbox.btrfs_clone_launch_rootfs",
            sandbox_id=sandbox_id,
            metadata={
                "source_subvolume": shared_dataset,
                "target_subvolume": dataset,
                "snapshot": shared_snapshot,
                "mountpoint": str(rootfs_path),
            },
        )
        self._bind_rootfs(dataset, rootfs_path, sandbox_id=sandbox_id)

    def destroy_dataset(
        self,
        dataset: str,
        *,
        operation: str,
        sandbox_id: SandboxId | None = None,
    ) -> None:
        # ZFS `destroy -r` also removes the dataset's snapshots; mirror
        # that by deleting the flat `<dataset>@*` siblings first.
        for snapshot in self._sibling_snapshots(dataset):
            self._run_command(
                [self._btrfs_bin, "subvolume", "delete", snapshot],
                operation=operation,
                sandbox_id=sandbox_id,
                check=False,
                metadata={"subvolume": dataset, "snapshot": snapshot},
            )
        self._run_command(
            [self._btrfs_bin, "subvolume", "delete", dataset],
            operation=operation,
            sandbox_id=sandbox_id,
            check=False,
            metadata={"subvolume": dataset},
        )

    def _sibling_snapshots(self, dataset: str) -> list[str]:
        base = Path(dataset)
        try:
            return sorted(str(path) for path in base.parent.glob(f"{base.name}@*"))
        except OSError:
            return []

    def _trash_paths(self, dataset: str) -> list[str]:
        base = Path(dataset)
        try:
            return sorted(str(path) for path in base.parent.glob(f"{base.name}{_TRASH_INFIX}*"))
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Checkpoint / restore / fork flows
    # ------------------------------------------------------------------

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        status = self._run_status(
            [self._btrfs_bin, "subvolume", "snapshot", "-r", dataset, snapshot],
            operation="sandbox.checkpoint_filesystem",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=self.filesystem_checkpoint_metadata(sandbox_id, checkpoint_id),
        )
        written_bytes, used_bytes = self._query_snapshot_sizes(snapshot, sandbox_id=sandbox_id, checkpoint_id=checkpoint_id)
        return replace(
            status,
            metadata={
                **status.metadata,
                "checkpoint_scope": "filesystem_only",
                "filesystem_checkpoint_written_bytes": written_bytes,
                "filesystem_checkpoint_used_bytes": used_bytes,
            },
        )

    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        mountpoint = self._rootfs_resolver(sandbox_id)
        metadata: dict[str, object] = {
            "phase": "filesystem_restore",
            "runtime": self._runtime_name,
            "dataset": dataset,
            "snapshot": snapshot,
            "mountpoint": str(mountpoint),
        }
        # Reclaim trash left by an earlier crashed restore before making
        # more; best-effort, never blocks the current restore.
        for stale in self._trash_paths(dataset):
            self._run_command(
                [self._btrfs_bin, "subvolume", "delete", stale],
                operation="sandbox.btrfs_reclaim_restore_trash",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"subvolume": dataset, "trash": stale},
            )
        self._unbind_rootfs(mountpoint, sandbox_id=sandbox_id)
        # Crash recovery: if a previous restore died between the rename
        # and the snapshot-back, the live path is already vacant — skip
        # straight to rematerializing from the checkpoint snapshot.
        if self.dataset_exists(dataset):
            trash = f"{dataset}{_TRASH_INFIX}{uuid.uuid4().hex}"
            self._run_command(
                ["mv", dataset, trash],
                operation="sandbox.btrfs_rollback_swap_out",
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                metadata={**metadata, "trash": trash},
            )
        status = self._run_status(
            [self._btrfs_bin, "subvolume", "snapshot", snapshot, dataset],
            operation="sandbox.restore_filesystem",
            sandbox_id=sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata=metadata,
        )
        self._bind_rootfs(dataset, mountpoint, sandbox_id=sandbox_id)
        for trash_path in self._trash_paths(dataset):
            self._run_command(
                [self._btrfs_bin, "subvolume", "delete", trash_path],
                operation="sandbox.btrfs_rollback_delete_trash",
                sandbox_id=sandbox_id,
                check=False,
                metadata={"subvolume": dataset, "trash": trash_path},
            )
        return status

    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        return {
            "phase": "filesystem_checkpoint",
            "runtime": self._runtime_name,
            "dataset": dataset,
            "snapshot": snapshot,
            "mountpoint": str(self._rootfs_resolver(sandbox_id)),
            "fs_ref": f"{_FS_REF_PREFIX}{snapshot}",
        }

    def destroy_filesystem_dataset(self, sandbox_id: SandboxId, dataset: str) -> None:
        self._unbind_rootfs(self._rootfs_resolver(sandbox_id), sandbox_id=sandbox_id)
        self.destroy_dataset(dataset, operation="sandbox.btrfs_destroy", sandbox_id=sandbox_id)

    def promote_filesystem_dataset(self, sandbox_id: SandboxId) -> None:
        # Btrfs snapshots share extents without a clone-origin dependency;
        # a fork never needs promotion to outlive its source.
        logger.debug(
            "Btrfs promote is a no-op (no clone-origin dependency) sandbox=%s",
            sandbox_id,
        )

    def clone_filesystem_snapshot(
        self,
        source_sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        target_sandbox_id: SandboxId,
        *,
        target_rootfs_path: Path,
    ) -> str:
        source_dataset = self._dataset_resolver(source_sandbox_id)
        target_dataset = str(self._root / "sandboxes" / str(target_sandbox_id))
        self.destroy_dataset(
            target_dataset,
            operation="sandbox.btrfs_destroy_clone_target",
            sandbox_id=target_sandbox_id,
        )
        snapshot = f"{source_dataset}@{checkpoint_id}"
        Path(target_dataset).parent.mkdir(parents=True, exist_ok=True)
        self._run_command(
            [self._btrfs_bin, "subvolume", "snapshot", snapshot, target_dataset],
            operation="sandbox.btrfs_clone",
            sandbox_id=target_sandbox_id,
            checkpoint_id=checkpoint_id,
            metadata={
                "source_subvolume": source_dataset,
                "target_subvolume": target_dataset,
                "mountpoint": str(target_rootfs_path),
            },
        )
        self._bind_rootfs(target_dataset, target_rootfs_path, sandbox_id=target_sandbox_id)
        return target_dataset

    def discard_partial_checkpoint(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> None:
        dataset = self._dataset_resolver(sandbox_id)
        snapshot = f"{dataset}@{checkpoint_id}"
        if self.snapshot_exists(snapshot):
            try:
                self._run_command(
                    [self._btrfs_bin, "subvolume", "delete", snapshot],
                    operation="sandbox.discard_partial_checkpoint",
                    check=False,
                    metadata={"snapshot": snapshot},
                )
            except Exception:
                logger.exception(
                    "Failed to destroy partial btrfs snapshot sandbox=%s checkpoint=%s snapshot=%s",
                    sandbox_id,
                    checkpoint_id,
                    snapshot,
                )

    def destroy_snapshot_ref(self, fs_ref: str) -> None:
        snapshot = fs_ref[len(_FS_REF_PREFIX):] if fs_ref.startswith(_FS_REF_PREFIX) else fs_ref
        if not snapshot:
            return
        result = self._run_command(
            [self._btrfs_bin, "subvolume", "delete", snapshot],
            operation="storage.btrfs_destroy_snapshot_ref",
            check=False,
            metadata={"snapshot": snapshot, "fs_ref": fs_ref},
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if "No such file or directory" in stderr or "not a subvolume" in stderr:
                return
            logger.warning(
                "Failed to destroy filesystem snapshot %s rc=%d stderr=%s",
                snapshot,
                result.returncode,
                stderr,
            )

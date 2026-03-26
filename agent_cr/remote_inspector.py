from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime
from threading import Lock

from .contracts import SandboxInspector
from .http_utils import HttpStatusError, ThreadLocalHttpClient
from .ids import SandboxId
from .models import SandboxSnapshot, utc_now

logger = logging.getLogger(__name__)


def _parse_ts(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


@dataclass(frozen=True)
class HostInspectorServiceClient:
    base_url: str
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_http_client",
            ThreadLocalHttpClient(self.base_url, timeout_seconds=self.timeout_s),
        )

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        return self._http_client.post_json(path, payload)

    def register_sandbox(
        self,
        sandbox_id: SandboxId,
        runtime: str,
        object_id: str,
        *,
        ignore_process_rules: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {"sandbox_id": str(sandbox_id), "runtime": runtime, "object_id": object_id}
        if ignore_process_rules is not None:
            payload["ignore_process_rules"] = ignore_process_rules
        return self._post("/register", payload)

    def unregister_sandbox(self, sandbox_id: SandboxId) -> dict[str, object]:
        return self._post("/unregister", {"sandbox_id": str(sandbox_id)})

    def get_proc_and_fs_status(self, sandbox_id: SandboxId) -> dict[str, object]:
        return self._post("/get_proc_and_fs_status", {"sandbox_id": str(sandbox_id)})

    def reset_sandbox(self, sandbox_id: SandboxId, at: datetime | None) -> dict[str, object]:
        payload: dict[str, object] = {"sandbox_id": str(sandbox_id)}
        if at is not None:
            payload["at"] = at.isoformat()
        return self._post("/reset", payload)

    def close(self) -> None:
        self._http_client.close()


class RemoteSandboxInspector(SandboxInspector):
    def __init__(self, service_client: HostInspectorServiceClient) -> None:
        self._service_client = service_client
        self._lock = Lock()
        self._snapshots: dict[SandboxId, SandboxSnapshot] = {}
        self._pending_reset_at: dict[SandboxId, datetime] = {}

    def upsert_snapshot(self, snapshot: SandboxSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.sandbox_id] = snapshot

        if snapshot.process_changed or snapshot.filesystem_changed:
            with self._lock:
                self._pending_reset_at.pop(snapshot.sandbox_id, None)
            return

        reset_at = snapshot.last_checkpoint_at or snapshot.observed_at
        try:
            self._service_client.reset_sandbox(snapshot.sandbox_id, reset_at)
        except Exception as exc:  # noqa: BLE001
            if self._is_unknown_sandbox_error(exc):
                with self._lock:
                    self._pending_reset_at[snapshot.sandbox_id] = reset_at
                logger.debug(f"pending a sandbox reset for sandbox={snapshot.sandbox_id}")
                return
            logger.debug(
                "Failed to sync clean snapshot baseline for sandbox %s",
                snapshot.sandbox_id,
                exc_info=True,
            )
        else:
            with self._lock:
                self._pending_reset_at.pop(snapshot.sandbox_id, None)

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        self._apply_pending_reset(sandbox_id)
        with self._lock:
            local_snapshot = self._snapshots.get(sandbox_id)
        try:
            remote_snapshot = self._read_remote_snapshot(sandbox_id)
        except Exception as exc:  # noqa: BLE001
            if local_snapshot is not None:
                return local_snapshot
            return SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name="remote-inspector-unavailable",
                is_running=True,
                process_changed=True,
                filesystem_changed=True,
                observed_at=utc_now(),
                last_checkpoint_at=None,
                metadata={"inspector_error": str(exc)},
            )
        return self._merge_snapshots(local_snapshot, remote_snapshot)

    def mark_checkpoint_complete(
        self,
        sandbox_id: SandboxId,
        *,
        process: bool,
        filesystem: bool,
        at: datetime,
    ) -> None:
        if not process and not filesystem:
            return
        self._service_client.reset_sandbox(sandbox_id, at)
        with self._lock:
            snapshot = self._snapshots.get(sandbox_id)
            self._pending_reset_at.pop(sandbox_id, None)
            if snapshot is None:
                return
            self._snapshots[sandbox_id] = replace(
                snapshot,
                process_changed=False if process else snapshot.process_changed,
                filesystem_changed=False if filesystem else snapshot.filesystem_changed,
                observed_at=max(snapshot.observed_at, at),
                last_checkpoint_at=at,
            )

    def _apply_pending_reset(self, sandbox_id: SandboxId) -> None:
        with self._lock:
            reset_at = self._pending_reset_at.get(sandbox_id)
        if reset_at is None:
            return
        try:
            self._service_client.reset_sandbox(sandbox_id, reset_at)
        except Exception as exc:  # noqa: BLE001
            if not self._is_unknown_sandbox_error(exc):
                logger.debug(
                    "Failed to apply deferred remote reset for sandbox %s",
                    sandbox_id,
                    exc_info=True,
                )
            return
        with self._lock:
            self._pending_reset_at.pop(sandbox_id, None)

    def _read_remote_snapshot(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        payload = self._service_client.get_proc_and_fs_status(sandbox_id)
        status = dict(payload["status"])
        metadata = dict(status.get("metadata", {}))
        last_reset_at = _parse_ts(status.get("last_reset_at"))
        if last_reset_at is not None:
            metadata["host_last_reset_at"] = last_reset_at.isoformat()
        return SandboxSnapshot(
            sandbox_id=sandbox_id,
            runtime_name=str(status["runtime_name"]),
            is_running=bool(status["is_running"]),
            process_changed=bool(status["process_changed"]),
            filesystem_changed=bool(status["filesystem_changed"]),
            observed_at=_parse_ts(status.get("observed_at")) or utc_now(),
            # Host-inspector resets also happen for non-checkpoint baselines.
            # Only explicit mark_checkpoint_complete() calls should advance the
            # scheduler-visible checkpoint timestamp.
            last_checkpoint_at=None,
            metadata=metadata,
        )

    def _merge_snapshots(
        self,
        local_snapshot: SandboxSnapshot | None,
        remote_snapshot: SandboxSnapshot,
    ) -> SandboxSnapshot:
        if local_snapshot is None:
            return remote_snapshot

        prefer_local = local_snapshot.observed_at >= remote_snapshot.observed_at
        if prefer_local:
            runtime_name = local_snapshot.runtime_name
            is_running = local_snapshot.is_running
        else:
            runtime_name = remote_snapshot.runtime_name
            is_running = remote_snapshot.is_running

        checkpoints = [x for x in (local_snapshot.last_checkpoint_at, remote_snapshot.last_checkpoint_at) if x is not None]
        last_checkpoint_at = max(checkpoints) if checkpoints else None
        return SandboxSnapshot(
            sandbox_id=remote_snapshot.sandbox_id,
            runtime_name=runtime_name,
            is_running=is_running,
            process_changed=local_snapshot.process_changed or remote_snapshot.process_changed,
            filesystem_changed=local_snapshot.filesystem_changed or remote_snapshot.filesystem_changed,
            observed_at=max(local_snapshot.observed_at, remote_snapshot.observed_at),
            last_checkpoint_at=last_checkpoint_at,
            metadata={**remote_snapshot.metadata, **local_snapshot.metadata},
        )

    def _is_unknown_sandbox_error(self, exc: Exception) -> bool:
        if isinstance(exc, KeyError):
            return True
        if isinstance(exc, HttpStatusError):
            return exc.status_code == 404
        return False

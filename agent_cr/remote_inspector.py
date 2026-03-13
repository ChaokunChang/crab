from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime

from .contracts import SandboxInspector
from .ids import SandboxId
from .models import SandboxSnapshot, utc_now


def _parse_ts(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    return datetime.fromisoformat(raw)


@dataclass(frozen=True)
class HostInspectorServiceClient:
    base_url: str
    timeout_s: float = 5.0

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def register_sandbox(self, sandbox_id: SandboxId, runtime: str, object_id: str) -> dict[str, object]:
        return self._post(
            "/register",
            {"sandbox_id": str(sandbox_id), "runtime": runtime, "object_id": object_id},
        )

    def unregister_sandbox(self, sandbox_id: SandboxId) -> dict[str, object]:
        return self._post("/unregister", {"sandbox_id": str(sandbox_id)})

    def get_proc_and_fs_status(self, sandbox_id: SandboxId) -> dict[str, object]:
        return self._post("/get_proc_and_fs_status", {"sandbox_id": str(sandbox_id)})

    def reset_sandbox(self, sandbox_id: SandboxId, at: datetime | None) -> dict[str, object]:
        payload: dict[str, object] = {"sandbox_id": str(sandbox_id)}
        if at is not None:
            payload["at"] = at.isoformat()
        return self._post("/reset", payload)


class RemoteSandboxInspector(SandboxInspector):
    def __init__(self, service_client: HostInspectorServiceClient) -> None:
        self._service_client = service_client

    def inspect(self, sandbox_id: SandboxId) -> SandboxSnapshot:
        try:
            payload = self._service_client.get_proc_and_fs_status(sandbox_id)
            status = dict(payload["status"])
            return SandboxSnapshot(
                sandbox_id=sandbox_id,
                runtime_name=str(status["runtime_name"]),
                is_running=bool(status["is_running"]),
                process_changed=bool(status["process_changed"]),
                filesystem_changed=bool(status["filesystem_changed"]),
                observed_at=_parse_ts(status.get("observed_at")) or utc_now(),
                last_checkpoint_at=_parse_ts(status.get("last_reset_at")),
                metadata=dict(status.get("metadata", {})),
            )
        except Exception as exc:  # noqa: BLE001
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

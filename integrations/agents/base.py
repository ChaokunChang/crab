from __future__ import annotations

import json
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event

from crab import EBPFEvent, EBPFEventKind, InMemoryEBPFEventCollector, Runtime, SandboxId
from crab.contracts import TelemetrySink
from crab.models import utc_now

from .contracts import SandboxHandle


@dataclass(frozen=True)
class TaskDescription:
    prompt: str

    @classmethod
    def from_json_value(cls, value: object) -> "TaskDescription":
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(prompt=value)
        if isinstance(value, dict):
            prompt = value.get("prompt")
            if isinstance(prompt, str):
                return cls(prompt=prompt)
        raise ValueError(f"invalid task description: {value!r}")


@dataclass(frozen=True)
class TaskConfig:
    options: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json_value(cls, value: object) -> "TaskConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError(f"invalid task config: {value!r}")
        options = value.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"task config options must be a dict, got {options!r}")
        return cls(options=dict(options))


class BaseAgent(ABC):
    agent_type = "base"
    requires_manual_task_launch = False
    requires_network_namespace = False

    @property
    def sandbox_manager(self) -> Runtime | None:
        return self.runtime

    @sandbox_manager.setter
    def sandbox_manager(self, value: Runtime | None) -> None:
        self.runtime = value

    def __init__(
        self,
        sandbox: SandboxHandle,
        task_description: TaskDescription,
        task_config: TaskConfig,
        *,
        runtime_state_root: Path | None = None,
        runtime: Runtime | None = None,
        sandbox_manager: Runtime | None = None,
        agent_host_dir: Path | None = None,
        llm_base_url: str | None = None,
        telemetry: TelemetrySink | None = None,
    ) -> None:
        if runtime is not None and sandbox_manager is not None and runtime is not sandbox_manager:
            raise ValueError("runtime and sandbox_manager refer to the same runtime and cannot differ")
        self.sandbox = sandbox
        self.task_description = task_description
        self.task_config = task_config
        self.runtime_state_root = runtime_state_root
        self.runtime = runtime if runtime is not None else sandbox_manager
        self.agent_host_dir = agent_host_dir
        self.llm_base_url = llm_base_url
        self.telemetry = telemetry
        self._activity_collector = InMemoryEBPFEventCollector()
        self._stop_requested = Event()

    @abstractmethod
    def perform_task(self) -> None:
        raise NotImplementedError

    def post_task_finish(self) -> None:
        if self.runtime is None:
            return
        self.runtime.delete_runtime(self.sandbox.sandbox_id, force=True, ignore_missing=True)

    def prepare_sandbox(self) -> None:
        return None

    def configure_bundle(self) -> None:
        return None

    def rootfs_init_dirs(self) -> list[str]:
        return [
            "work",
            "tmp",
            "proc",
            "dev",
            "dev/pts",
            "dev/shm",
            "dev/mqueue",
            "sys",
            "run",
            "var",
        ]

    def extra_launch_metadata(self) -> dict[str, object]:
        return {}

    def supports_relaunch_task(self) -> bool:
        return self.requires_manual_task_launch

    def survives_fault_relaunch(self) -> bool:
        return False

    def poll_status(self) -> dict[str, object]:
        raise RuntimeError(f"agent {self.agent_type} does not expose a benchmark status endpoint")

    def wait_for_progress(self, *, minimum_actions: int) -> dict[str, object]:
        raise RuntimeError(f"agent {self.agent_type} does not support progress-based readiness checks")

    def wait_for_action_delta(self, *, delta: int) -> dict[str, object]:
        raise RuntimeError(f"agent {self.agent_type} does not support action-delta progress checks")

    def wait_for_task_ready(self) -> None:
        return None

    def request_stop(self) -> None:
        self._stop_requested.set()

    def on_restore_complete(self) -> None:
        return None

    def resolve_sandbox_id(self) -> SandboxId:
        return SandboxId(str(self.sandbox.sandbox_id))

    def wait_for_sandbox_exit(self, *, poll_interval_s: float = 0.2) -> None:
        if self.runtime is None:
            return
        while True:
            if self._stop_requested.is_set():
                return
            state = self.runtime.inspect_runtime(self.sandbox.sandbox_id)
            if state.status.lower() in {"missing", "stopped"}:
                return
            if not state.is_running:
                return
            time.sleep(poll_interval_s)

    def wait_for_http_json(self, url: str, *, timeout_s: float = 60.0) -> dict[str, object]:
        deadline = time.time() + timeout_s
        last_exc: Exception | None = None
        while time.time() < deadline:
            if self._stop_requested.is_set():
                return dict(self.sandbox.last_status)
            try:
                with urllib.request.urlopen(url, timeout=2.0) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as exc:
                last_exc = exc
                time.sleep(0.2)
        raise RuntimeError(f"timed out waiting for {url}: {last_exc}")

    def wait_for_condition(
        self,
        predicate,
        *,
        timeout_s: float = 45.0,
        interval_s: float = 0.2,
        raise_on_timeout: bool = True,
    ) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self._stop_requested.is_set():
                return False
            if predicate():
                return True
            time.sleep(interval_s)
        if raise_on_timeout:
            raise RuntimeError("timed out waiting for agent condition")
        return False

    def _set_status(self, payload: dict[str, object]) -> None:
        self.sandbox.last_status = dict(payload)

    def _record_activity(self, payload: dict[str, object]) -> None:
        previous = dict(self.sandbox.last_status)
        current = dict(payload)
        mappings = [
            ("filesystem_actions", EBPFEventKind.FILE_WRITE, {"path": "/work/tool_artifact.txt"}),
            ("process_actions", EBPFEventKind.PROCESS_EXEC, {"command": "/bin/sh"}),
            ("network_actions", EBPFEventKind.NETWORK_EGRESS, {"address": "127.0.0.1"}),
        ]
        for field, kind, metadata in mappings:
            if int(current.get(field, 0)) > int(previous.get(field, 0)):
                self._activity_collector.record(
                    EBPFEvent(
                        sandbox_id=self.sandbox.sandbox_id,
                        kind=kind,
                        observed_at=utc_now(),
                        metadata=metadata,
                    )
                )
        self._set_status(current)

    def _recorded_activity_events(self) -> list[EBPFEvent]:
        return self._activity_collector.poll(self.sandbox.sandbox_id)

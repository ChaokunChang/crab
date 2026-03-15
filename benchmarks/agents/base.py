from __future__ import annotations

import json
import subprocess
import time
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_cr import SandboxId

if TYPE_CHECKING:
    from benchmarks.real_host_scenario_base import RealHostScenarioHarness, SandboxHandle


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
    minimum_actions: int = 0
    options: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_json_value(cls, value: object) -> "TaskConfig":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise ValueError(f"invalid task config: {value!r}")
        minimum_actions = int(value.get("minimum_actions", 0))
        options = value.get("options", {})
        if not isinstance(options, dict):
            raise ValueError(f"task config options must be a dict, got {options!r}")
        return cls(minimum_actions=minimum_actions, options=dict(options))


class BaseAgent(ABC):
    agent_type = "base"
    requires_manual_task_launch = False
    requires_network_namespace = False

    def __init__(
        self,
        harness: RealHostScenarioHarness,
        sandbox: SandboxHandle,
        task_description: TaskDescription,
        task_config: TaskConfig,
    ) -> None:
        self.harness = harness
        self.sandbox = sandbox
        self.task_description = task_description
        self.task_config = task_config

    @abstractmethod
    def perform_task(self) -> None:
        raise NotImplementedError

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

    def poll_status(self) -> dict[str, object]:
        raise RuntimeError(f"agent {self.agent_type} does not expose a benchmark status endpoint")

    def wait_for_progress(self, *, minimum_actions: int) -> dict[str, object]:
        raise RuntimeError(f"agent {self.agent_type} does not support progress-based readiness checks")

    def wait_for_action_delta(self, *, delta: int) -> dict[str, object]:
        raise RuntimeError(f"agent {self.agent_type} does not support action-delta progress checks")

    def wait_for_task_ready(self) -> None:
        return None

    def on_restore_complete(self) -> None:
        return None

    def resolve_sandbox_id(self) -> SandboxId:
        return SandboxId(str(self.sandbox.sandbox_id))

    def wait_for_sandbox_exit(self, *, poll_interval_s: float = 0.2) -> None:
        assert self.harness.runtime_state_root is not None
        command = ["runc", "--root", str(self.harness.runtime_state_root), "state", str(self.sandbox.sandbox_id)]
        while True:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                return
            try:
                payload = json.loads(result.stdout or "{}")
            except json.JSONDecodeError:
                return
            if str(payload.get("status", "")).lower() in {"stopped", "missing"}:
                return
            time.sleep(poll_interval_s)

    def wait_for_http_json(self, url: str, *, timeout_s: float = 30.0) -> dict[str, object]:
        deadline = time.time() + timeout_s
        last_exc: Exception | None = None
        while time.time() < deadline:
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
            if predicate():
                return True
            time.sleep(interval_s)
        if raise_on_timeout:
            raise RuntimeError("timed out waiting for agent condition")
        return False

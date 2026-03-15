from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from agent_cr import SandboxId

if TYPE_CHECKING:
    from .base import BaseAgent, TaskConfig, TaskDescription


@dataclass
class SandboxHandle:
    sandbox_id: SandboxId
    bundle_dir: Path
    status_port: int
    last_status: dict[str, object]
    status_host: str = "127.0.0.1"
    agent_type: str = "simulated"
    llm_service_type: str = "simulated"
    llm_base_url: str | None = None
    task_description: TaskDescription | None = None
    task_config: TaskConfig | None = None
    launch_source: str = "runc"
    launch_metadata: dict[str, object] = field(default_factory=dict)
    task_run: BaseAgent | None = None
    task_future: Future[None] | None = None

    @property
    def status_url(self) -> str:
        return f"http://{self.status_host}:{self.status_port}/status"

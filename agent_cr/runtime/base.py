from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..contracts import SandboxRuntimeAdapter
from ..ids import CheckpointId, SandboxId
from ..models import RuntimeCapabilities, RuntimeOperationStatus


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(ABC):
    @abstractmethod
    def run(self, command: list[str], *, cwd: Path | None = None) -> CommandResult:
        raise NotImplementedError


class SubprocessCommandRunner(CommandRunner):
    def __init__(self, timeout_seconds: float = 60.0):
        self._timeout_seconds = timeout_seconds

    def run(self, command: list[str], *, cwd: Path | None = None) -> CommandResult:
        detached = "-d" in command
        completed = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            check=False,
            stdout=(subprocess.DEVNULL if detached else subprocess.PIPE),
            stderr=(subprocess.DEVNULL if detached else subprocess.PIPE),
            text=True,
            timeout=self._timeout_seconds,
        )
        return CommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout="" if completed.stdout is None else completed.stdout,
            stderr="" if completed.stderr is None else completed.stderr,
        )


class CommandRuntimeAdapter(SandboxRuntimeAdapter):
    def __init__(
        self,
        *,
        name: str,
        version: str | None,
        command_runner: CommandRunner | None = None,
    ):
        self._name = name
        self._version = version
        self._runner = command_runner or SubprocessCommandRunner()

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str | None:
        return self._version

    def capabilities(self) -> RuntimeCapabilities:
        return RuntimeCapabilities(
            supports_process_checkpoint=True,
            supports_filesystem_checkpoint=True,
            supports_incremental_filesystem=True,
            supports_custom_checkpoint_dir=True,
        )

    def checkpoint_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        return self._execute(
            self._checkpoint_cmd(sandbox_id=sandbox_id, checkpoint_id=checkpoint_id),
            metadata=self._process_metadata(sandbox_id, checkpoint_id, phase="process_checkpoint"),
        )

    def restore_process(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        return self._execute(
            self._restore_cmd(sandbox_id=sandbox_id, checkpoint_id=checkpoint_id),
            metadata=self._process_metadata(sandbox_id, checkpoint_id, phase="process_restore"),
        )

    def checkpoint_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        return self._execute(
            self._filesystem_checkpoint_cmd(sandbox_id=sandbox_id, checkpoint_id=checkpoint_id),
            metadata=self._filesystem_metadata(sandbox_id, checkpoint_id, phase="filesystem_checkpoint"),
        )

    def restore_filesystem(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> RuntimeOperationStatus:
        return self._execute(
            self._filesystem_restore_cmd(sandbox_id=sandbox_id, checkpoint_id=checkpoint_id),
            metadata=self._filesystem_metadata(sandbox_id, checkpoint_id, phase="filesystem_restore"),
        )

    def process_checkpoint_location(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> str | None:
        metadata = self._process_metadata(sandbox_id, checkpoint_id, phase="process_checkpoint")
        image_path = metadata.get("image_path")
        return None if image_path is None else str(image_path)

    def filesystem_checkpoint_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
    ) -> dict[str, object]:
        return self._filesystem_metadata(sandbox_id, checkpoint_id, phase="filesystem_checkpoint")

    def _execute(self, command: list[str], *, metadata: dict[str, Any]) -> RuntimeOperationStatus:
        result = self._runner.run(command)
        if result.returncode != 0:
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(command)}"
                f"\nstderr: {result.stderr.strip()}"
            )
        merged = dict(metadata)
        merged["stdout"] = result.stdout.strip()
        merged["stderr"] = result.stderr.strip()
        return RuntimeOperationStatus(
            executed=True,
            reason="command_executed",
            command=result.command,
            metadata=merged,
        )

    @abstractmethod
    def _process_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        phase: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _filesystem_metadata(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        phase: str,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def _restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def _filesystem_checkpoint_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def _filesystem_restore_cmd(self, sandbox_id: SandboxId, checkpoint_id: CheckpointId) -> list[str]:
        raise NotImplementedError

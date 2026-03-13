from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

from ..contracts import SandboxRuntimeAdapter
from ..ids import CheckpointId, SandboxId
from ..models import RuntimeCapabilities, RuntimeOperationStatus

logger = logging.getLogger(__name__)


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
        detached = self._should_use_file_stdio(command)
        if detached:
            with tempfile.TemporaryFile(mode="w+") as stdout_file, tempfile.TemporaryFile(mode="w+") as stderr_file:
                completed = subprocess.run(
                    command,
                    cwd=None if cwd is None else str(cwd),
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    timeout=self._timeout_seconds,
                )
                stdout_file.seek(0)
                stderr_file.seek(0)
                stdout = stdout_file.read()
                stderr = stderr_file.read()
        else:
            completed = subprocess.run(
                command,
                cwd=None if cwd is None else str(cwd),
                check=False,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self._timeout_seconds,
            )
            stdout = "" if completed.stdout is None else completed.stdout
            stderr = "" if completed.stderr is None else completed.stderr
        return CommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=stdout,
            stderr=stderr,
        )

    def _should_use_file_stdio(self, command: list[str]) -> bool:
        if "-d" in command:
            return True
        if not command:
            return False
        executable = Path(command[0]).name
        return executable == "runc" and any(verb in command for verb in ("create", "start"))


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
        *,
        leave_running: bool,
    ) -> RuntimeOperationStatus:
        return self._execute(
            self._checkpoint_cmd(
                sandbox_id=sandbox_id,
                checkpoint_id=checkpoint_id,
                leave_running=leave_running,
            ),
            metadata=self._process_metadata(
                sandbox_id,
                checkpoint_id,
                phase="process_checkpoint",
                leave_running=leave_running,
            ),
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
        logger.debug(
            "Executing runtime command phase=%s sandbox=%s checkpoint=%s command=%s",
            metadata.get("phase"),
            metadata.get("sandbox_id"),
            metadata.get("checkpoint_id"),
            " ".join(command),
        )
        result = self._runner.run(command)
        if result.returncode != 0:
            logger.error(
                "Runtime command failed rc=%d phase=%s sandbox=%s checkpoint=%s stderr=%s",
                result.returncode,
                metadata.get("phase"),
                metadata.get("sandbox_id"),
                metadata.get("checkpoint_id"),
                result.stderr.strip(),
            )
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(command)}"
                f"\nstderr: {result.stderr.strip()}"
            )
        merged = dict(metadata)
        merged["stdout"] = result.stdout.strip()
        merged["stderr"] = result.stderr.strip()
        logger.debug(
            "Runtime command completed phase=%s sandbox=%s checkpoint=%s",
            metadata.get("phase"),
            metadata.get("sandbox_id"),
            metadata.get("checkpoint_id"),
        )
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
        leave_running: bool | None = None,
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
    def _checkpoint_cmd(
        self,
        sandbox_id: SandboxId,
        checkpoint_id: CheckpointId,
        *,
        leave_running: bool,
    ) -> list[str]:
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

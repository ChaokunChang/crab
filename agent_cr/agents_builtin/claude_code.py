"""Claude Code agent adapter for the SDK.

This is the SDK-flavored profile, distinct from the harness-internal
`integrations.agents.claude_code.ClaudeCodeAgent`. The harness version bakes
the task into the OCI bundle (one-shot, replay-friendly); this version uses
the multi-task SDK contract — `install()` ensures the binary is available,
    `execute(sbx, task)` re-execs the CLI inside the still-running sandbox.

What this adapter does NOT do that the harness version does:
  - mount a host-side Claude home dir for restore-safety
  - pin the binary to a recorded trace version
  - wire response gating against `claude_code_trace_replay`
  - inject the wrapper script that writes /work/.task_done markers

Those features are benchmark-specific. SDK users get a clean Claude Code
session per task, with no replay machinery in the way.

For users who want the harness's restore-safe Claude mount layout, the
harness path still exists at `integrations/agents/claude_code.py` and is
unchanged.
"""
from __future__ import annotations

import logging
import shlex
from typing import TYPE_CHECKING

from ..agent import Agent, TaskResult

if TYPE_CHECKING:
    from ..sandbox import Sandbox

logger = logging.getLogger(__name__)


_DEFAULT_MODEL_ENV = "ANTHROPIC_MODEL"
_DEFAULT_MODEL = "claude-opus-4-6"


class ClaudeCodeAgent(Agent):
    """SDK profile for the Claude Code CLI.

    `install()` ensures `claude` is on PATH inside the sandbox. The default
    install path is `npm install -g @anthropic-ai/claude-code`; users who
    bring an image with claude-code already installed get a fast no-op.

    `execute(sbx, task)` invokes `claude -p TASK` with `--dangerously-skip-permissions`
    (safe inside the sandbox isolation) and reads stdout + exit code into a
    `TaskResult`.
    """

    name = "claude-code"
    llm_protocol = "anthropic"
    version = "sdk-1"
    requires_network_namespace = True

    # Default to ubuntu:22.04 — users override via Sandbox(image=...). The
    # SDK doesn't bundle a custom image; install() handles binary placement.
    default_image = "ubuntu:22.04"

    def __init__(
        self,
        *,
        model: str | None = None,
        binary_path: str | None = None,
        skip_install: bool = False,
        extra_args: list[str] | None = None,
    ) -> None:
        self._model = model or _DEFAULT_MODEL
        self._binary_path = binary_path or "claude"
        self._skip_install = skip_install
        self._extra_args = list(extra_args or [])

    # ------------------------------------------------------------------
    # SDK contract
    # ------------------------------------------------------------------

    def install(self, sbx: "Sandbox") -> None:
        if self._skip_install:
            return
        # Check if claude is already on PATH.
        probe = sbx.commands.run(
            argv=["sh", "-c", f"command -v {shlex.quote(self._binary_path)} >/dev/null"],
            capture_output=True,
            check=False,
        )
        if probe.returncode == 0:
            logger.info("claude-code already installed in sandbox %s", sbx.sandbox_id)
            return
        # Install npm if needed, then claude-code. Strict failure if either
        # step fails — per design decision (5).
        install_script = (
            "set -e; "
            "if ! command -v node >/dev/null 2>&1; then "
            "  if command -v apt-get >/dev/null 2>&1; then "
            "    export DEBIAN_FRONTEND=noninteractive; "
            "    apt-get update >/dev/null && apt-get install -y --no-install-recommends nodejs npm >/dev/null; "
            "  else "
            "    echo 'claude-code install: no apt-get available; provide an image with node preinstalled' >&2; "
            "    exit 1; "
            "  fi; "
            "fi; "
            "npm install -g @anthropic-ai/claude-code"
        )
        result = sbx.commands.run(install_script, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"claude-code install failed: rc={result.returncode}\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    def execute(self, sbx: "Sandbox", task: str) -> TaskResult:
        argv = [self._binary_path, "--dangerously-skip-permissions"]
        argv.extend(["--model", self._model])
        argv.extend(self._extra_args)
        argv.extend(["-p", task])
        # The interceptor env vars are already set in the engine's thread
        # environment AND propagated by Sandbox._command_env into the
        # sandbox exec. So claude inside the sandbox will read
        # ANTHROPIC_BASE_URL from its env and route through the interceptor.
        result = sbx.commands.run(
            argv=argv,
            capture_output=True,
            check=False,
        )
        return TaskResult(
            exit_code=result.returncode,
            output=result.stdout,
            extra={"stderr": result.stderr},
        )


__all__ = ["ClaudeCodeAgent"]

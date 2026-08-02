"""iFlow SDK adapter.

This profile uses the real iFlow sandbox ingredients that the benchmark
harness already ships, but exposes them through the multi-task SDK contract.
The benchmark adapter bakes one task into the OCI bundle; this adapter keeps
the sandbox alive and invokes iFlow with `runc exec` for each `agent.run()`.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from integrations.sandboxes.iflow.harness import (
    prepare_iflow_runtime,
    prepare_iflow_state,
)
from integrations.sandboxes.runtime.image import build_image, image_exists

from ..agent import Agent, TaskResult

if TYPE_CHECKING:
    from ..engine import Engine
    from ..sandbox import Sandbox

logger = logging.getLogger(__name__)

_DEFAULT_IMAGE = "crab-iflow-bench:workspace"
_ENTRYPOINT = "/opt/iflow-runtime/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js"
_NODE = "/opt/iflow-runtime/node/bin/node"


class IFlowAgent(Agent):
    name = "iflow"
    llm_protocol = "openai"
    version = "sdk-1"
    default_image = _DEFAULT_IMAGE
    requires_network_namespace = True

    # Host-inspector filters specific to the iFlow runtime. Callers pass
    # these to `Sandbox(host_inspector_ignore_process_rules=..., ...)` so
    # the change signal isn't dominated by the agent's own bookkeeping.
    # Mirrors integrations/sandboxes/iflow/harness.py:PreparedIFlowRuntime.
    HOST_INSPECTOR_IGNORE_PROCESS_RULES: tuple[dict[str, object], ...] = (
        {
            "executable_basename": "node",
            "cmdline_contains": [_NODE, "--crab-iflow-wrapper"],
            "scope": "process_only",
        },
        {
            "executable_basename": "node",
            "cmdline_contains": [_NODE, "@iflow-ai/iflow-cli/bundle/"],
            "scope": "process_only",
        },
    )
    HOST_INSPECTOR_IGNORED_PATH_PREFIXES: tuple[str, ...] = (
        "/root/.iflow/",
        "/root/.npm/",
        "/opt/iflow-logs/",
    )

    def __init__(
        self,
        *,
        model: str | None = None,
        timeout: float | None = 900.0,
        max_session_turns: int = 32,
    ) -> None:
        self._model = model or os.environ.get("CRAB_IFLOW_MODEL_NAME", "crab-iflow-sdk")
        self._timeout = timeout
        self._max_session_turns = int(max_session_turns)

    def prepare_image(self, engine: "Engine", image: str) -> str:
        if image != _DEFAULT_IMAGE:
            return image
        if image_exists(tag=image):
            return image
        dockerfile = Path(__file__).resolve().parents[2] / "integrations" / "sandboxes" / "iflow" / "Dockerfile"
        build_image(
            tag=image,
            build_context=dockerfile.parent,
            dockerfile_path=dockerfile,
            telemetry=engine.system.telemetry,
        )
        return image

    def install(self, sbx: "Sandbox") -> None:
        rootfs = sbx._host_rootfs_path()
        state_root = sbx.engine.agent_state_root / str(sbx.sandbox_id) / "iflow"
        state_root.mkdir(parents=True, exist_ok=True)
        prepared_runtime = prepare_iflow_runtime(
            work_root=state_root,
            telemetry=sbx.engine.system.telemetry,
            sandbox_id=str(sbx.sandbox_id),
        )
        prepared_state = prepare_iflow_state(
            work_root=state_root,
            base_url=self.openai_base_url or "",
            model_name=self._model,
            max_session_turns=self._max_session_turns,
            telemetry=sbx.engine.system.telemetry,
            sandbox_id=str(sbx.sandbox_id),
        )

        runtime_dest = rootfs / "opt" / "iflow-runtime"
        if not (runtime_dest / "node" / "bin" / "node").is_file():
            if runtime_dest.exists():
                shutil.rmtree(runtime_dest)
            shutil.copytree(prepared_runtime.root, runtime_dest, symlinks=True)
        self._copy_dir(prepared_state.iflow_home, rootfs / "root" / ".iflow")
        self._copy_dir(prepared_state.npm_home, rootfs / "root" / ".npm")
        (rootfs / "opt" / "iflow-logs").mkdir(parents=True, exist_ok=True)
        logger.info("Installed iFlow SDK runtime into sandbox %s", sbx.sandbox_id)

    def execute(self, sbx: "Sandbox", task: str) -> TaskResult:
        env = self.command_env(
            {
                "PATH": "/opt/iflow-runtime/global/bin:/opt/iflow-runtime/node/bin:/usr/local/bin:/usr/bin:/bin",
                "HOME": "/root",
                "IFLOW_NON_INTERACTIVE": "true",
                "UV_USE_IO_URING": "0",
                "OPENAI_API_KEY": os.environ.get("CRAB_IFLOW_API_KEY", "sk-crab-iflow"),
            }
        )
        result = sbx.commands.run(
            argv=[_NODE, _ENTRYPOINT, "-p", task],
            cwd=sbx.process_cwd,
            env=env,
            timeout=self._timeout,
            capture_output=True,
            check=False,
        )
        return TaskResult(
            exit_code=result.returncode,
            output=result.stdout,
            extra={"stderr": result.stderr},
        )

    def _copy_dir(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination, symlinks=True)


__all__ = ["IFlowAgent"]

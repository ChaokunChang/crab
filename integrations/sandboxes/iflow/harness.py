from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
REQUIRED_CACHE_FILES = (
    "node-v22.18.0-linux-x64.tar.xz",
    "iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz",
)
IFLOW_WRAPPER_ARG = "--agent-cr-iflow-wrapper"
IFLOW_TOOL_OUTPUT_LIMIT_ENV = "AGENT_CR_IFLOW_TOOL_OUTPUT_LIMIT"
RUNTIME_MOUNT_PATH = "/opt/iflow-runtime"
IFLOW_HOME_MOUNT_PATH = "/root/.iflow"
NPM_HOME_MOUNT_PATH = "/root/.npm"
LOGS_MOUNT_PATH = "/opt/iflow-logs"
_IO_URING_SECCOMP = {
    "defaultAction": "SCMP_ACT_ALLOW",
    "architectures": ["SCMP_ARCH_X86_64", "SCMP_ARCH_X86", "SCMP_ARCH_X32"],
    "syscalls": [
        {
            "names": ["io_uring_setup", "io_uring_enter", "io_uring_register"],
            "action": "SCMP_ACT_ERRNO",
            "errnoRet": 1,
        }
    ],
}


@dataclass(frozen=True)
class PreparedIFlowRuntime:
    root: Path
    node_root: Path
    global_prefix: Path
    iflow_bin: Path
    runtime_strategy: str
    node_source: Path

    @property
    def ignore_process_rules(self) -> list[dict[str, object]]:
        return [
            {
                "executable_basename": "node",
                "cmdline_contains": [
                    f"{RUNTIME_MOUNT_PATH}/node/bin/node",
                    IFLOW_WRAPPER_ARG,
                ],
            },
            {
                "executable_basename": "node",
                "cmdline_contains": [
                    f"{RUNTIME_MOUNT_PATH}/node/bin/node",
                    "@iflow-ai/iflow-cli/bundle/",
                ],
            },
        ]

    @property
    def mounted_entrypoint(self) -> str:
        return f"{RUNTIME_MOUNT_PATH}/global/lib/node_modules/@iflow-ai/iflow-cli/bundle/entry.js"


@dataclass(frozen=True)
class PreparedIFlowState:
    root: Path
    iflow_home: Path
    npm_home: Path
    logs_dir: Path


def cache_dir_from_env() -> Path:
    return Path(os.environ.get("AGENT_CR_IFLOW_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


def required_cache_paths(cache_dir: Path | None = None) -> dict[str, Path]:
    root = cache_dir or cache_dir_from_env()
    return {name: root / name for name in REQUIRED_CACHE_FILES}


def ensure_cache_files(cache_dir: Path | None = None) -> dict[str, Path]:
    paths = required_cache_paths(cache_dir)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing iflow cache files in {cache_dir or cache_dir_from_env()}: {missing}")
    return paths


def prepare_iflow_runtime(
    *,
    work_root: Path,
    cache_dir: Path | None = None,
    alternate_node_runtime_dir: Path | None = None,
) -> PreparedIFlowRuntime:
    cache_files = ensure_cache_files(cache_dir)
    runtime_root = work_root / "iflow-runtime"
    # if runtime_root.exists():
    #     shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True, exist_ok=True)

    node_root = runtime_root / "node"
    if alternate_node_runtime_dir is None:
        with tarfile.open(cache_files["node-v22.18.0-linux-x64.tar.xz"]) as archive:
            archive.extractall(runtime_root, filter="data")
        extracted = next(runtime_root.glob("node-v22.18.0-linux-*"))
        extracted.rename(node_root)
        runtime_strategy = "mounted_cached_node22"
        node_source = cache_files["node-v22.18.0-linux-x64.tar.xz"]
    else:
        if not (alternate_node_runtime_dir / "bin" / "node").is_file():
            raise FileNotFoundError(f"alternate node runtime missing bin/node: {alternate_node_runtime_dir}")
        shutil.copytree(alternate_node_runtime_dir, node_root, symlinks=True, dirs_exist_ok=True)
        runtime_strategy = "mounted_alternate_node_runtime"
        node_source = alternate_node_runtime_dir

    global_prefix = runtime_root / "global"
    global_prefix.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{node_root / 'bin'}:{env.get('PATH', '')}",
            "HOME": str(runtime_root / "npm-home"),
            "npm_config_fund": "false",
            "npm_config_audit": "false",
            "npm_config_update_notifier": "false",
        }
    )
    Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(node_root / "bin" / "npm"),
            "install",
            "--global",
            "--prefix",
            str(global_prefix),
            str(cache_files["iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz"]),
        ],
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    iflow_bin = global_prefix / "bin" / "iflow"
    if not iflow_bin.is_file():
        raise FileNotFoundError(f"prepared runtime missing iflow binary: {iflow_bin}")
    return PreparedIFlowRuntime(
        root=runtime_root,
        node_root=node_root,
        global_prefix=global_prefix,
        iflow_bin=iflow_bin,
        runtime_strategy=runtime_strategy,
        node_source=node_source,
    )


def prepare_iflow_state(
    *,
    work_root: Path,
    base_url: str,
    model_name: str,
    max_session_turns: int | None = None,
) -> PreparedIFlowState:
    state_root = work_root / "iflow-state"
    if state_root.exists():
        shutil.rmtree(state_root)
    iflow_home = state_root / ".iflow"
    npm_home = state_root / ".npm"
    logs_dir = state_root / "logs"
    iflow_home.mkdir(parents=True, exist_ok=True)
    npm_home.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    (iflow_home / "settings.json").write_text(
        json.dumps(
            {
                "selectedAuthType": "openai-compatible",
                "apiKey": os.environ.get("AGENT_CR_IFLOW_API_KEY", "sk-agent-cr-iflow"),
                "baseUrl": base_url,
                "modelName": model_name,
                "bootAnimationShown": True,
                "disableAutoUpdate": True,
                "maxSessionTurns": (
                    int(max_session_turns)
                    if max_session_turns is not None
                    else int(os.environ.get("AGENT_CR_IFLOW_MAX_SESSION_TURNS", "32"))
                ),
                "approvalMode": "yolo",
                "mcpServers": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PreparedIFlowState(root=state_root, iflow_home=iflow_home, npm_home=npm_home, logs_dir=logs_dir)

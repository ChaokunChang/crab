from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from agent_cr.contracts import TelemetrySink
from agent_cr.telemetry import NoopTelemetrySink, start_operation

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"
REQUIRED_CACHE_FILES = (
    "node-v22.18.0-linux-x64.tar.xz",
    "iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz",
)
_PREPARED_RUNTIME_CACHE_DIRNAME = "prepared-runtimes"
_PREPARED_RUNTIME_METADATA_FILENAME = ".agent-cr-iflow-runtime.json"
IFLOW_WRAPPER_ARG = "--agent-cr-iflow-wrapper"
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
                "scope": "process_only",
            },
            {
                "executable_basename": "node",
                "cmdline_contains": [
                    f"{RUNTIME_MOUNT_PATH}/node/bin/node",
                    "@iflow-ai/iflow-cli/bundle/",
                ],
                "scope": "process_only",
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


def _prepared_runtime_cache_root(cache_dir: Path) -> Path:
    return cache_dir / _PREPARED_RUNTIME_CACHE_DIRNAME


def _prepared_runtime_metadata_path(runtime_root: Path) -> Path:
    return runtime_root / _PREPARED_RUNTIME_METADATA_FILENAME


def _prepared_runtime_entrypoint(runtime_root: Path) -> Path:
    return runtime_root / "global" / "lib" / "node_modules" / "@iflow-ai" / "iflow-cli" / "bundle" / "entry.js"


def _path_fingerprint(path: Path) -> dict[str, object]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _prepared_runtime_cache_key(
    *,
    cache_files: dict[str, Path],
    alternate_node_runtime_dir: Path | None,
) -> str:
    payload: dict[str, object] = {
        "version": 2,
        "node_archive": _path_fingerprint(cache_files["node-v22.18.0-linux-x64.tar.xz"]),
        "iflow_cli_tgz": _path_fingerprint(cache_files["iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz"]),
    }
    if alternate_node_runtime_dir is None:
        payload["alternate_node_runtime_dir"] = None
    else:
        resolved = alternate_node_runtime_dir.expanduser().resolve()
        payload["alternate_node_runtime_dir"] = {
            "path": str(resolved),
            "node": _path_fingerprint(resolved / "bin" / "node"),
            "npm": _path_fingerprint(resolved / "bin" / "npm"),
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:32]


def _prepared_runtime_attributes(
    *,
    sandbox_id: str | None,
    work_root: Path,
    cache_dir: Path,
    runtime_root: Path,
    cache_key: str,
    alternate_node_runtime_dir: Path | None,
) -> dict[str, object]:
    attributes: dict[str, object] = {
        "component": "iflow",
        "phase": "setup",
        "work_root": str(work_root),
        "cache_dir": str(cache_dir),
        "runtime_root": str(runtime_root),
        "runtime_cache_key": cache_key,
    }
    if sandbox_id is not None:
        attributes["sandbox_id"] = sandbox_id
    if alternate_node_runtime_dir is not None:
        attributes["alternate_node_runtime_dir"] = str(alternate_node_runtime_dir.expanduser().resolve())
    return attributes


def _load_prepared_runtime(runtime_root: Path) -> PreparedIFlowRuntime | None:
    metadata_path = _prepared_runtime_metadata_path(runtime_root)
    if not metadata_path.is_file():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    runtime_strategy = payload.get("runtime_strategy")
    node_source = payload.get("node_source")
    if not isinstance(runtime_strategy, str) or not runtime_strategy:
        return None
    if not isinstance(node_source, str) or not node_source:
        return None
    node_root = runtime_root / "node"
    global_prefix = runtime_root / "global"
    iflow_bin = global_prefix / "bin" / "iflow"
    entrypoint = _prepared_runtime_entrypoint(runtime_root)
    if not node_root.joinpath("bin", "node").is_file():
        return None
    if not node_root.joinpath("bin", "npm").is_file():
        return None
    if not iflow_bin.is_file():
        return None
    if not entrypoint.is_file():
        return None
    return PreparedIFlowRuntime(
        root=runtime_root,
        node_root=node_root,
        global_prefix=global_prefix,
        iflow_bin=iflow_bin,
        runtime_strategy=runtime_strategy,
        node_source=Path(node_source),
    )


def prepare_iflow_runtime(
    *,
    work_root: Path,
    cache_dir: Path | None = None,
    alternate_node_runtime_dir: Path | None = None,
    telemetry: TelemetrySink | None = None,
    sandbox_id: str | None = None,
) -> PreparedIFlowRuntime:
    sink = telemetry or NoopTelemetrySink()
    resolved_cache_dir = (cache_dir or cache_dir_from_env()).expanduser().resolve()
    cache_files = ensure_cache_files(resolved_cache_dir)
    cache_key = _prepared_runtime_cache_key(
        cache_files=cache_files,
        alternate_node_runtime_dir=alternate_node_runtime_dir,
    )
    cache_root = _prepared_runtime_cache_root(resolved_cache_dir)
    runtime_root = cache_root / cache_key
    attributes = _prepared_runtime_attributes(
        sandbox_id=sandbox_id,
        work_root=work_root,
        cache_dir=resolved_cache_dir,
        runtime_root=runtime_root,
        cache_key=cache_key,
        alternate_node_runtime_dir=alternate_node_runtime_dir,
    )
    operation = start_operation(sink, "iflow.runtime.prepare", attributes)
    cache_root.mkdir(parents=True, exist_ok=True)
    lock_path = cache_root / f"{cache_key}.lock"
    with lock_path.open("w", encoding="utf-8") as lock_fh:
        lock_wait_started = time.perf_counter()
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        lock_wait_ms = (time.perf_counter() - lock_wait_started) * 1000.0
        sink.emit_metric("iflow.runtime.cache_lock_wait_ms", lock_wait_ms, attributes)
        try:
            prepared = _load_prepared_runtime(runtime_root)
            if prepared is not None:
                sink.emit_event("iflow.runtime.cache_hit", attributes)
                operation.finish(
                    status="succeeded",
                    attributes={
                        "cache_hit": True,
                        "runtime_strategy": prepared.runtime_strategy,
                        "node_source": str(prepared.node_source),
                    },
                )
                return prepared

            sink.emit_event("iflow.runtime.cache_miss", attributes)
            if runtime_root.exists():
                shutil.rmtree(runtime_root, ignore_errors=True)
            staging_root: Path | None = None
            try:
                staging_root = Path(tempfile.mkdtemp(prefix=f"{cache_key}-", dir=cache_root))
                node_root = staging_root / "node"
                stage_node = start_operation(sink, "iflow.runtime.stage_node", attributes)
                try:
                    if alternate_node_runtime_dir is None:
                        with tarfile.open(cache_files["node-v22.18.0-linux-x64.tar.xz"]) as archive:
                            archive.extractall(staging_root, filter="data")
                        extracted = next(staging_root.glob("node-v22.18.0-linux-*"))
                        extracted.rename(node_root)
                        runtime_strategy = "shared_cached_node22"
                        node_source = cache_files["node-v22.18.0-linux-x64.tar.xz"]
                    else:
                        resolved_alternate_node = alternate_node_runtime_dir.expanduser().resolve()
                        if not (resolved_alternate_node / "bin" / "node").is_file():
                            raise FileNotFoundError(
                                f"alternate node runtime missing bin/node: {resolved_alternate_node}"
                            )
                        if not (resolved_alternate_node / "bin" / "npm").is_file():
                            raise FileNotFoundError(
                                f"alternate node runtime missing bin/npm: {resolved_alternate_node}"
                            )
                        shutil.copytree(
                            resolved_alternate_node,
                            node_root,
                            symlinks=True,
                        )
                        runtime_strategy = "shared_alternate_node_runtime"
                        node_source = resolved_alternate_node
                except Exception:
                    stage_node.finish(status="failed")
                    raise
                stage_node.finish(
                    status="succeeded",
                    attributes={
                        "runtime_strategy": runtime_strategy,
                        "node_source": str(node_source),
                    },
                )

                global_prefix = staging_root / "global"
                global_prefix.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update(
                    {
                        "PATH": f"{node_root / 'bin'}:{env.get('PATH', '')}",
                        "HOME": str(staging_root / "npm-home"),
                        "npm_config_fund": "false",
                        "npm_config_audit": "false",
                        "npm_config_update_notifier": "false",
                    }
                )
                Path(env["HOME"]).mkdir(parents=True, exist_ok=True)
                install_cli = start_operation(sink, "iflow.runtime.install_cli", attributes)
                try:
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
                    entrypoint = _prepared_runtime_entrypoint(staging_root)
                    if not entrypoint.is_file():
                        raise FileNotFoundError(f"prepared runtime missing iflow entrypoint: {entrypoint}")
                except Exception:
                    install_cli.finish(status="failed")
                    raise
                install_cli.finish(status="succeeded")

                _prepared_runtime_metadata_path(staging_root).write_text(
                    json.dumps(
                        {
                            "runtime_strategy": runtime_strategy,
                            "node_source": str(node_source),
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                staging_root.rename(runtime_root)
            except Exception:
                if staging_root is not None:
                    shutil.rmtree(staging_root, ignore_errors=True)
                operation.finish(status="failed")
                raise
            prepared = _load_prepared_runtime(runtime_root)
            if prepared is None:
                operation.finish(status="failed")
                raise FileNotFoundError(f"prepared runtime cache incomplete: {runtime_root}")
            operation.finish(
                status="succeeded",
                attributes={
                    "cache_hit": False,
                    "runtime_strategy": prepared.runtime_strategy,
                    "node_source": str(prepared.node_source),
                },
            )
            return prepared
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def prepare_iflow_state(
    *,
    work_root: Path,
    base_url: str,
    model_name: str,
    max_session_turns: int | None = None,
    telemetry: TelemetrySink | None = None,
    sandbox_id: str | None = None,
) -> PreparedIFlowState:
    sink = telemetry or NoopTelemetrySink()
    resolved_max_session_turns = (
        int(max_session_turns)
        if max_session_turns is not None
        else int(os.environ.get("AGENT_CR_IFLOW_MAX_SESSION_TURNS", "32"))
    )
    operation = start_operation(
        sink,
        "iflow.state.prepare",
        {
            "component": "iflow",
            "phase": "setup",
            "work_root": str(work_root),
            "base_url": base_url,
            "model_name": model_name,
            "max_session_turns": resolved_max_session_turns,
            **({} if sandbox_id is None else {"sandbox_id": sandbox_id}),
        },
    )
    state_root = work_root / "iflow-state"
    try:
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
                    "maxSessionTurns": resolved_max_session_turns,
                    "approvalMode": "yolo",
                    "mcpServers": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        operation.finish(status="failed")
        raise
    operation.finish(status="succeeded")
    return PreparedIFlowState(root=state_root, iflow_home=iflow_home, npm_home=npm_home, logs_dir=logs_dir)

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from dataclasses import dataclass
from pathlib import Path

from agent_cr.ids import SandboxId


DEFAULT_CACHE_DIR = Path("/root/workspace/rbenv/rbenv/iflow/cache")
REQUIRED_CACHE_FILES = (
    "node-v22.18.0-linux-x64.tar.xz",
    "iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz",
)
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


def rootfs_copy_paths(*, exported_rootfs: Path) -> list[dict[str, str]]:
    return [{"source": str(exported_rootfs), "destination": "/"}]


def prepare_iflow_runtime(
    *,
    work_root: Path,
    cache_dir: Path | None = None,
    alternate_node_runtime_dir: Path | None = None,
) -> PreparedIFlowRuntime:
    cache_files = ensure_cache_files(cache_dir)
    runtime_root = work_root / "iflow-runtime"
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
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
                "maxSessionTurns": int(os.environ.get("AGENT_CR_IFLOW_MAX_SESSION_TURNS", "32")),
                "approvalMode": "yolo",
                "mcpServers": {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return PreparedIFlowState(root=state_root, iflow_home=iflow_home, npm_home=npm_home, logs_dir=logs_dir)


def write_bundle_config(
    *,
    bundle_dir: Path,
    interceptor_port: int,
    cgroup_path: str,
    sandbox_id: SandboxId,
    task_description: str,
    prepared_runtime: PreparedIFlowRuntime,
    prepared_state: PreparedIFlowState,
    network_namespace_path: str | None = None,
    base_url: str | None = None,
) -> None:
    config_path = bundle_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    linux_cfg = cfg.get("linux", {})
    namespaces = [ns for ns in linux_cfg.get("namespaces", []) if ns.get("type") != "cgroup"]
    if network_namespace_path is not None:
        updated_namespaces: list[dict[str, str]] = []
        network_attached = False
        for namespace in namespaces:
            if namespace.get("type") == "network":
                updated_namespaces.append({"type": "network", "path": network_namespace_path})
                network_attached = True
            else:
                updated_namespaces.append(namespace)
        if not network_attached:
            updated_namespaces.append({"type": "network", "path": network_namespace_path})
        namespaces = updated_namespaces
    linux_cfg["namespaces"] = namespaces
    linux_cfg["cgroupsPath"] = cgroup_path
    if os.environ.get("AGENT_CR_IFLOW_BLOCK_IO_URING", "1") == "1":
        linux_cfg["seccomp"] = _IO_URING_SECCOMP
    else:
        linux_cfg.pop("seccomp", None)
    cfg["linux"] = linux_cfg

    cfg["process"]["terminal"] = False
    cfg["process"]["cwd"] = "/work"
    cfg["process"]["args"] = [
        "/bin/sh",
        "-lc",
        (
            "export HOME=/root; "
            "export IFLOW_NON_INTERACTIVE=true; "
            "cd /work && "
            f"exec {RUNTIME_MOUNT_PATH}/node/bin/node {prepared_runtime.mounted_entrypoint} "
            f"-p \"$AGENT_CR_IFLOW_TASK\" >{LOGS_MOUNT_PATH}/iflow.stdout 2>{LOGS_MOUNT_PATH}/iflow.stderr"
        ),
    ]
    cfg["process"]["env"] = [
        f"PATH={RUNTIME_MOUNT_PATH}/global/bin:{RUNTIME_MOUNT_PATH}/node/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONUNBUFFERED=1",
        "UV_USE_IO_URING=0",
        f"AGENT_CR_IFLOW_BASE_URL={base_url or os.environ.get('AGENT_CR_IFLOW_BASE_URL', f'http://172.17.0.1:{interceptor_port}/v1')}",
        f"AGENT_CR_IFLOW_MODEL_NAME={os.environ.get('AGENT_CR_IFLOW_MODEL_NAME', 'agent-cr-iflow-scripted')}",
        f"AGENT_CR_IFLOW_TASK={task_description}",
        "AGENT_CR_IFLOW_MAX_SESSION_TURNS=32",
        "HOME=/root",
        "IFLOW_NON_INTERACTIVE=true",
    ]
    cfg["mounts"] = [
        mount
        for mount in cfg.get("mounts", [])
        if mount.get("destination")
        not in {RUNTIME_MOUNT_PATH, IFLOW_HOME_MOUNT_PATH, NPM_HOME_MOUNT_PATH, LOGS_MOUNT_PATH}
    ]
    cfg["mounts"].extend(
        [
            {
                "destination": RUNTIME_MOUNT_PATH,
                "type": "bind",
                "source": str(prepared_runtime.root),
                "options": ["rbind", "ro"],
            },
            {
                "destination": IFLOW_HOME_MOUNT_PATH,
                "type": "bind",
                "source": str(prepared_state.iflow_home),
                "options": ["rbind", "rw"],
            },
            {
                "destination": NPM_HOME_MOUNT_PATH,
                "type": "bind",
                "source": str(prepared_state.npm_home),
                "options": ["rbind", "rw"],
            },
            {
                "destination": LOGS_MOUNT_PATH,
                "type": "bind",
                "source": str(prepared_state.logs_dir),
                "options": ["rbind", "rw"],
            },
        ]
    )
    cfg["root"]["path"] = "rootfs"
    cfg["root"]["readonly"] = False
    config_path.write_text(json.dumps(cfg, indent=2))


@dataclass
class BridgeNetworkNamespace:
    name: str
    ip_address: str
    bridge_name: str = "docker0"
    gateway: str = "172.17.0.1"

    def __post_init__(self) -> None:
        suffix = hashlib.sha1(self.name.encode("utf-8")).hexdigest()[:6]
        self.host_veth = f"vethh{suffix}"[:15]
        self.peer_veth = f"vethc{suffix}"[:15]
        self.namespace_path = f"/var/run/netns/{self.name}"

    def create(self) -> None:
        if shutil.which("ip") is None:
            raise RuntimeError("ip command is required for bridge namespace setup")
        self._run(["ip", "netns", "add", self.name])
        try:
            self._run(["ip", "link", "add", self.host_veth, "type", "veth", "peer", "name", self.peer_veth])
            self._run(["ip", "link", "set", self.host_veth, "master", self.bridge_name])
            self._run(["ip", "link", "set", self.host_veth, "up"])
            self._run(["ip", "link", "set", self.peer_veth, "netns", self.name])
            self._run(["ip", "netns", "exec", self.name, "ip", "link", "set", "lo", "up"])
            self._run(["ip", "netns", "exec", self.name, "ip", "addr", "add", f"{self.ip_address}/16", "dev", self.peer_veth])
            self._run(["ip", "netns", "exec", self.name, "ip", "link", "set", self.peer_veth, "up"])
            self._run(["ip", "netns", "exec", self.name, "ip", "route", "add", "default", "via", self.gateway])
        except Exception:
            self.cleanup()
            raise

    def cleanup(self) -> None:
        subprocess.run(["ip", "link", "del", self.host_veth], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["ip", "netns", "del", self.name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _run(self, argv: list[str]) -> None:
        subprocess.run(argv, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from crab.contracts import TelemetrySink
from crab.telemetry import NoopTelemetrySink, start_operation

DEFAULT_CLAUDE_CODE_VERSIONS_DIR = Path("/root/.local/share/claude/versions")
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "cache"

RUNTIME_MOUNT_PATH = "/opt/claude-code-runtime"
LOGS_MOUNT_PATH = "/opt/claude-code-logs"
CLAUDE_HOME_ROOT_MOUNT_PATH = "/opt/claude-code-home"
CLAUDE_HOME_MOUNT_PATH = f"{CLAUDE_HOME_ROOT_MOUNT_PATH}/.claude"
CLAUDE_CODE_WRAPPER_ARG = "--crab-claude-code-wrapper"

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
class PreparedClaudeCodeRuntime:
    root: Path
    claude_bin: Path
    runtime_strategy: str
    source_binary: Path
    resolved_version: str | None
    supports_bare_flag: bool

    @property
    def mounted_claude_bin(self) -> str:
        return f"{RUNTIME_MOUNT_PATH}/claude"

    @property
    def ignore_process_rules(self) -> list[dict[str, object]]:
        return [
            {
                "executable_basename": "claude",
                "cmdline_contains": [self.mounted_claude_bin],
            },
            # The long-lived shell wrapper only launches Claude and writes completion
            # markers. Ignoring it avoids permanent soft-dirty noise between request windows.
            {
                "cmdline_contains": [CLAUDE_CODE_WRAPPER_ARG],
            },
        ]


@dataclass(frozen=True)
class PreparedClaudeCodeState:
    root: Path
    home_root: Path
    claude_home: Path
    logs_dir: Path


def cache_dir_from_env() -> Path:
    return Path(os.environ.get("CRAB_CLAUDE_CODE_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


def _runtime_prepare_attributes(
    *,
    sandbox_id: str | None,
    work_root: Path,
    runtime_root: Path,
    requested_version: str | None,
) -> dict[str, object]:
    attributes: dict[str, object] = {
        "component": "claude_code",
        "phase": "setup",
        "work_root": str(work_root),
        "runtime_root": str(runtime_root),
    }
    if sandbox_id is not None:
        attributes["sandbox_id"] = sandbox_id
    if requested_version:
        attributes["requested_version"] = requested_version
    return attributes


def _state_prepare_attributes(
    *,
    sandbox_id: str | None,
    work_root: Path,
    state_root: Path,
    base_url: str,
    model_name: str,
) -> dict[str, object]:
    attributes: dict[str, object] = {
        "component": "claude_code",
        "phase": "setup",
        "work_root": str(work_root),
        "state_root": str(state_root),
        "base_url": base_url,
        "model_name": model_name,
    }
    if sandbox_id is not None:
        attributes["sandbox_id"] = sandbox_id
    return attributes


def _claude_versions_dir() -> Path:
    raw_value = os.environ.get("CRAB_CLAUDE_CODE_VERSIONS_DIR")
    if raw_value:
        return Path(raw_value).expanduser().resolve()
    return DEFAULT_CLAUDE_CODE_VERSIONS_DIR.expanduser().resolve()


def _requested_version_from_env() -> str | None:
    raw_value = os.environ.get("CRAB_CLAUDE_CODE_VERSION", "").strip()
    return raw_value or None


def _binary_url_template() -> str | None:
    raw_value = os.environ.get("CRAB_CLAUDE_CODE_BINARY_URL_TEMPLATE", "").strip()
    return raw_value or None


def _version_binary_path(version: str) -> Path:
    return _claude_versions_dir() / version


def _infer_version_from_path(path: Path) -> str | None:
    name = path.name.strip()
    return name or None


def _version_sort_key(path: Path) -> tuple[int, ...]:
    parts = []
    for part in path.name.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            parts.append(-1)
    return tuple(parts)


def _download_version_binary(version: str, *, destination: Path) -> None:
    url_template = _binary_url_template()
    if url_template is None:
        raise FileNotFoundError(
            f"Claude Code binary version {version} is missing at {destination}. "
            "Set CRAB_CLAUDE_CODE_BINARY_URL_TEMPLATE or preinstall the versioned binary."
        )
    url = url_template.format(version=version)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with urllib.request.urlopen(url, timeout=120.0) as response:
            with tempfile.NamedTemporaryFile(
                prefix=f".claude-{version}-",
                suffix=".tmp",
                dir=str(destination.parent),
                delete=False,
            ) as handle:
                shutil.copyfileobj(response, handle)
                temp_path = Path(handle.name)
        assert temp_path is not None
        temp_path.chmod(temp_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        os.replace(temp_path, destination)
    except Exception:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
        raise


def _resolve_fallback_binary() -> tuple[Path, str | None, str]:
    versions_dir = _claude_versions_dir()
    if versions_dir.exists():
        version_binaries = [
            path
            for path in versions_dir.iterdir()
            if path.is_file() and os.access(path, os.X_OK)
        ]
        if version_binaries:
            resolved = max(version_binaries, key=_version_sort_key)
            return resolved, _infer_version_from_path(resolved), "fallback_cached_version"
    which_claude = shutil.which("claude")
    if which_claude:
        resolved = Path(which_claude).resolve()
        if resolved.is_file():
            return resolved, _infer_version_from_path(resolved), "fallback_path_binary"
    raise FileNotFoundError(
        "Claude Code binary not found. Set CRAB_CLAUDE_CODE_BINARY, "
        "CRAB_CLAUDE_CODE_VERSION plus CRAB_CLAUDE_CODE_BINARY_URL_TEMPLATE, "
        "or preinstall a version under ~/.local/share/claude/versions/."
    )


def _resolve_claude_code_binary(*, requested_version: str | None = None) -> tuple[Path, str | None, str]:
    env_path = os.environ.get("CRAB_CLAUDE_CODE_BINARY", "").strip()
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if path.is_file():
            return path, _infer_version_from_path(path), "explicit_binary"
        raise FileNotFoundError(f"CRAB_CLAUDE_CODE_BINARY={env_path} does not exist")

    resolved_version = _requested_version_from_env() or (requested_version.strip() if requested_version else None)
    if resolved_version:
        cached_binary = _version_binary_path(resolved_version)
        if cached_binary.is_file():
            return cached_binary, resolved_version, "version_cache"
        _download_version_binary(resolved_version, destination=cached_binary)
        return cached_binary, resolved_version, "downloaded_version"

    return _resolve_fallback_binary()


def _detect_supports_bare_flag(binary_path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(binary_path), "--help"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    help_text = "\n".join(part for part in (result.stdout, result.stderr) if part)
    return "--bare" in help_text


def prepare_claude_code_runtime(
    *,
    work_root: Path,
    requested_version: str | None = None,
    telemetry: TelemetrySink | None = None,
    sandbox_id: str | None = None,
) -> PreparedClaudeCodeRuntime:
    """Prepare Claude Code runtime by copying a pinned binary into the sandbox mount."""
    sink = telemetry or NoopTelemetrySink()
    runtime_root = work_root / "claude-code-runtime"
    operation = start_operation(
        sink,
        "claude_code.runtime.prepare",
        _runtime_prepare_attributes(
            sandbox_id=sandbox_id,
            work_root=work_root,
            runtime_root=runtime_root,
            requested_version=requested_version,
        ),
    )
    try:
        runtime_root.mkdir(parents=True, exist_ok=True)
        source_binary, resolved_version, runtime_strategy = _resolve_claude_code_binary(
            requested_version=requested_version,
        )
        supports_bare_flag = _detect_supports_bare_flag(source_binary)
        dest_binary = runtime_root / "claude"
        shutil.copy2(str(source_binary), str(dest_binary))
        dest_binary.chmod(0o755)
    except Exception:
        operation.finish(status="failed")
        raise
    operation.finish(
        status="succeeded",
        attributes={
            "runtime_strategy": runtime_strategy,
            "resolved_version": resolved_version or "",
            "source_binary": str(source_binary),
            "supports_bare_flag": supports_bare_flag,
        },
    )
    return PreparedClaudeCodeRuntime(
        root=runtime_root,
        claude_bin=dest_binary,
        runtime_strategy=runtime_strategy,
        source_binary=source_binary,
        resolved_version=resolved_version,
        supports_bare_flag=supports_bare_flag,
    )


def prepare_claude_code_state(
    *,
    work_root: Path,
    base_url: str,
    model_name: str = "claude-opus-4-6",
    telemetry: TelemetrySink | None = None,
    sandbox_id: str | None = None,
) -> PreparedClaudeCodeState:
    """Prepare Claude Code state directory with settings for trace replay."""
    sink = telemetry or NoopTelemetrySink()
    _ = base_url
    state_root = work_root / "claude-code-state"
    operation = start_operation(
        sink,
        "claude_code.state.prepare",
        _state_prepare_attributes(
            sandbox_id=sandbox_id,
            work_root=work_root,
            state_root=state_root,
            base_url=base_url,
            model_name=model_name,
        ),
    )
    try:
        if state_root.exists():
            shutil.rmtree(state_root)
        # Keep the full Claude HOME on mounted host state so temporary files like
        # `.claude.json.tmp.*` stay out of checkpointed rootfs state.
        home_root = state_root / "home"
        claude_home = home_root / ".claude"
        logs_dir = state_root / "logs"
        claude_home.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        (claude_home / "settings.json").write_text(
            json.dumps(
                {
                    "model": model_name,
                    "permissions": {
                        "allow": ["Bash(*)", "Read(*)", "Write(*)", "Edit(*)", "Grep(*)", "Glob(*)"],
                        "deny": [],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        operation.finish(status="failed")
        raise
    operation.finish(status="succeeded")
    return PreparedClaudeCodeState(
        root=state_root,
        home_root=home_root,
        claude_home=claude_home,
        logs_dir=logs_dir,
    )

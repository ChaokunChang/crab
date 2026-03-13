from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProcessIdentity:
    pid: int
    executable_path: str | None
    executable_basename: str | None
    cmdline: tuple[str, ...]


@dataclass(frozen=True)
class ProcessIgnoreRule:
    executable_basename: str | None = None
    executable_path_contains: str | None = None
    cmdline_contains: tuple[str, ...] = ()

    @classmethod
    def from_raw(cls, raw: object) -> "ProcessIgnoreRule":
        if not isinstance(raw, dict):
            raise TypeError(f"ignore rule must be an object, got {type(raw).__name__}")
        executable_basename = raw.get("executable_basename")
        executable_path_contains = raw.get("executable_path_contains")
        cmdline_contains_raw = raw.get("cmdline_contains", [])
        if executable_basename is not None and not isinstance(executable_basename, str):
            raise TypeError("executable_basename must be a string")
        if executable_path_contains is not None and not isinstance(executable_path_contains, str):
            raise TypeError("executable_path_contains must be a string")
        if not isinstance(cmdline_contains_raw, list) or any(not isinstance(item, str) for item in cmdline_contains_raw):
            raise TypeError("cmdline_contains must be a list of strings")
        return cls(
            executable_basename=executable_basename,
            executable_path_contains=executable_path_contains,
            cmdline_contains=tuple(cmdline_contains_raw),
        )

    def matches(self, identity: ProcessIdentity) -> bool:
        if self.executable_basename is not None and identity.executable_basename != self.executable_basename:
            return False
        if self.executable_path_contains is not None:
            if identity.executable_path is None or self.executable_path_contains not in identity.executable_path:
                return False
        if self.cmdline_contains:
            haystack = "\0".join(identity.cmdline)
            for needle in self.cmdline_contains:
                if needle not in haystack:
                    return False
        return True

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {}
        if self.executable_basename is not None:
            payload["executable_basename"] = self.executable_basename
        if self.executable_path_contains is not None:
            payload["executable_path_contains"] = self.executable_path_contains
        if self.cmdline_contains:
            payload["cmdline_contains"] = list(self.cmdline_contains)
        return payload


def parse_process_ignore_rules(raw_rules: object | None) -> tuple[ProcessIgnoreRule, ...]:
    if raw_rules is None:
        return ()
    if not isinstance(raw_rules, list):
        raise TypeError("ignore_process_rules must be a list")
    return tuple(ProcessIgnoreRule.from_raw(item) for item in raw_rules)


def read_process_identity(pid: int) -> ProcessIdentity | None:
    if pid <= 0:
        return None
    executable_path: str | None = None
    executable_basename: str | None = None
    cmdline: tuple[str, ...] = ()
    try:
        executable_path = str(Path(f"/proc/{pid}/exe").resolve(strict=True))
        executable_basename = Path(executable_path).name
    except OSError:
        executable_path = None
        executable_basename = None
    try:
        raw_cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        cmdline = tuple(
            part.decode("utf-8", errors="replace")
            for part in raw_cmdline.split(b"\0")
            if part
        )
    except OSError:
        cmdline = ()
    if executable_path is None and not cmdline:
        return None
    if executable_basename is None and cmdline:
        executable_basename = Path(cmdline[0]).name
    return ProcessIdentity(
        pid=pid,
        executable_path=executable_path,
        executable_basename=executable_basename,
        cmdline=cmdline,
    )


def pid_matches_ignore_rules(pid: int, rules: tuple[ProcessIgnoreRule, ...]) -> bool:
    if pid <= 0 or not rules:
        return False
    identity = read_process_identity(pid)
    if identity is None:
        return False
    return any(rule.matches(identity) for rule in rules)

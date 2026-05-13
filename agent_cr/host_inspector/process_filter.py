from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

# A rule with scope="all" suppresses both the process-state-change signal
# AND filesystem events (the rule's PID is also added to the eBPF kernel
# filter so the kernel never delivers events for it). scope="process_only"
# suppresses ONLY the process-state-change signal — fs events from those
# PIDs still flow to the daemon and `_handle_fs_event`. The latter is what
# the terminus tmux integration uses for the pane bash (matched as
# basename="bash" AND ancestor="tmux"): bash's heap dirties on every
# command parse and would otherwise trip `process_changed` every turn,
# but its file writes (shell redirections) are real signal we keep
# counting. Long-running non-bash tmux descendants (e.g. background
# daemons the agent launches with `&`) are NOT matched and contribute
# to `process_changed` normally, so their in-memory state is captured
# by process checkpoints.
SCOPE_ALL = "all"
SCOPE_PROCESS_ONLY = "process_only"
_VALID_SCOPES = (SCOPE_ALL, SCOPE_PROCESS_ONLY)


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
    ancestor_executable_basename: str | None = None
    scope: str = SCOPE_ALL

    @classmethod
    def from_raw(cls, raw: object) -> "ProcessIgnoreRule":
        if not isinstance(raw, dict):
            raise TypeError(f"ignore rule must be an object, got {type(raw).__name__}")
        executable_basename = raw.get("executable_basename")
        executable_path_contains = raw.get("executable_path_contains")
        cmdline_contains_raw = raw.get("cmdline_contains", [])
        ancestor_executable_basename = raw.get("ancestor_executable_basename")
        scope = raw.get("scope", SCOPE_ALL)
        if executable_basename is not None and not isinstance(executable_basename, str):
            raise TypeError("executable_basename must be a string")
        if executable_path_contains is not None and not isinstance(executable_path_contains, str):
            raise TypeError("executable_path_contains must be a string")
        if not isinstance(cmdline_contains_raw, list) or any(not isinstance(item, str) for item in cmdline_contains_raw):
            raise TypeError("cmdline_contains must be a list of strings")
        if ancestor_executable_basename is not None and not isinstance(ancestor_executable_basename, str):
            raise TypeError("ancestor_executable_basename must be a string")
        if not isinstance(scope, str) or scope not in _VALID_SCOPES:
            raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
        return cls(
            executable_basename=executable_basename,
            executable_path_contains=executable_path_contains,
            cmdline_contains=tuple(cmdline_contains_raw),
            ancestor_executable_basename=ancestor_executable_basename,
            scope=scope,
        )

    def matches(
        self,
        identity: ProcessIdentity,
        *,
        ancestor_basenames: frozenset[str] | None = None,
    ) -> bool:
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
        if self.ancestor_executable_basename is not None:
            if ancestor_basenames is None or self.ancestor_executable_basename not in ancestor_basenames:
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
        if self.ancestor_executable_basename is not None:
            payload["ancestor_executable_basename"] = self.ancestor_executable_basename
        if self.scope != SCOPE_ALL:
            payload["scope"] = self.scope
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


def _read_ppid(pid: int) -> int | None:
    """Return the parent pid for `pid` from `/proc/{pid}/status`, or None
    if it cannot be read. We use /proc/PID/status (text, named field) over
    /proc/PID/stat to avoid the parens-in-comm parsing quirk."""
    try:
        with open(f"/proc/{pid}/status", "r") as handle:
            for line in handle:
                if line.startswith("PPid:"):
                    try:
                        return int(line.split()[1])
                    except (IndexError, ValueError):
                        return None
    except OSError:
        return None
    return None


def read_ancestor_basenames(pid: int, *, max_depth: int = 32) -> frozenset[str]:
    """Walk the parent chain from `pid` and return the set of ancestor exe
    basenames. Stops at PID 1 (init), at a depth cap, or on cycles. Used
    by `ProcessIgnoreRule.ancestor_executable_basename` matching."""
    basenames: set[str] = set()
    visited: set[int] = set()
    current = _read_ppid(pid)
    for _ in range(max_depth):
        if current is None or current <= 1 or current in visited:
            break
        visited.add(current)
        try:
            executable_path = str(Path(f"/proc/{current}/exe").resolve(strict=True))
            basenames.add(Path(executable_path).name)
        except OSError:
            # Permissions or process exited — keep walking, the chain may
            # still hit a tmux/etc ancestor higher up.
            pass
        current = _read_ppid(current)
    return frozenset(basenames)


class PidIdentityCache:
    """Process-identity cache shared across the high-rate fs-event hot path.

    Each fs event would otherwise trigger 2 syscalls
    (`/proc/PID/exe` + `/proc/PID/cmdline`) inside `read_process_identity`,
    plus a parent-chain walk in `read_ancestor_basenames` if any ignore
    rule needs ancestors. Under a tmux pane workload the host inspector
    sees ~10–40 k events/sec; that proc-read fan-out was the dominant
    cost in `_handle_fs_event` and was directly responsible for the
    `fs_monitor.sync(timeout_s=5.0)` barrier missing its budget. Caching
    the identity for the lifetime of the burst eliminates almost all of
    those syscalls without changing the observable rule semantics.

    PID-reuse handling: cached entries expire after `ttl_s`. PID reuse
    inside that window is essentially impossible on a normal Linux system
    (the kernel cycles through a 4M PID space), and the worst-case
    consequence is one or two events dropped or kept against the wrong
    rule for a freshly-recycled PID — far cheaper than the per-event
    syscall overhead. LRU-bounded so a long-lived sandbox can't grow the
    cache without limit.
    """

    def __init__(self, *, max_entries: int = 8192, ttl_s: float = 5.0) -> None:
        self._lock = Lock()
        self._max = int(max_entries)
        self._ttl_s = float(ttl_s)
        self._cache: OrderedDict[
            int, tuple[float, ProcessIdentity | None, frozenset[str] | None]
        ] = OrderedDict()

    def get(
        self, pid: int, *, needs_ancestors: bool
    ) -> tuple[ProcessIdentity | None, frozenset[str]]:
        now = time.monotonic()
        with self._lock:
            entry = self._cache.get(pid)
            if entry is not None:
                deadline, identity, ancestors = entry
                # Cache hits must satisfy the caller's ancestor requirement.
                # An entry computed without ancestors is reusable for ancestor-
                # less callers but a miss for ancestor-needing ones (so we
                # don't have to re-read /proc when the rule set has rules of
                # both kinds).
                if deadline > now and (not needs_ancestors or ancestors is not None):
                    self._cache.move_to_end(pid)
                    return identity, ancestors if ancestors is not None else frozenset()
                del self._cache[pid]
        identity = read_process_identity(pid)
        ancestors_value: frozenset[str] | None = None
        if needs_ancestors and identity is not None:
            ancestors_value = read_ancestor_basenames(pid)
        with self._lock:
            self._cache[pid] = (now + self._ttl_s, identity, ancestors_value)
            self._cache.move_to_end(pid)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)
        return identity, ancestors_value if ancestors_value is not None else frozenset()

    def invalidate(self, pid: int) -> None:
        with self._lock:
            self._cache.pop(pid, None)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


def classify_pid_against_rules(
    pid: int,
    rules: tuple[ProcessIgnoreRule, ...],
    *,
    cache: PidIdentityCache | None = None,
) -> tuple[bool, bool]:
    """Classify `pid` against `rules` in a single proc read (or zero, on
    cache hit).

    Returns (matched_any, matched_fs_eligible):
    - `matched_any`: pid matches at least one rule (any scope) — used to
      remove the pid from process tracking.
    - `matched_fs_eligible`: pid matches at least one rule with scope=
      SCOPE_ALL — used to also drop its fs events / add it to the eBPF
      kernel filter. scope=process_only matches contribute to
      `matched_any` but NOT to `matched_fs_eligible`, so file writes from
      a process-only-ignored pid still surface as filesystem changes.

    Pass `cache` to amortize `/proc/PID/...` reads across an event burst.
    """
    if pid <= 0 or not rules:
        return False, False
    needs_ancestors = any(rule.ancestor_executable_basename is not None for rule in rules)
    if cache is not None:
        identity, ancestors_set = cache.get(pid, needs_ancestors=needs_ancestors)
        ancestors = ancestors_set if needs_ancestors else None
    else:
        identity = read_process_identity(pid)
        ancestors = read_ancestor_basenames(pid) if needs_ancestors else None
    if identity is None:
        return False, False
    matched_any = False
    matched_fs = False
    for rule in rules:
        if rule.matches(identity, ancestor_basenames=ancestors):
            matched_any = True
            if rule.scope == SCOPE_ALL:
                matched_fs = True
        if matched_any and matched_fs:
            break
    return matched_any, matched_fs


def pid_matches_ignore_rules(
    pid: int,
    rules: tuple[ProcessIgnoreRule, ...],
    *,
    fs_only: bool = False,
    cache: PidIdentityCache | None = None,
) -> bool:
    """Return True iff `pid` matches at least one applicable rule.

    `fs_only=True` restricts matching to rules with scope == SCOPE_ALL, so
    that scope=process_only rules don't affect filesystem-event handling
    or the eBPF kernel filter — the whole point of that scope is to
    suppress the process-changed signal *without* dropping fs events from
    those pids.

    Pass `cache` to amortize `/proc/PID/...` reads across the hot path.
    """
    matched_any, matched_fs = classify_pid_against_rules(pid, rules, cache=cache)
    return matched_fs if fs_only else matched_any

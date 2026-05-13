from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
import urllib.request
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from integrations.sandboxes.runtime import bundle as sandbox_bundle

from .base import BaseAgent, TaskConfig, TaskDescription

logger = logging.getLogger(__name__)

_OBSERVATION_MAX_BYTES = 10000
_RESTORE_WAIT_TIMEOUT_S = 300.0
_DEFAULT_LLM_REQUEST_TIMEOUT_S = 900.0
_DEFAULT_COMMAND_DURATION_CAP_S = 60.0
_SPEC_FUTURE_DRAIN_TIMEOUT_S = 5.0
_SPEC_ROLE_DRAFT = "draft"
_SPEC_ROLE_ORACLE = "oracle"
# Upper bound on per-turn wait-for-quiescence when the pre-fork
# active-busy guard fires. Long enough to drain the slowest builds
# in the spec_friendly trace set (compcert / povray / cython
# typically clear within ~10 min once the agent stops issuing new
# keystrokes); shorter than the agent's overall task budget so a
# genuinely stuck foreground command doesn't hang the run. On
# timeout we fall through to running the oracle on the still-busy
# active — bash queues the new keystrokes behind the in-flight
# command exactly as before this guard existed, so timeout is
# always recoverable.
_SPEC_PRE_FORK_QUIESCENCE_TIMEOUT_S = 1800.0
# Fast-forward settle window: when the trace's next turn is a pure-wait
# (`keystrokes=""`, see `send_keys` line 384) and the active pane is
# already idle, we'd otherwise burn the recorded `min_timeout_sec` (capped
# to 60s) sleeping for output that won't come. Confirm "idle" with two
# `is_pane_idle()` checks and two `capture_pane()` snapshots separated by
# this short settle window — this absorbs the few ms in which a just-
# finished foreground command is still flushing its last line of output to
# the pane buffer. 0.3s is empirically enough to let bash repaint its
# prompt without inflating the per-skip overhead too far above the
# tmux RTT.
_FAST_FORWARD_SETTLE_S = 0.3
_RETRYABLE_EXEC_ERROR_FRAGMENTS = (
    "container does not exist",
    "container not running",
    "container not found",
    "unable to start container process",
    "failed to exec in container",
    "cannot allocate tty",
    # CRIU/runc fault-injection scenarios: the container is being dumped
    # / torn down / has just stopped while we tried to exec into it. All
    # should fall into the wait-for-restore retry path. Patterns are
    # narrowed to the runc-emitted forms so they don't accidentally match
    # unrelated user-command stderr.
    "cannot exec in a stopped container",
    "exec failed: unable to create new parent process",
    # `runc exec` racing a CRIU restore that has already torn down the old
    # init: libcontainer tries to enter the old init's namespaces and
    # lstats `/proc/<old-pid>/ns/<ns>`, which is gone.
    "namespace path: lstat /proc/",
    # `runc exec` racing a CRIU dump: libcontainer's init-p pipe to the
    # in-container helper is broken because the helper has been frozen.
    "write init-p: broken pipe",
    # `runc exec` while the spot preemption flow has paused the
    # container for CRIU dump. The pause window is short (single-digit
    # seconds); the wait-for-restore retry loop reissues the exec once
    # the container resumes (or the post-restore container takes its
    # place).
    "cannot exec in a paused container",
)
# tmux 3.x rejects send-keys whose serialized command exceeds the imsg
# payload buffer (~16345 bytes after framing) with "command too long".
# Pick a chunk size that leaves comfortable headroom for the command
# verb, target, and option bytes; consecutive `send-keys -l` invocations
# accumulate into the pane's input stream so chunking on byte boundaries
# is byte-exact for heredoc bodies and shell-pipeline payloads alike.
_TMUX_SEND_KEYS_CHUNK_BYTES = 8192
# Per-attempt timeout for the cancellable `tmux wait-for marker` loop in
# send_keys. Short enough that a stop request is honored within ~5 s of the
# request, long enough that quick commands (the common case) still complete
# in a single attempt without paying the runc-exec startup cost twice.
_TMUX_WAIT_FOR_POLL_INTERVAL_S = 60.0
"""How long each `tmux wait-for marker` blocking call holds before
re-arming. Each re-arm spawns a fresh tmux client via runc exec, which
generates a burst of fs events that the host inspector has to filter
before they back up the per-sandbox event-worker queue. The previous
5 s interval meant a 30 s `apt-get install` produced ~6 spurious
client spawns just to check whether the agent had requested
cancellation; at the cluster level that translated to a measurable
fraction of the host inspector's backlog. The marker fires
synchronously the moment the command's compound `tmux wait-for -S
<marker>` runs, so this interval has zero effect on the latency of
normal completion — it only bounds how long a stop_check signal
takes to land. 60 s is well below the agent's task-completion
deadline (1800 s default) and far enough above typical long-command
durations to amortize the spurious-client cost almost completely.
"""

# Tmux named keys (subset that show up in real terminal-bench traces).
# Keystrokes that match one of these names — optionally prefixed by C-/M-/S-
# modifiers — must be dispatched via `tmux send-keys` WITHOUT `-l`, so tmux
# interprets them as a key event rather than as raw bytes. With `-l` "C-c"
# would be sent as the three ASCII bytes C, -, c (verified: bash echoes
# `C-c: command not found`); without `-l` tmux delivers the actual Ctrl-C
# control byte (verified: bash shows `^C` and clears the line).
_TMUX_NAMED_KEYS = frozenset({
    "Enter", "Tab", "Space", "BSpace", "Backspace", "Escape",
    "Up", "Down", "Left", "Right",
    "PageUp", "PageDown", "PgUp", "PgDn",
    "Home", "End", "Delete", "Insert", "DC", "IC",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
})
_TMUX_KEY_MODIFIERS = frozenset({"C", "M", "S"})


def _is_tmux_key_name(payload: str) -> bool:
    """Return True iff `payload` is a single tmux key name eligible for
    key-name dispatch. Multi-key payloads (space-separated), payloads
    containing newlines/whitespace, or anything that is plain text fall
    through to the literal-byte path so the existing wait-for-marker
    machinery applies. The matched forms are:

      * exact named keys (`Enter`, `Tab`, `Up`, `F5`, ...)
      * single character (single ASCII letter/digit) preceded by one or
        more `C-`/`M-`/`S-` modifier groups (`C-c`, `C-d`, `M-x`, `C-M-x`)
      * named keys preceded by modifier groups (`C-Up`, `S-PgDn`, ...)
    """
    if not payload or any(c in payload for c in (" ", "\t", "\n", "\r")):
        return False
    if payload in _TMUX_NAMED_KEYS:
        return True
    parts = payload.split("-")
    if len(parts) < 2:
        return False
    for mod in parts[:-1]:
        if mod not in _TMUX_KEY_MODIFIERS:
            return False
    last = parts[-1]
    if not last:
        return False
    if len(last) == 1 and (last.isalpha() or last.isdigit()):
        return True
    return last in _TMUX_NAMED_KEYS


class _TmuxProtocolError(RuntimeError):
    """Raised when tmux itself rejects a command (e.g. payload exceeds the
    imsg cap, malformed argv). Distinct from preemption/restore-recoverable
    failures whose stderr matches `_RETRYABLE_EXEC_ERROR_FRAGMENTS`; the
    caller treats this as fatal and skips the wait-for-restore retry path."""

_COMPLETION_CONFIRMATION_TEMPLATE = (
    "Current terminal state:\n{terminal_state}\n\n"
    "Are you sure you want to mark the task as complete? "
    "This will trigger your solution to be graded and you won't be able to "
    'make any further corrections. If so, include "task_complete": true '
    "in your JSON response again."
)


@dataclass(frozen=True)
class _RecordedCommand:
    action_index: int
    keystrokes: str
    duration_s: float
    cwd: str
    env: dict[str, object]
    user: str | None
    timeout_s: float


# Default tmux session geometry — matches terminal-bench's TmuxSession defaults
# so traces authored against the original protocol see the same pane shape.
_TMUX_SESSION_NAME = "agent-cr-terminus"
_TMUX_PANE_X = 160
_TMUX_PANE_Y = 40
_TMUX_HISTORY_LIMIT = 50000
# Foreground process names that indicate the pane is at a shell prompt
# (i.e. no user-spawned command is currently running). Matched against
# `tmux display -p '#{pane_current_command}'`.
_QUIESCENT_FG_CMDS = frozenset({"bash", "sh", "zsh", "dash", "fish", "ash"})


@dataclass(frozen=True)
class _TmuxExecResult:
    """Stand-in for the SandboxExecResult that runtime.exec used to return
    from per-command runc-exec calls. Tmux send-keys is fire-and-forget at
    the protocol level, so we synthesize a "succeeded" result for the
    surrounding harness; the actual command's exit status lives inside the
    pane and is not directly observable from outside tmux."""
    returncode: int
    stdout: str
    stderr: str


class _TmuxSandboxSession:
    """Persistent tmux session inside a sandbox, modeled after
    terminal_bench/terminal/tmux_session.py.

    The original Terminus 2 agent runs every tool_call's keystrokes through a
    tmux pane, so shell state (cwd, exported vars, here-doc continuations,
    background jobs) survives across `cd dir` then `make` style sequences. Our
    earlier per-command `runc exec` path lost that state and broke any
    tool-heavy trace whose author assumed tmux semantics. This wrapper
    re-creates the same surface using `runtime.exec` to drive `tmux send-keys`
    inside the sandbox.

    Lifecycle is per (TerminusAgent, sandbox_id):

    * `ensure_started()` is idempotent — first call validates that tmux is
      installed (apt-get install tmux on the fly if missing), then issues
      `tmux new-session -d -s ...`.
    * `send_keys()` sends one keystrokes string and sleeps `min_timeout_sec` to
      mirror Terminus 2's per-tool_call wait hint.
    * `capture_pane()` reads the pane buffer; the agent loop doesn't depend on
      this for trace replay (the LLM responses are pre-recorded), but it is
      useful for telemetry/debug.
    """

    def __init__(
        self,
        runtime: Any,
        sandbox_id: Any,
        *,
        cwd: str,
        user: str | None,
        env: dict[str, object] | None = None,
        session_name: str = _TMUX_SESSION_NAME,
        stop_check: Callable[[], bool] | None = None,
    ) -> None:
        self._runtime = runtime
        self._sandbox_id = sandbox_id
        self._cwd = cwd or "/app"
        self._user = user
        self._env = dict(env or {})
        self._session_name = session_name
        self._lock = threading.Lock()
        self._started = False
        # Optional callable returning True when the agent has been told to
        # stop. Used by `send_keys` to break out of the wait-for loop early
        # rather than block for the full per-command budget — otherwise a
        # trace with a multi-minute final command can pin the agent past
        # the harness's `wait_for_task_completion` deadline.
        self._stop_check = stop_check

    @property
    def session_name(self) -> str:
        return self._session_name

    def mark_for_reinit(self) -> None:
        """Drop the cached `started` flag so the next `send_keys` re-runs
        `ensure_started`. Called after a CRIU restore that may have left
        the in-container tmux server gone (e.g., when recovery composed
        process state from a pre-tmux baseline checkpoint with later
        filesystem state). `ensure_started`'s `tmux has-session ... ||
        tmux new-session` form attaches to a surviving server or spawns
        a fresh one; either way the agent stops trying to talk to a
        server that isn't there."""
        with self._lock:
            self._started = False

    def _exec(
        self,
        argv: list[str],
        *,
        timeout_s: float = 30.0,
        capture_output: bool = True,
    ) -> Any:
        return self._runtime.exec(
            self._sandbox_id,
            argv,
            cwd=self._cwd,
            env=dict(self._env),
            user=self._user,
            timeout_s=timeout_s,
            capture_output=capture_output,
        )

    def ensure_started(self, *, install_timeout_s: float = 180.0) -> None:
        """Lazy initialization. Safe to call repeatedly from any thread."""
        with self._lock:
            if self._started:
                return
            # Check tmux is installed; auto-install if not — most
            # Terminal-Bench task Dockerfiles ship tmux already, but our
            # docker-compose-derived rootfs may not.
            check = self._exec(["tmux", "-V"], timeout_s=10.0)
            if int(getattr(check, "returncode", -1)) != 0:
                logger.info(
                    "tmux missing in sandbox=%s; installing via apt-get",
                    self._sandbox_id,
                )
                install = self._exec(
                    [
                        "/bin/bash",
                        "-lc",
                        "apt-get update -q && apt-get install -y -q tmux",
                    ],
                    timeout_s=install_timeout_s,
                )
                if int(getattr(install, "returncode", -1)) != 0:
                    raise RuntimeError(
                        "failed to install tmux in sandbox "
                        f"{self._sandbox_id}: rc={getattr(install, 'returncode', -1)} "
                        f"stderr={getattr(install, 'stderr', '')[:200]}"
                    )
            # Skip session-already-exists errors so a checkpoint+restore that
            # carried the tmux server back to life can keep using its session.
            self._exec(
                [
                    "/bin/bash",
                    "-lc",
                    (
                        f"tmux has-session -t {self._session_name} 2>/dev/null || "
                        f"tmux new-session -x {_TMUX_PANE_X} -y {_TMUX_PANE_Y} "
                        f"-d -s {self._session_name} \\; "
                        f"set-option -t {self._session_name} history-limit "
                        f"{_TMUX_HISTORY_LIMIT}"
                    ),
                ],
                timeout_s=15.0,
            )
            self._started = True

    def send_keys(
        self,
        keystrokes: str,
        *,
        min_timeout_sec: float = 0.0,
        max_timeout_sec: float = 600.0,
    ) -> None:
        """Mirror upstream terminal-bench's TmuxSession.send_keys with the
        Terminus2 agent's default `block=False` mode (see
        `terminal_bench/terminal/tmux_session.py:_send_non_blocking_keys`
        and `terminal_bench/agents/terminus_2/terminus_2.py:_execute_commands`):

        1. write the keystrokes to the pane via `tmux send-keys`,
        2. sleep `min_timeout_sec` (the trace's per-command wait hint), and
        3. return — no `tmux wait-for` blocking, no marker append.

        The duration hint is clamped to 60 s to mirror Terminus2's
        `_handle_llm_interaction` (`duration_sec=min(parsed_cmd.duration,
        60)`); the upstream prompt explicitly tells the agent "Never wait
        longer than 60 seconds; prefer to poll to see intermediate result
        status" and the trace's polling-via-empty-keystrokes pattern was
        recorded under exactly that contract. Honoring the cap keeps spec
        replays' per-turn cadence aligned with the original capture, so
        spec checkpoints still get scheduled at the same rate as
        nofault — instead of being parked inside a multi-minute
        `_wait_for_marker` call that drops every spec_pair_request_window
        between checkpoints.

        Earlier this method appended `; tmux wait-for -S <marker>` to the
        user's keystrokes and blocked on the marker. That was strictly
        more conservative than the upstream model and matched no real
        terminus_2 deployment — and worse, it broke spec mode for
        build-heavy traces (compcert, povray, cython): the agent stayed
        parked inside `_wait_for_marker` for the entire `make all` run,
        no LLM requests fired during that window, no checkpoints were
        scheduled, and every fork the next spec turn spawned cloned the
        stale pre-`make` checkpoint. An accepted promote then replaced
        the live container (with the build complete) with a fork
        container resumed from the stale state, rolling the build back
        ~9 min per long-command turn. Verification then found
        /tmp/CompCert/ccomp / /usr/local/bin/povray / pyknotid missing.

        End-of-task quiescence (waiting for an in-flight build to drain
        before verification) is handled separately by
        `_wait_for_tmux_quiescence`, which polls
        `#{pane_current_command}` exactly the way upstream's harness
        wraps its own quiescence check. The marker path is no longer
        used by send_keys; `_wait_for_marker`,
        `_compose_inline_trigger`, and the `_TMUX_*_POLL_*` constants
        remain in the file for callers that opt into block-true
        semantics explicitly (none today), but they are otherwise dead
        code.
        """
        self.ensure_started()
        # Match upstream Terminus2._handle_llm_interaction line 485:
        # `duration_sec=min(parsed_cmd.duration, 60)`. The trace's
        # recorded duration is the agent's pre-cap value; replay must
        # apply the same cap so per-turn wall-clock matches the capture
        # environment.
        sleep_budget = min(max(0.0, min_timeout_sec), 60.0)
        if max_timeout_sec > 0:
            sleep_budget = min(sleep_budget, max_timeout_sec)
        send_timeout_s = min(max(30.0, max_timeout_sec), 120.0)
        if not keystrokes or not keystrokes.strip():
            # Pure-wait turn (empty or whitespace-only keystrokes —
            # the trace's polling pattern, see the upstream prompt
            # template's "It is always possible to wait again ... by
            # running {keystrokes: '', duration: 10.0}"). Upstream
            # `_send_non_blocking_keys` still sleeps for the recorded
            # min_timeout_sec even when no real keys are sent; that
            # sleep is the *only* work the wait turn performs and is
            # what gives in-flight bash commands the wall-clock budget
            # they need to finish before the next observation. Skip
            # the actual `tmux send-keys` round-trip (it would just
            # send an empty payload) but keep the sleep.
            pass
        elif _is_tmux_key_name(keystrokes):
            # Tmux key-name keystrokes (e.g. 'C-c', 'Enter', 'F5'):
            # dispatch via `tmux send-keys` WITHOUT `-l` so tmux
            # interprets them as keys rather than as raw bytes.
            self._send_key_name(keystrokes, timeout_s=send_timeout_s)
        else:
            # Real shell input: hand the bytes straight to the pane and
            # let bash buffer/queue them. Chunking handles tmux's imsg
            # cap so heredocs and large pipelines fit.
            self._send_keys_payload(keystrokes, timeout_s=send_timeout_s)
        # Always sleep the (capped) min_timeout_sec — for pure-wait
        # turns this is the entire wait, for real commands it gives
        # bash a chance to start producing output before the next
        # turn's prompt is built.
        if sleep_budget > 0:
            time.sleep(sleep_budget)

    def _wait_for_marker(self, marker: str, *, wait_budget: float) -> None:
        """Block until `tmux wait-for marker` returns or `_stop_check`
        signals stop.

        The wait is split into short polls so a long-running command does
        not pin the agent past the harness's task-completion deadline. On
        stop, we also fire `tmux wait-for -S marker` to release any future
        re-arm and to short-circuit any concurrent waiter cleanly.
        """
        deadline = time.monotonic() + wait_budget
        attempts = 0
        while True:
            attempts += 1
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                logger.warning(
                    "tmux wait-for %s timed out after %.1fs sandbox=%s attempts=%d; verification may race",
                    marker,
                    wait_budget,
                    self._sandbox_id,
                    attempts,
                )
                return
            if self._stop_check is not None and self._stop_check():
                # Release the marker so any concurrent or future wait-for
                # for the same name unblocks immediately.
                try:
                    self._exec(
                        ["tmux", "wait-for", "-S", marker],
                        timeout_s=10.0,
                    )
                except Exception:
                    pass
                logger.info(
                    "tmux wait-for %s cancelled by stop_check sandbox=%s after ~%.1fs",
                    marker,
                    self._sandbox_id,
                    wait_budget - max(0.0, deadline - time.monotonic()),
                )
                return
            attempt_budget = min(_TMUX_WAIT_FOR_POLL_INTERVAL_S, remaining)
            try:
                wait_result = self._exec(
                    ["tmux", "wait-for", marker],
                    timeout_s=attempt_budget,
                )
            except subprocess.TimeoutExpired:
                # Short poll timeout — re-check stop_check and budget.
                continue
            except Exception as exc:
                logger.warning(
                    "tmux wait-for %s raised sandbox=%s err=%s",
                    marker,
                    self._sandbox_id,
                    exc.__class__.__name__,
                )
                return
            rc = int(getattr(wait_result, "returncode", -1))
            if rc == 0:
                return
            stderr_raw = getattr(wait_result, "stderr", "") or ""
            stderr_text = (
                stderr_raw if isinstance(stderr_raw, str) else stderr_raw.decode(errors="replace")
            )
            stderr_lower = stderr_text.lower()
            # Spot preemption drains in-flight runc execs (SIGTERM →
            # rc<0) before checkpoint, and the container is paused
            # during the dump (`cannot exec in a paused container`).
            # In both cases the marker hasn't fired yet; the bash
            # session is still mid-command and will execute its
            # appended `tmux wait-for -S marker` once the container
            # resumes (or once the post-restore container takes its
            # place). Retry until the budget runs out instead of
            # returning early — returning early lets the agent send
            # the next set of keystrokes while the previous command
            # is still running, which scrambles the pane.
            if rc < 0 or any(fragment in stderr_lower for fragment in _RETRYABLE_EXEC_ERROR_FRAGMENTS):
                logger.info(
                    "tmux wait-for %s transient rc=%s sandbox=%s; retrying",
                    marker,
                    rc,
                    self._sandbox_id,
                )
                time.sleep(0.2)
                continue
            logger.warning(
                "tmux wait-for %s returned rc=%s stderr=%s sandbox=%s; verification may race",
                marker,
                rc,
                stderr_text[:200],
                self._sandbox_id,
            )
            return

    def _send_keys_payload(self, payload: str, *, timeout_s: float) -> None:
        """Dispatch one logical `send-keys -l` payload, transparently
        chunking into pieces that fit under tmux's imsg payload cap.

        Empirically, tmux 3.x rejects `send-keys -l <arg>` with
        `command too long` once the serialized command (verb + target +
        option bytes + arg) exceeds ~16345 bytes. A 22.9 KB heredoc body
        from the GLM-5 traces tripped that on every replica that touched
        adaptive-rejection-sampler. Multiple consecutive `-l` sends to
        the same pane append to the pane's input stream verbatim, so a
        chunked dispatch is byte-exact for both heredocs (where the
        terminator must remain on its own line) and inline-trigger
        payloads (where bash buffers until the trailing `\\n` arrives).

        Errors are classified by stderr substring: container-runtime
        failures map to `RuntimeError` (which the caller treats as
        "sandbox preempted, wait for restore"); everything else maps to
        `_TmuxProtocolError` so the caller fails fast instead of waiting
        300 s for a restore that will never come.
        """
        if not payload:
            return
        for chunk in self._chunk_for_tmux(payload):
            send_result = self._exec(
                ["tmux", "send-keys", "-t", self._session_name, "-l", chunk],
                timeout_s=timeout_s,
            )
            rc = int(getattr(send_result, "returncode", -1))
            if rc == 0:
                continue
            stderr_raw = getattr(send_result, "stderr", "") or ""
            stderr_text = (
                stderr_raw if isinstance(stderr_raw, str) else stderr_raw.decode(errors="replace")
            )
            stderr_lc = stderr_text.lower()
            # rc<0 (Python's signal-killed convention, e.g. -SIGTERM=-15)
            # and rc=128+SIGNAL (shell convention, e.g. 143=128+15) both
            # mean the runc-exec subprocess was killed by a signal — the
            # spot preemption flow's `_drain_active_runtime_execs` does
            # exactly this to clear in-flight execs before CRIU dump.
            # No stderr accompanies signal kills; treat as transient and
            # let the caller wait for restore + retry.
            signal_killed = rc < 0 or rc in (143, 137)
            preemption = signal_killed or any(
                fragment in stderr_lc for fragment in _RETRYABLE_EXEC_ERROR_FRAGMENTS
            )
            err_cls = RuntimeError if preemption else _TmuxProtocolError
            raise err_cls(
                "tmux send-keys failed for sandbox "
                f"{self._sandbox_id}: rc={rc} stderr={stderr_text[:200]}"
            )

    def _send_key_name(self, key_name: str, *, timeout_s: float) -> None:
        """Dispatch a single tmux key-name keystroke (e.g. `C-c`, `Enter`,
        `F5`) via `tmux send-keys` WITHOUT `-l`. Errors are classified the
        same way as `_send_keys_payload` so callers see a consistent
        exception surface (preemption stderr → `RuntimeError`, anything
        else → `_TmuxProtocolError`)."""
        send_result = self._exec(
            ["tmux", "send-keys", "-t", self._session_name, key_name],
            timeout_s=timeout_s,
        )
        rc = int(getattr(send_result, "returncode", -1))
        if rc == 0:
            return
        stderr_raw = getattr(send_result, "stderr", "") or ""
        stderr_text = (
            stderr_raw if isinstance(stderr_raw, str) else stderr_raw.decode(errors="replace")
        )
        stderr_lc = stderr_text.lower()
        signal_killed = rc < 0 or rc in (143, 137)
        preemption = signal_killed or any(
            fragment in stderr_lc for fragment in _RETRYABLE_EXEC_ERROR_FRAGMENTS
        )
        err_cls = RuntimeError if preemption else _TmuxProtocolError
        raise err_cls(
            "tmux send-keys (key name) failed for sandbox "
            f"{self._sandbox_id}: rc={rc} key={key_name!r} stderr={stderr_text[:200]}"
        )

    @staticmethod
    def _chunk_for_tmux(payload: str) -> list[str]:
        """Split `payload` into UTF-8-safe chunks no larger than
        `_TMUX_SEND_KEYS_CHUNK_BYTES` bytes when encoded. The byte cap
        avoids tmux's imsg payload limit; backing off to a UTF-8 lead
        byte avoids splitting a multibyte sequence in half (which would
        decode to a different code point on the rejoin)."""
        encoded = payload.encode("utf-8")
        if len(encoded) <= _TMUX_SEND_KEYS_CHUNK_BYTES:
            return [payload]
        chunks: list[str] = []
        start = 0
        total = len(encoded)
        while start < total:
            end = min(start + _TMUX_SEND_KEYS_CHUNK_BYTES, total)
            if end < total:
                # Walk back to a UTF-8 lead byte (top bits != 10xxxxxx).
                while end > start and (encoded[end] & 0xC0) == 0x80:
                    end -= 1
                if end == start:
                    # Single multibyte char wider than the chunk size —
                    # shouldn't happen at 8 KiB, but emit it whole rather
                    # than splitting it.
                    end = min(start + _TMUX_SEND_KEYS_CHUNK_BYTES, total)
            chunks.append(encoded[start:end].decode("utf-8", errors="strict"))
            start = end
        return chunks

    @staticmethod
    def _compose_inline_trigger(keystrokes: str, marker: str) -> str | None:
        """Return a single send-keys payload that appends the wait-for
        trigger to the user's command line, or None if the keystrokes
        contain heredoc syntax (in which case appending in-line would
        break the heredoc terminator).

        Inline form: `<keystrokes_minus_trailing_newline>; tmux wait-for
        -S <marker>\\n`. Bash parses the whole compound command before
        executing, so the trigger is immune to FG-process stdin drainage.
        Requires the user's keystrokes to end with `\\n` (which Terminus 2
        bash_command tool_calls always do).
        """
        if "\n" not in keystrokes:
            return None
        if not keystrokes.endswith("\n"):
            return None
        if "<<" in keystrokes:
            # Heredoc — terminator must be alone on a line.
            return None
        body = keystrokes.rstrip("\n")
        if not body or body.isspace():
            return None
        return f"{body}; tmux wait-for -S {marker}\n"

    def capture_pane(self, *, capture_entire: bool = False) -> str:
        self.ensure_started()
        argv = ["tmux", "capture-pane", "-p", "-t", self._session_name]
        if capture_entire:
            argv[3:3] = ["-S", "-"]
        result = self._exec(argv, timeout_s=15.0)
        out = getattr(result, "stdout", "") or ""
        return out if isinstance(out, str) else out.decode(errors="replace")

    def is_pane_idle(self, *, timeout_s: float = 5.0) -> bool:
        """Single-shot variant of `wait_for_quiescence`: read the pane's
        foreground process once and return True iff it is the shell
        (build/install/etc. show as `make`, `gcc`, `cc1`, `apt`,
        `dpkg`, etc.). Used by the spec controller to decide whether
        promoting an accepted fork is safe — promoting while the
        active's bash has a long-running foreground command in flight
        rolls the active's container back to the fork's checkpoint
        state, killing the in-progress build."""
        self.ensure_started()
        try:
            result = self._exec(
                [
                    "tmux",
                    "display-message",
                    "-p",
                    "-t",
                    self._session_name,
                    "#{pane_current_command}",
                ],
                timeout_s=timeout_s,
            )
        except Exception as exc:
            logger.debug(
                "is_pane_idle query failed sandbox=%s err=%s; assuming idle",
                self._sandbox_id,
                exc,
            )
            return True
        raw = getattr(result, "stdout", "") or ""
        cmd = (raw if isinstance(raw, str) else raw.decode(errors="replace")).strip()
        return cmd in _QUIESCENT_FG_CMDS

    def wait_for_quiescence(
        self,
        *,
        timeout_s: float = 180.0,
        idle_seconds: float = 2.0,
        poll_interval_s: float = 0.5,
    ) -> bool:
        """Block until the pane's foreground process has been the shell
        for `idle_seconds` continuously.

        Trace tool_calls carry only a `duration` hint; nothing forces the
        agent to wait for the *actual* foreground command to exit before
        moving on. When the trace ends with a long build (e.g. `make
        install`), perform_task can return while tmux is still draining
        the build, and verification then races the in-progress writer.

        tmux exposes `#{pane_current_command}` — the FG process name in
        the pane's pty. While a build runs it shows `make`, `gcc`, `cc1`,
        `python`, etc.; once the shell prompt regains FG it shows the
        login shell (`bash` here). We treat the pane as quiescent after
        the FG name has been a known shell for at least `idle_seconds`,
        which absorbs brief inter-command gaps without false positives.
        """
        self.ensure_started()
        deadline = time.monotonic() + max(1.0, timeout_s)
        idle_since: float | None = None
        last_seen = ""
        while time.monotonic() < deadline:
            result = self._exec(
                [
                    "tmux",
                    "display-message",
                    "-p",
                    "-t",
                    self._session_name,
                    "#{pane_current_command}",
                ],
                timeout_s=10.0,
            )
            raw = getattr(result, "stdout", "") or ""
            cmd = (raw if isinstance(raw, str) else raw.decode(errors="replace")).strip()
            last_seen = cmd
            if cmd in _QUIESCENT_FG_CMDS:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= idle_seconds:
                    return True
            else:
                idle_since = None
            time.sleep(poll_interval_s)
        logger.debug(
            "quiescence timeout sandbox=%s last_pane_current_command=%r",
            self._sandbox_id,
            last_seen,
        )
        return False


@dataclass(frozen=True)
class _ParsedCommand:
    keystrokes: str
    duration_s: float


@dataclass
class _ParseResult:
    commands: list[_ParsedCommand] = field(default_factory=list)
    is_task_complete: bool = False
    error: str = ""
    warning: str = ""


@dataclass(frozen=True)
class _AssistantTurn:
    role: str
    pair_id: str | None
    content: str
    parsed: _ParseResult


@dataclass(frozen=True)
class _SpeculativeForkHandle:
    handle: Any | None
    created: bool = False
    reused: bool = False


def _limit_output_length(output: str, max_bytes: int = _OBSERVATION_MAX_BYTES) -> str:
    encoded = output.encode("utf-8", errors="ignore")
    if len(encoded) <= max_bytes:
        return output
    portion = max_bytes // 2
    head = encoded[:portion].decode("utf-8", errors="ignore")
    tail = encoded[-portion:].decode("utf-8", errors="ignore")
    omitted = len(encoded) - len(head.encode("utf-8")) - len(tail.encode("utf-8"))
    return (
        f"{head}\n[... output limited to {max_bytes} bytes; "
        f"{omitted} interior bytes omitted ...]\n{tail}"
    )


def _parse_terminus_response(response: str) -> _ParseResult:
    json_start = response.find("{")
    if json_start < 0:
        return _ParseResult(error="No valid JSON object found")
    brace_count = 0
    in_string = False
    escape_next = False
    json_end = -1
    for index, char in enumerate(response[json_start:], start=json_start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                json_end = index + 1
                break
    if json_end < 0:
        return _ParseResult(error="No complete JSON object found")
    body = response[json_start:json_end]
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return _ParseResult(error=f"Invalid JSON: {exc}")
    if not isinstance(payload, dict):
        return _ParseResult(error="Response must be a JSON object")
    raw_commands = payload.get("commands", [])
    if not isinstance(raw_commands, list):
        return _ParseResult(error="Field 'commands' must be an array")
    parsed_commands: list[_ParsedCommand] = []
    for index, item in enumerate(raw_commands):
        if not isinstance(item, dict):
            return _ParseResult(error=f"Command {index + 1} must be an object")
        keystrokes = item.get("keystrokes")
        if not isinstance(keystrokes, str):
            return _ParseResult(error=f"Command {index + 1} 'keystrokes' must be a string")
        duration_value = item.get("duration", 1.0)
        try:
            duration = float(duration_value)
        except (TypeError, ValueError):
            duration = 1.0
        parsed_commands.append(
            _ParsedCommand(
                keystrokes=keystrokes,
                duration_s=min(max(0.0, duration), _DEFAULT_COMMAND_DURATION_CAP_S),
            )
        )
    raw_complete = payload.get("task_complete", False)
    if isinstance(raw_complete, str):
        is_task_complete = raw_complete.strip().lower() in {"true", "1", "yes"}
    else:
        is_task_complete = bool(raw_complete)
    return _ParseResult(commands=parsed_commands, is_task_complete=is_task_complete)


class TerminusAgent(BaseAgent):
    agent_type = "terminus"

    def __init__(
        self,
        sandbox,
        task_description: TaskDescription,
        task_config: TaskConfig,
        *,
        runtime_state_root: Path | None = None,
        runtime=None,
        sandbox_manager=None,
        agent_host_dir: Path | None = None,
        llm_base_url: str | None = None,
        telemetry=None,
        speculation_controller=None,
        fast_forward_idle_waits: bool = True,
    ) -> None:
        super().__init__(
            sandbox,
            task_description,
            task_config,
            runtime_state_root=runtime_state_root,
            runtime=runtime,
            sandbox_manager=sandbox_manager,
            agent_host_dir=agent_host_dir,
            llm_base_url=llm_base_url,
            telemetry=telemetry,
        )
        self._lock = threading.Lock()
        self._state = "idle"
        self._messages: list[dict[str, str]] = []
        self._completed_actions = 0
        self._command_in_flight = False
        self._error = ""
        self._command_history: list[_RecordedCommand] = []
        self._current_command: _RecordedCommand | None = None
        self._pending_restore_trace_cursor: int | None = None
        self._pending_restore_generation = 0
        self._handled_restore_generation = 0
        self._restore_event = threading.Event()
        # Cache the latest router-observed trace cursor so poll_status keeps
        # reporting accurate progress even after the run loop ends and the
        # router HTTP query starts returning empty (e.g. wait_for_http_json
        # short-circuits to last_status once `_stop_requested` is set, which
        # used to clobber the cursor back to 0 in row finalization).
        self._last_known_trace_cursor: int = 0
        self._last_known_total_responses: int | None = None
        self._last_known_replay_is_complete: bool = False
        self._initial_prompt: str | None = None
        self._speculation_controller = speculation_controller
        self._spec_executor: ThreadPoolExecutor | None = (
            ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix=f"terminus-spec-{self.sandbox.sandbox_id}",
            )
            if self._is_spec_replay_mode()
            else None
        )
        self._active_spec_futures: set[Future[Any]] = set()
        self._spec_total_turns = 0
        self._spec_accept_count = 0
        self._spec_reject_count = 0
        self._spec_fork_create_count = 0
        self._spec_fork_reuse_count = 0
        self._spec_saved_ms = 0.0
        self._spec_penalty_ms = 0.0
        self._spec_hidden_penalty_ms = 0.0
        # Replay-cadence handling: see docs/replay-cadence-handling.md.
        # The slow-replay guards (`_SPEC_PRE_FORK_QUIESCENCE_TIMEOUT_S`,
        # `is_pane_idle` callsites) and the fast-forward path
        # (`_maybe_fast_forward_idle_wait`) both have telemetry
        # accumulators here so the report can show per-sandbox frequency
        # and overhead without re-scanning the event stream.
        self._fast_forward_idle_waits = bool(fast_forward_idle_waits)
        self._fast_forward_skip_count = 0
        self._fast_forward_saved_ms = 0.0
        self._guard_pre_fork_drain_count = 0
        self._guard_pre_fork_drain_ms = 0.0
        self._guard_pre_fork_drain_timeout_count = 0
        self._guard_post_match_drain_count = 0
        self._guard_post_match_drain_ms = 0.0
        self._guard_post_match_drain_timeout_count = 0
        # Non-spec (nofault) execution path drain: parity with the spec
        # pre-fork guard so wall-clock comparisons between nofault and
        # spec runs aren't biased by the drain mechanism only existing
        # on one side. See `_execute_commands` for the probe site.
        self._guard_non_spec_drain_count = 0
        self._guard_non_spec_drain_ms = 0.0
        self._guard_non_spec_drain_timeout_count = 0
        # The previous turn's `_finalize_speculative_fork` runs asynchronously
        # so its state_changed × 2 inspector roundtrips and the
        # release_fork → destroy_sandbox_dataset call do NOT sit on the
        # critical path. The next turn's `_ensure_speculative_fork` awaits
        # this future before reading the fork-reuse cache, so reuse
        # effectiveness is preserved (the cache decision is what release_fork
        # writes — we just don't wait for it on the way out of the current
        # turn). The await happens in parallel with the next turn's
        # draft+oracle wait, hiding the cost.
        self._pending_finalize_future: Future[Any] | None = None
        # Per-sandbox tmux sessions. Keyed by sandbox_id so that a
        # speculative fork (which has its own sandbox_id until promotion
        # swaps it back) gets its own session, while subsequent reuse of the
        # same sandbox hits the cached entry. Lazy-initialized in
        # _tmux_session_for to keep prepare_sandbox lightweight.
        self._tmux_sessions: dict[Any, _TmuxSandboxSession] = {}
        self._tmux_sessions_lock = threading.Lock()

    def survives_fault_relaunch(self) -> bool:
        return True

    def prepare_sandbox(self) -> None:
        if self.sandbox.llm_service_type not in {
            "terminus_trace_replay",
            "terminus_spec_trace_replay",
        }:
            raise RuntimeError(
                "TerminusAgent currently only supports llm_service_type in "
                "{'terminus_trace_replay', 'terminus_spec_trace_replay'}"
            )

    def configure_bundle(self) -> None:
        config_path = self.sandbox.bundle_dir / "config.json"
        if not config_path.exists():
            return
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        process = cfg.get("process")
        if not isinstance(process, dict):
            raise RuntimeError(f"unsupported process config for sandbox {self.sandbox.sandbox_id}")
        current_env = process.get("env", [])
        if not isinstance(current_env, list):
            raise RuntimeError(
                f"unsupported process env for sandbox {self.sandbox.sandbox_id}: {current_env!r}"
            )
        env_items = [str(item) for item in current_env]
        env_overrides = [
            "HOME=/root",
            "PAGER=cat",
            "MANPAGER=cat",
            "LESS=-R",
            "PIP_PROGRESS_BAR=off",
            "TQDM_DISABLE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            "DEBIAN_FRONTEND=noninteractive",
        ]
        if not any(item.startswith("PATH=") for item in env_items):
            env_overrides.insert(
                1, "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            )
        process["args"] = ["/bin/sh", "-c", "exec /bin/sleep infinity >/dev/null 2>&1"]
        process["terminal"] = False
        if not isinstance(process.get("cwd"), str) or not process["cwd"].strip():
            process["cwd"] = "/app"
        process["env"] = sandbox_bundle.merge_environment_defaults(env_items, env_overrides)
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    def rootfs_init_dirs(self) -> list[str]:
        dirs = super().rootfs_init_dirs()
        if "app" not in dirs:
            dirs.append("app")
        return dirs

    def extra_launch_metadata(self) -> dict[str, object]:
        return {
            "host_inspector_ignore_process_rules": [
                {"executable_basename": "sleep"},
                # Persistent tmux server (and any short-lived `tmux send-keys`
                # client) forks/respawns on every tool call. The rule below
                # uses scope="all" because tmux's own filesystem activity
                # (sockets in /tmp, etc.) is not signal we care about.
                {"executable_basename": "tmux"},
                # Filter ONLY the pane shell (bash whose ancestor is tmux),
                # not every descendant of tmux. The criteria in a single
                # `ProcessIgnoreRule` are AND'd in `matches()`, so this rule
                # matches a process iff its basename is `bash` AND `tmux`
                # appears anywhere on its parent chain — i.e., the long-lived
                # bash inside the tmux pane plus any short-lived
                # `bash -c "..."` subshells bash itself forks.
                #
                # Consequence of NOT having this rule:
                #   Pane bash sits in the cgroup as a long-lived process and
                #   stays in `tracked_pids`. Every command terminus sends is
                #   forked from bash, which dirties bash's heap pages as it
                #   parses the heredoc and builds argv. The soft-dirty bit
                #   on those pages flips on every turn, so
                #   `dirty_pids({bash_pid})` is non-empty every turn,
                #   `process_changed=True` fires every turn, and the
                #   scheduler promotes every checkpoint to a full
                #   process+fs CRIU dump — the per-turn-full-checkpoint
                #   regression observed after the tmux migration.
                #
                # Consequence of HAVING this rule:
                #   Bash's process-state-change signal is suppressed, so the
                #   scheduler can pick fs-only checkpoints on turns where
                #   only bash's heap dirtied. The trade-off is that bash's
                #   *in-memory* state (env vars, shell functions defined
                #   in-session, current parser state) between two FULL
                #   checkpoints is not preserved on fault recovery: bash
                #   restores from whichever full checkpoint last fired,
                #   even if later fs-only checkpoints ran. For terminus
                #   that is acceptable because the user-meaningful pane
                #   state (cwd, command output, history) is persisted to
                #   disk or is reproducible from the agent's trace.
                #
                # Why this rule is narrowed to basename=bash (and NOT the
                # broader `ancestor_executable_basename: tmux` shape used
                # previously):
                #   The prior shape silenced EVERY tmux descendant. That
                #   correctly suppressed bash's heap-dirty noise, but also
                #   dropped the process-changed signal from long-running
                #   processes the agent itself launches with `&` (e.g.
                #   `python -m http.server &`, sshd, custom daemons). Those
                #   processes hold real, non-reproducible state in memory;
                #   under the blanket rule their heap dirties were never
                #   observed, so the scheduler never promoted a checkpoint
                #   to full while they were running, and a fault recovery
                #   restored the sandbox to a process state that predated
                #   the daemon entirely while the filesystem advanced past
                #   it. Pinning the basename to `bash` keeps the
                #   bash-noise filter and lets every non-bash tmux
                #   descendant contribute to `process_changed` normally.
                #
                # `scope="process_only"` preserves fs events from bash so
                # shell redirections (`echo … > file`, heredoc `cat > f
                # <<EOF`) still surface as `filesystem_changed=True`. CRIU
                # dumps the entire pidns regardless of which PIDs the
                # inspector classifies as "interesting," so this rule
                # only affects WHEN a checkpoint fires, not WHAT it
                # contains.
                {
                    "executable_basename": "bash",
                    "ancestor_executable_basename": "tmux",
                    "scope": "process_only",
                },
                # ---- OPTIONAL build-helper noise allowlist (disabled) -----
                # Uncomment these to also silence the process-changed signal
                # for common compile/link helpers (make, gcc, ld, cc1) when
                # they run inside the tmux pane. Each rule ANDs basename
                # with ancestor=tmux, so it only matches the in-pane
                # invocation — a `make` launched outside the tmux subtree
                # (e.g. by a long-running agent daemon) still trips
                # process_changed.
                #
                # Consequence of ENABLING these rules:
                #   For build-heavy turns (apt-get build-dep, configure +
                #   make -jN) the scheduler stops promoting every poll to
                #   a full process+fs checkpoint just because gcc/ld/cc1
                #   spawned and dirtied their heap. Build helpers hold
                #   no non-reproducible state — their work is captured by
                #   the object files they emit, which the fs side already
                #   reports. So dropping their process signal trades a
                #   small recovery-window regression (in-flight compile
                #   restarts from the last full checkpoint) for a large
                #   reduction in full-checkpoint churn during builds.
                #
                # Consequence of LEAVING these rules disabled (current
                # behaviour):
                #   Long-running build helpers contribute their heap
                #   dirties to `process_changed` and force a full
                #   checkpoint at every poll while a build is in flight.
                #   On a multi-minute `make -j`, that means many extra
                #   CRIU dumps — most of them capturing throwaway gcc/ld
                #   process state. Correct, just wasteful.
                #
                # Recommendation:
                #   Leave commented until benchmarks show the build-noise
                #   overhead is meaningful for your workload. Every entry
                #   added here is a chance to silently drop real state
                #   from some future workload that happens to spawn the
                #   same basename as a long-lived helper.
                # {"executable_basename": "make", "ancestor_executable_basename": "tmux", "scope": "process_only"},
                # {"executable_basename": "gcc",  "ancestor_executable_basename": "tmux", "scope": "process_only"},
                # {"executable_basename": "ld",   "ancestor_executable_basename": "tmux", "scope": "process_only"},
                # {"executable_basename": "cc1",  "ancestor_executable_basename": "tmux", "scope": "process_only"},
            ]
        }

    def perform_task(self) -> None:
        if self.runtime is None:
            raise RuntimeError("TerminusAgent requires a runtime for sandbox command execution")
        self._stop_requested.clear()
        self._restore_event.clear()
        initial_prompt = self._resolve_initial_prompt()
        with self._lock:
            self._state = "running"
            self._messages = [{"role": "user", "content": initial_prompt}]
            self._completed_actions = 0
            self._command_in_flight = False
            self._error = ""
            self._command_history = []
            self._current_command = None
            self._pending_restore_trace_cursor = None
            self._pending_restore_generation = 0
            self._handled_restore_generation = 0
        self._set_status(self.poll_status())
        pending_completion = False
        turn_index = 0
        try:
            logger.info(
                "Starting terminus task sandbox=%s timeout_s=%.1f",
                self.sandbox.sandbox_id,
                self._llm_request_timeout_seconds(),
            )
            # Pre-start tmux BEFORE the first LLM request fires the
            # `no_previous_checkpoint` snapshot. Otherwise the very first
            # CRIU dump captures only the container's `sleep infinity`
            # init, every subsequent checkpoint is fs-only (because tmux's
            # process tree is process_only-ignored), and a fault-recovery
            # restore composes process state from the pre-tmux baseline
            # with later filesystem state — leaving the container running
            # but without a tmux server, while the agent's session cache
            # still believes tmux is up. The next `tmux send-keys` then
            # fails with `no server running`. Pre-starting here puts tmux
            # into the first checkpoint's process image so recovery can
            # actually restore it.
            try:
                self._tmux_session_for(self.sandbox.sandbox_id).ensure_started()
            except Exception:
                logger.warning(
                    "Pre-start tmux failed for sandbox=%s; falling back to lazy ensure_started",
                    self.sandbox.sandbox_id,
                    exc_info=True,
                )
            while not self._stop_requested.is_set():
                turn_index += 1
                self._drain_pending_restore_replays()
                if self._replay_trace_is_complete():
                    logger.info(
                        "Terminus trace exhausted; ending replay sandbox=%s actions=%d turns=%d",
                        self.sandbox.sandbox_id,
                        self._completed_actions,
                        turn_index - 1,
                    )
                    self._wait_for_tmux_quiescence()
                    with self._lock:
                        self._state = "finished"
                    self._set_status(self.poll_status())
                    return
                if self._is_spec_replay_mode():
                    assistant_turn, observation_text = self._spec_perform_step()
                else:
                    assistant_turn = self._request_assistant_turn()
                    with self._lock:
                        self._messages.append(
                            {"role": "assistant", "content": assistant_turn.content}
                        )
                    observation_text = None
                parse_result = assistant_turn.parsed
                if parse_result.error:
                    feedback_prompt = (
                        f"Previous response had parsing errors:\nERROR: {parse_result.error}\n\n"
                        "Please fix these issues and provide a proper JSON response."
                    )
                    with self._lock:
                        self._messages.append({"role": "user", "content": feedback_prompt})
                    pending_completion = False
                    payload = self.poll_status()
                    self._record_activity(payload)
                    continue
                if observation_text is None:
                    observation_text = self._execute_commands(parse_result.commands)
                if parse_result.is_task_complete:
                    if pending_completion:
                        self._wait_for_tmux_quiescence()
                        with self._lock:
                            self._state = "finished"
                        payload = self.poll_status()
                        self._set_status(payload)
                        logger.info(
                            "Completed terminus task sandbox=%s actions=%d turns=%d",
                            self.sandbox.sandbox_id,
                            int(payload.get("total_actions", 0)),
                            turn_index,
                        )
                        return
                    pending_completion = True
                    confirmation_prompt = _COMPLETION_CONFIRMATION_TEMPLATE.format(
                        terminal_state=_limit_output_length(observation_text)
                    )
                    with self._lock:
                        self._messages.append({"role": "user", "content": confirmation_prompt})
                    payload = self.poll_status()
                    self._record_activity(payload)
                    continue
                pending_completion = False
                next_prompt = _limit_output_length(observation_text)
                with self._lock:
                    self._messages.append({"role": "user", "content": next_prompt})
                payload = self.poll_status()
                self._record_activity(payload)
            self._wait_for_tmux_quiescence()
            with self._lock:
                if self._state == "running":
                    self._state = "finished"
        except Exception as exc:
            with self._lock:
                self._state = "failed"
                self._error = str(exc) or exc.__class__.__name__
            self._set_status(self.poll_status())
            logger.error("Terminus task failed sandbox=%s error=%s", self.sandbox.sandbox_id, exc)
            raise
        finally:
            self.post_task_finish()

    def post_task_finish(self) -> None:
        if self._spec_executor is not None:
            with self._lock:
                active = list(self._active_spec_futures)
            if active:
                wait(active, timeout=_SPEC_FUTURE_DRAIN_TIMEOUT_S)
            self._spec_executor.shutdown(wait=False, cancel_futures=True)
            self._spec_executor = None
        if self._is_compose_replay_mode():
            return
        super().post_task_finish()

    def poll_status(self) -> dict[str, object]:
        with self._lock:
            state = self._state
            completed_actions = self._completed_actions
            command_in_flight = self._command_in_flight
            error = self._error
            spec_total_turns = self._spec_total_turns
            spec_accept_count = self._spec_accept_count
            spec_reject_count = self._spec_reject_count
            spec_fork_create_count = self._spec_fork_create_count
            spec_fork_reuse_count = self._spec_fork_reuse_count
            spec_saved_ms = self._spec_saved_ms
            spec_penalty_ms = self._spec_penalty_ms
            spec_hidden_penalty_ms = self._spec_hidden_penalty_ms
            fast_forward_skip_count = self._fast_forward_skip_count
            fast_forward_saved_ms = self._fast_forward_saved_ms
            guard_pre_fork_drain_count = self._guard_pre_fork_drain_count
            guard_pre_fork_drain_ms = self._guard_pre_fork_drain_ms
            guard_pre_fork_drain_timeouts = self._guard_pre_fork_drain_timeout_count
            guard_post_match_drain_count = self._guard_post_match_drain_count
            guard_post_match_drain_ms = self._guard_post_match_drain_ms
            guard_post_match_drain_timeouts = self._guard_post_match_drain_timeout_count
            guard_non_spec_drain_count = self._guard_non_spec_drain_count
            guard_non_spec_drain_ms = self._guard_non_spec_drain_ms
            guard_non_spec_drain_timeouts = self._guard_non_spec_drain_timeout_count
        replay_state = self._replay_router_state()
        total_responses = replay_state.get("total_responses")
        # `replay_trace_cursor` must measure progress in the same units as
        # `trace_response_count` (assistant turns consumed). Each terminus turn
        # carries 1-N shell commands, so reporting `completed_actions` here
        # would let the harness's short-circuit (replay_trace_cursor >=
        # trace_response_count) fire mid-trace as soon as commands outpaced
        # turns. Pull the LLM service's own turn cursor instead — same field
        # iflow/claude_code already use.
        try:
            raw_trace_cursor = max(0, int(replay_state.get("trace_cursor", 0)))
        except (TypeError, ValueError):
            raw_trace_cursor = 0
        # Once the router HTTP query starts returning {} (after _stop_requested
        # is set, or transient HTTP failures), `raw_trace_cursor` collapses to
        # 0 and finalize_replay_row would mis-report completed runs as
        # "cursor=0 of N, replay_is_complete=False". Pin to the highest cursor
        # we have ever observed so progress stays monotonic.
        with self._lock:
            if raw_trace_cursor >= self._last_known_trace_cursor:
                self._last_known_trace_cursor = raw_trace_cursor
            else:
                raw_trace_cursor = self._last_known_trace_cursor
            if isinstance(total_responses, int) and total_responses > 0:
                self._last_known_total_responses = total_responses
            elif self._last_known_total_responses is not None:
                total_responses = self._last_known_total_responses
        trace_cursor = raw_trace_cursor
        replay_is_complete = state == "finished" or bool(replay_state.get("is_complete", False))
        if isinstance(total_responses, int) and total_responses > 0:
            replay_is_complete = replay_is_complete or trace_cursor >= total_responses
        with self._lock:
            if replay_is_complete:
                self._last_known_replay_is_complete = True
            elif self._last_known_replay_is_complete:
                replay_is_complete = True
        spec_total_decisions = spec_accept_count + spec_reject_count
        payload = {
            "agent_type": self.agent_type,
            "state": state,
            "total_actions": completed_actions,
            "filesystem_actions": 0,
            "process_actions": 0,
            "network_actions": 0,
            "stateful_actions": completed_actions,
            "replay_trace_cursor": trace_cursor,
            "replay_total_responses": total_responses,
            "replay_is_complete": replay_is_complete,
            "command_in_flight": command_in_flight,
            "spec_total_turns": spec_total_turns,
            "spec_accept_count": spec_accept_count,
            "spec_reject_count": spec_reject_count,
            "spec_accept_rate": (
                0.0 if spec_total_decisions <= 0 else spec_accept_count / spec_total_decisions
            ),
            "spec_fork_create_count": spec_fork_create_count,
            "spec_fork_reuse_count": spec_fork_reuse_count,
            "spec_saved_ms": spec_saved_ms,
            "spec_penalty_ms": spec_penalty_ms,
            "spec_hidden_penalty_ms": spec_hidden_penalty_ms,
            "spec_net_gain_ms": spec_saved_ms - spec_penalty_ms,
            "fast_forward_skip_count": fast_forward_skip_count,
            "fast_forward_saved_ms": fast_forward_saved_ms,
            "guard_pre_fork_drain_count": guard_pre_fork_drain_count,
            "guard_pre_fork_drain_ms": guard_pre_fork_drain_ms,
            "guard_pre_fork_drain_timeouts": guard_pre_fork_drain_timeouts,
            "guard_post_match_drain_count": guard_post_match_drain_count,
            "guard_post_match_drain_ms": guard_post_match_drain_ms,
            "guard_post_match_drain_timeouts": guard_post_match_drain_timeouts,
            "guard_non_spec_drain_count": guard_non_spec_drain_count,
            "guard_non_spec_drain_ms": guard_non_spec_drain_ms,
            "guard_non_spec_drain_timeouts": guard_non_spec_drain_timeouts,
        }
        if error:
            payload["error"] = error
        return payload

    def wait_for_progress(self, *, minimum_actions: int) -> dict[str, object]:
        self._wait_for_action_count(minimum_actions)
        payload = self.poll_status()
        self._record_activity(payload)
        return payload

    def wait_for_action_delta(self, *, delta: int) -> dict[str, object]:
        baseline = int(self.sandbox.last_status.get("total_actions", 0))
        self._wait_for_action_count(baseline + delta)
        payload = self.poll_status()
        self._record_activity(payload)
        return payload

    def wait_for_task_ready(self) -> None:
        self._set_status(self.poll_status())

    def on_restore_complete(self) -> None:
        # Defensive: invalidate every cached tmux session so the next
        # `send_keys` re-runs `ensure_started`. The pre-task `ensure_started`
        # in `perform_task` puts tmux into the first checkpoint's process
        # image, but a degenerate compose of process+fs (e.g., process
        # from pre-tmux baseline if the very first checkpoint won the race
        # against pre-start, or runtime relaunch via `relaunch_handler`)
        # could still leave the restored container without tmux. Resetting
        # the cache here closes that window — `tmux has-session ... || tmux
        # new-session` either attaches to the surviving server or spawns a
        # fresh one before drain_pending_restore_replays starts replaying
        # commands.
        with self._tmux_sessions_lock:
            sessions = list(self._tmux_sessions.values())
        for session in sessions:
            try:
                session.mark_for_reinit()
            except Exception:
                logger.debug(
                    "Failed to mark tmux session for reinit sandbox=%s",
                    self.sandbox.sandbox_id,
                    exc_info=True,
                )
        self._restore_event.set()

    def record_restore_trace_cursor(self, trace_cursor: int) -> None:
        new_cursor = max(0, int(trace_cursor))
        with self._lock:
            self._pending_restore_trace_cursor = new_cursor
            self._pending_restore_generation += 1
            # Restore can roll the router cursor backwards (e.g. fault
            # recovery). The monotonic cursor cache must follow that
            # rewind, otherwise poll_status keeps reporting the
            # pre-restore high-water mark and finalize_replay_row
            # mis-reports replay as complete after a partial recovery.
            if new_cursor < self._last_known_trace_cursor:
                self._last_known_trace_cursor = new_cursor
                self._last_known_replay_is_complete = False

    # ── Internal helpers ──────────────────────────────────────────────

    def _is_compose_replay_mode(self) -> bool:
        return self.sandbox.launch_source == "compose" and self.sandbox.llm_service_type in {
            "terminus_trace_replay",
            "terminus_spec_trace_replay",
        }

    def _is_spec_replay_mode(self) -> bool:
        return self.sandbox.llm_service_type == "terminus_spec_trace_replay"

    def _resolve_initial_prompt(self) -> str:
        if self._initial_prompt is not None:
            return self._initial_prompt
        replay_state = self._replay_router_state()
        prompt = replay_state.get("initial_user_prompt")
        if isinstance(prompt, str) and prompt.strip():
            self._initial_prompt = prompt
            return prompt
        # Fallback: synthesize a minimal initial prompt from the task description.
        instruction = self.task_description.prompt
        self._initial_prompt = (
            f"Task Description:\n{instruction}\n\n"
            "Current terminal state:\n"
            "Current Terminal Screen:\n"
        )
        return self._initial_prompt

    def _request_assistant_turn(
        self,
        *,
        role: str | None = None,
        pair_id: str | None = None,
    ) -> _AssistantTurn:
        if not self.llm_base_url:
            raise RuntimeError(f"missing llm base url for sandbox {self.sandbox.sandbox_id}")
        with self._lock:
            messages = list(self._messages)
        request_payload = {
            "model": "agent-cr-terminus-trace-replay",
            "messages": messages,
        }
        headers = {
            "Content-Type": "application/json",
            "X-Agent-Sandbox-Id": str(self.sandbox.sandbox_id),
        }
        if pair_id:
            headers["X-AgentCR-Spec-Pair-Id"] = pair_id
        if role:
            headers["X-AgentCR-Spec-Role"] = role
        request = urllib.request.Request(
            f"{self.llm_base_url}/chat/completions",
            data=json.dumps(request_payload).encode("utf-8"),
            method="POST",
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=self._llm_request_timeout_seconds()) as response:
            payload = json.loads(response.read().decode("utf-8"))
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("terminus llm replay returned no choices")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise RuntimeError("terminus llm replay returned a malformed assistant message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("terminus llm replay returned empty assistant content")
        parsed = _parse_terminus_response(content)
        return _AssistantTurn(
            role=_SPEC_ROLE_ORACLE if role is None else role,
            pair_id=pair_id,
            content=content,
            parsed=parsed,
        )

    def _spec_perform_step(self) -> tuple[_AssistantTurn, str]:
        """Speculative execution loop: race draft+oracle, execute draft in fork, accept on match.

        Returns (oracle_assistant_turn, observation_text). The agent loop's main
        message-history bookkeeping happens here too — the oracle's content is
        what gets appended to `self._messages`, and `self._completed_actions` /
        `self._command_history` are advanced as commands are recorded.

        This is also the source of the per-turn speculative-execution telemetry
        (`spec.turn.finish` event + `benchmark.spec.*` metrics) that the
        telemetry-report's speculative-execution section consumes — it tracks
        oracle-wait, draft-exec-duration, and post-oracle drain windows so
        accepted turns can attribute saved time and rejected turns can
        attribute visible vs hidden penalty.
        """
        pair_id = uuid.uuid4().hex
        draft_future = self._submit_spec_future(
            self._request_assistant_turn, role=_SPEC_ROLE_DRAFT, pair_id=pair_id
        )
        oracle_future = self._submit_spec_future(
            self._request_assistant_turn, role=_SPEC_ROLE_ORACLE, pair_id=pair_id
        )
        done, _ = wait([draft_future, oracle_future], return_when=FIRST_COMPLETED)
        if oracle_future in done:
            try:
                oracle_turn = oracle_future.result()
            except Exception:
                wait([draft_future], timeout=_SPEC_FUTURE_DRAIN_TIMEOUT_S)
                raise
            self._invalidate_speculative_fork()
            with self._lock:
                self._messages.append({"role": "assistant", "content": oracle_turn.content})
            observation = self._execute_commands(oracle_turn.parsed.commands)
            self._record_spec_metrics(accepted=False, pair_id=pair_id, draft_first=False)
            return oracle_turn, observation

        try:
            draft_turn = draft_future.result()
        except Exception:
            oracle_turn = oracle_future.result()
            self._invalidate_speculative_fork()
            with self._lock:
                self._messages.append({"role": "assistant", "content": oracle_turn.content})
            observation = self._execute_commands(oracle_turn.parsed.commands)
            self._record_spec_metrics(accepted=False, pair_id=pair_id, draft_first=True)
            return oracle_turn, observation

        # Fast path for "no-shell-work" turns (mark_task_complete and
        # the agent's polling-while-waiting turns where the trace's
        # bash_command keystrokes are empty/whitespace — `send_keys`
        # treats those as a sleep, not a command). `_replace_first_command`
        # is a no-op when the first command keystrokes is empty (or even
        # when the commands list is empty), so an empty/whitespace draft
        # implies the oracle has nothing for the active to do. Skip fork
        # creation entirely and let the active sandbox's bash session
        # keep running whatever long-flight command (e.g. a `make all`
        # from a prior turn) it was paused on. Promoting an empty-turn
        # fork would replace the active runtime container and kill that
        # in-flight command — for build-heavy traces with many wait
        # turns, doing this once per wait re-rewinds the build by the
        # LLM-wait window every iteration and the build never finishes
        # (verification then misses /tmp/CompCert/ccomp,
        # /usr/local/bin/povray, etc.).
        if all(not cmd.keystrokes.strip() for cmd in draft_turn.parsed.commands):
            oracle_turn = oracle_future.result()
            self._invalidate_speculative_fork()
            with self._lock:
                self._messages.append({"role": "assistant", "content": oracle_turn.content})
            observation = self._execute_commands(oracle_turn.parsed.commands)
            self._record_spec_metrics(accepted=False, pair_id=pair_id, draft_first=True)
            return oracle_turn, observation

        # Pre-fork active-busy guard. If the active sandbox's bash pane
        # is mid-foreground-command (typical of build-heavy traces:
        # `make all` from turn N is still grinding when turns N+1..M
        # arrive), creating a fork now and later promoting it would
        # roll the active's runtime container back to the fork's
        # checkpoint state — the fork was restored from a frozen
        # mid-build moment and will be strictly behind the active's
        # current build progress at promotion time. Detect via tmux's
        # `#{pane_current_command}` (`is_pane_idle()` returns True iff
        # the FG process is the shell itself).
        #
        # Bypassing fork creation here saves the CRIU restore + the
        # runc-exec storm of `_execute_records_for_sandbox` on the
        # fork — both pure waste because the post-match active-busy
        # guard would have killed the fork at promotion anyway.
        # Instead, wait for the active's pane to drain (so the next
        # oracle command runs on a clean prompt instead of queueing
        # behind the in-flight build, which on a long build cascades
        # into every subsequent turn also queuing), then execute the
        # oracle on the now-idle active. wait_for_quiescence has its
        # own timeout; on timeout we still run the oracle, which just
        # queues behind the still-running foreground — same behavior
        # as before this guard.
        active_session = self._tmux_session_for(self.sandbox.sandbox_id)
        if not active_session.is_pane_idle():
            busy_cmd = self._observed_pane_command(active_session)
            drain_started = time.perf_counter()
            drain_resolved = False
            try:
                drain_resolved = bool(
                    active_session.wait_for_quiescence(
                        timeout_s=_SPEC_PRE_FORK_QUIESCENCE_TIMEOUT_S
                    )
                )
            except Exception:
                logger.debug(
                    "wait_for_quiescence raised in pre-fork active-busy guard sandbox=%s",
                    self.sandbox.sandbox_id,
                    exc_info=True,
                )
            drain_wait_ms = (time.perf_counter() - drain_started) * 1000.0
            self._record_drain_event(
                site="pre_fork",
                busy_cmd=busy_cmd,
                drain_wait_ms=drain_wait_ms,
                drain_resolved=drain_resolved,
            )
            oracle_turn = oracle_future.result()
            self._invalidate_speculative_fork()
            with self._lock:
                self._messages.append({"role": "assistant", "content": oracle_turn.content})
            observation = self._execute_commands(oracle_turn.parsed.commands)
            self._record_spec_metrics(accepted=False, pair_id=pair_id, draft_first=True)
            return oracle_turn, observation

        fork_handle = _SpeculativeForkHandle(handle=None)
        fork_restore_started = time.perf_counter()
        if self._speculation_controller is not None and self._spec_executor is not None:
            fork_future = self._submit_spec_future(self._ensure_speculative_fork)
            wait([fork_future, oracle_future], return_when=FIRST_COMPLETED)
            if oracle_future.done() and not fork_future.done():
                oracle_turn = oracle_future.result()
                self._invalidate_speculative_fork()
                with self._lock:
                    self._messages.append({"role": "assistant", "content": oracle_turn.content})
                observation = self._execute_commands(oracle_turn.parsed.commands)
                self._record_spec_metrics(accepted=False, pair_id=pair_id, draft_first=True)
                return oracle_turn, observation
            if fork_future.done():
                fork_handle = fork_future.result()
        fork_restore_ms = (time.perf_counter() - fork_restore_started) * 1000.0

        if fork_handle.handle is None:
            oracle_turn = oracle_future.result()
            self._invalidate_speculative_fork()
            with self._lock:
                self._messages.append({"role": "assistant", "content": oracle_turn.content})
            observation = self._execute_commands(oracle_turn.parsed.commands)
            self._record_spec_metrics(accepted=False, pair_id=pair_id, draft_first=True)
            return oracle_turn, observation

        # Speculatively execute the draft batch in the fork. We don't await the
        # exec here — the oracle race result decides whether we promote it.
        # Capture exec timing via a done callback so we can compute the actual
        # overlap window even when oracle wins the race after the fork already
        # finished.
        spec_records = [self._build_command_record(c, action_index=0) for c in draft_turn.parsed.commands]
        spec_started = time.perf_counter()
        draft_exec_done_at: list[float] = []
        spec_exec_future = self._submit_spec_future(
            self._execute_records_for_sandbox,
            spec_records,
            fork_handle.handle.sandbox_id,
        )

        def _record_draft_exec_done(_completed: Future[Any]) -> None:
            draft_exec_done_at.append(time.perf_counter())

        spec_exec_future.add_done_callback(_record_draft_exec_done)

        oracle_turn = oracle_future.result()
        oracle_ready_at = time.perf_counter()
        commands_match = self._batches_match(draft_turn.parsed, oracle_turn.parsed)

        if commands_match:
            # No-op match (mark_task_complete or pure-wait turns where both
            # draft and oracle have zero shell commands): promoting the
            # fork would replace the active runtime container with the
            # fork's restored container. If the active's bash pane is
            # running an in-flight command from a previous turn — common
            # for build-heavy traces where the agent issues `make all`
            # then sleeps through several wait turns until the build
            # finishes — that command is killed by the active runtime
            # delete during release_fork, and the fork's bash resumes
            # from the (older) checkpoint state. Repeating this once per
            # wait turn rewinds the build by the LLM-wait window every
            # iteration; for long enough builds make never reaches
            # `make install`, leaving artifacts like /tmp/CompCert/ccomp
            # missing at verification. Skip promotion for empty turns:
            # active stays alive, its in-flight command keeps running,
            # and the fork is destroyed unceremoniously.
            if all(not cmd.keystrokes.strip() for cmd in oracle_turn.parsed.commands):
                self._cancel_fork_runtime(fork_handle.handle)
                with self._lock:
                    self._messages.append({"role": "assistant", "content": oracle_turn.content})
                self._finalize_speculative_fork(
                    accepted=False, fork=fork_handle.handle, reuse_candidate=False, pair_id=pair_id,
                )
                wait([spec_exec_future], timeout=_SPEC_FUTURE_DRAIN_TIMEOUT_S)
                speculative_exec_ms = (time.perf_counter() - spec_started) * 1000.0
                self._record_spec_metrics(
                    accepted=False,
                    pair_id=pair_id,
                    draft_first=True,
                    fork_created=fork_handle.created,
                    fork_reused=fork_handle.reused,
                    fork_restore_ms=fork_restore_ms,
                    speculative_exec_ms=speculative_exec_ms,
                    reuse_candidate=False,
                )
                return oracle_turn, ""
            try:
                spec_results = spec_exec_future.result()
            except Exception:
                self._finalize_speculative_fork(
                    accepted=False, fork=fork_handle.handle, reuse_candidate=False, pair_id=pair_id,
                )
                with self._lock:
                    self._messages.append({"role": "assistant", "content": oracle_turn.content})
                observation = self._execute_commands(oracle_turn.parsed.commands)
                speculative_exec_ms = (time.perf_counter() - spec_started) * 1000.0
                self._record_spec_metrics(
                    accepted=False,
                    pair_id=pair_id,
                    draft_first=True,
                    fork_created=fork_handle.created,
                    fork_reused=fork_handle.reused,
                    fork_restore_ms=fork_restore_ms,
                    speculative_exec_ms=speculative_exec_ms,
                    penalty_ms=speculative_exec_ms,
                    hidden_penalty_ms=speculative_exec_ms,
                    reuse_candidate=False,
                )
                return oracle_turn, observation
            # Promotion-safety safety net for the narrow race window
            # where the pre-fork active-busy guard saw an idle pane but
            # the active became busy by the time we reach this point
            # (e.g. a queued bash command from a prior turn started
            # executing during the LLM round-trip). The pre-fork guard
            # is the primary line of defense — it skips fork creation
            # entirely when the active is busy. Anything reaching here
            # already passed that check, so this branch is rare; we
            # keep it because promoting a fork onto a busy active
            # destroys the in-flight foreground command (replacing the
            # runc container kills the make/gcc/ld tree mid-run, and
            # the fork's resumed make is necessarily behind the
            # active's actual progress).
            #
            # Same recovery shape as the pre-fork guard: wait for the
            # active's pane to drain (so the oracle commands run on a
            # clean prompt instead of queuing behind whatever the
            # fork race surfaced), then run oracle on the active. On
            # quiescence-wait timeout we fall through to plain
            # _execute_commands which queues — same behavior as before
            # the wait was added.
            active_pane_idle = True
            if self._speculation_controller is not None:
                try:
                    active_session = self._tmux_session_for(self.sandbox.sandbox_id)
                    active_pane_idle = active_session.is_pane_idle()
                except Exception:
                    active_pane_idle = True
            if not active_pane_idle:
                self._cancel_fork_runtime(fork_handle.handle)
                busy_cmd = self._observed_pane_command(active_session)
                drain_started = time.perf_counter()
                drain_resolved = False
                try:
                    drain_resolved = bool(
                        active_session.wait_for_quiescence(
                            timeout_s=_SPEC_PRE_FORK_QUIESCENCE_TIMEOUT_S
                        )
                    )
                except Exception:
                    logger.debug(
                        "wait_for_quiescence raised in post-match active-busy guard sandbox=%s",
                        self.sandbox.sandbox_id,
                        exc_info=True,
                    )
                drain_wait_ms = (time.perf_counter() - drain_started) * 1000.0
                self._record_drain_event(
                    site="post_match",
                    busy_cmd=busy_cmd,
                    drain_wait_ms=drain_wait_ms,
                    drain_resolved=drain_resolved,
                )
                with self._lock:
                    self._messages.append({"role": "assistant", "content": oracle_turn.content})
                observation = self._execute_commands(oracle_turn.parsed.commands)
                wait([spec_exec_future], timeout=_SPEC_FUTURE_DRAIN_TIMEOUT_S)
                self._finalize_speculative_fork(
                    accepted=False, fork=fork_handle.handle, reuse_candidate=False, pair_id=pair_id,
                )
                speculative_exec_ms = (time.perf_counter() - spec_started) * 1000.0
                draft_exec_done_time = (
                    draft_exec_done_at[0] if draft_exec_done_at else time.perf_counter()
                )
                hidden_penalty_ms = max(0.0, (draft_exec_done_time - oracle_ready_at) * 1000.0)
                self._record_spec_metrics(
                    accepted=False,
                    pair_id=pair_id,
                    draft_first=True,
                    fork_created=fork_handle.created,
                    fork_reused=fork_handle.reused,
                    fork_restore_ms=fork_restore_ms,
                    speculative_exec_ms=speculative_exec_ms,
                    hidden_penalty_ms=hidden_penalty_ms,
                    reuse_candidate=False,
                )
                return oracle_turn, observation
            with self._lock:
                self._messages.append({"role": "assistant", "content": oracle_turn.content})
            for cmd in oracle_turn.parsed.commands:
                self._record_completed_command(self._build_command_record(cmd))
            if self._speculation_controller is not None:
                self._promote_speculative_fork(fork_handle.handle)
            # Reuse decision lives entirely in _finalize_speculative_fork.
            # We schedule it asynchronously and return immediately so the
            # next turn's draft+oracle requests can fire without waiting on
            # state_changed inspector roundtrips or the destroy of the
            # ex-active sandbox. The next ensure_fork() will await this
            # finalize to keep the cache decision well-ordered before
            # picking up _cached_fork.
            self._finalize_speculative_fork(
                accepted=True, fork=fork_handle.handle, reuse_candidate=True, pair_id=pair_id,
            )
            speculative_exec_ms = (time.perf_counter() - spec_started) * 1000.0
            # Saved-time accounting: the non-spec path would have started the
            # batch only after oracle returned, so the overlap window we
            # actually saved is bounded by both the draft-exec duration and
            # the oracle-wait window.
            draft_exec_done_time = (
                draft_exec_done_at[0] if draft_exec_done_at else time.perf_counter()
            )
            draft_exec_duration_ms = max(0.0, (draft_exec_done_time - spec_started) * 1000.0)
            oracle_wait_ms = max(0.0, (oracle_ready_at - spec_started) * 1000.0)
            saved_ms = max(0.0, min(draft_exec_duration_ms, oracle_wait_ms))
            self._record_spec_metrics(
                accepted=True,
                pair_id=pair_id,
                draft_first=True,
                fork_created=fork_handle.created,
                fork_reused=fork_handle.reused,
                fork_restore_ms=fork_restore_ms,
                speculative_exec_ms=speculative_exec_ms,
                saved_ms=saved_ms,
                reuse_candidate=True,
            )
            observation = "\n".join(
                filter(None, (self._format_exec_output(r) for r in spec_results))
            ).strip("\n")
            return oracle_turn, observation

        # Mismatch → discard draft work, run oracle on active sandbox.
        # First, kill the fork's container synchronously: the wasted
        # spec_exec is currently competing with the about-to-run oracle
        # exec for host CPU/IO. With the fork dead, the in-flight
        # `runc exec tmux ...` commands inside spec_exec_future return
        # almost immediately with errors, freeing host capacity. The
        # fork's heavier cleanup (ZFS dataset destroy, network lease,
        # inspector unregister) still happens later via the deferred
        # finalize path.
        self._cancel_fork_runtime(fork_handle.handle)
        with self._lock:
            self._messages.append({"role": "assistant", "content": oracle_turn.content})
        observation = self._execute_commands(oracle_turn.parsed.commands)
        # Drain the speculative future. After cancel above, the fork's
        # tmux exec calls fail fast, so the future resolves quickly —
        # the timeout is now defensive (slow runc-delete, edge cases),
        # not a normal-path budget.
        wait([spec_exec_future], timeout=_SPEC_FUTURE_DRAIN_TIMEOUT_S)
        self._finalize_speculative_fork(
            accepted=False, fork=fork_handle.handle, reuse_candidate=False, pair_id=pair_id,
        )
        # Penalty accounting: the agent-visible penalty is whatever extra time
        # the oracle path spent waiting before we could move on (here it's 0
        # because the active runs the oracle in parallel with the fork drain),
        # while hidden penalty captures the host-side cost of the speculative
        # work that ultimately was thrown away.
        speculative_exec_ms = (time.perf_counter() - spec_started) * 1000.0
        draft_exec_done_time = (
            draft_exec_done_at[0] if draft_exec_done_at else time.perf_counter()
        )
        hidden_penalty_ms = max(0.0, (draft_exec_done_time - oracle_ready_at) * 1000.0)
        self._record_spec_metrics(
            accepted=False,
            pair_id=pair_id,
            draft_first=True,
            fork_created=fork_handle.created,
            fork_reused=fork_handle.reused,
            fork_restore_ms=fork_restore_ms,
            speculative_exec_ms=speculative_exec_ms,
            hidden_penalty_ms=hidden_penalty_ms,
            reuse_candidate=False,
        )
        return oracle_turn, observation

    def _submit_spec_future(self, fn, /, *args, **kwargs) -> Future[Any]:
        if self._spec_executor is None:
            raise RuntimeError("speculative executor is unavailable")
        future = self._spec_executor.submit(fn, *args, **kwargs)
        with self._lock:
            self._active_spec_futures.add(future)

        def _discard(completed: Future[Any]) -> None:
            with self._lock:
                self._active_spec_futures.discard(completed)

        future.add_done_callback(_discard)
        return future

    def _ensure_speculative_fork(self) -> _SpeculativeForkHandle:
        if self._speculation_controller is None:
            return _SpeculativeForkHandle(handle=None)
        # Block on the previous turn's deferred finalize so the cache
        # decision (set or clear of `_cached_fork`) is observed before we
        # ask the controller for a fork. The await runs in the spec
        # executor and is hidden behind the current turn's draft+oracle
        # wait — by the time _ensure_speculative_fork is called (i.e.
        # after the draft response), finalize has typically been running
        # for at least the draft delay window already.
        self._await_pending_finalize()
        prepared = self._speculation_controller.ensure_fork()
        if prepared is None:
            return _SpeculativeForkHandle(handle=None)
        if isinstance(prepared, _SpeculativeForkHandle):
            return prepared
        if isinstance(prepared, tuple) and len(prepared) >= 1:
            handle = prepared[0]
            created = bool(prepared[1]) if len(prepared) >= 2 else False
            reused = bool(prepared[2]) if len(prepared) >= 3 else False
            return _SpeculativeForkHandle(handle=handle, created=created, reused=reused)
        return _SpeculativeForkHandle(handle=prepared)

    def _promote_speculative_fork(self, fork) -> None:
        if self._speculation_controller is None:
            return
        promoter = getattr(self._speculation_controller, "promote_fork", None)
        if callable(promoter):
            promoter(fork)

    def _invalidate_speculative_fork(self) -> None:
        if self._speculation_controller is None:
            return
        invalidator = getattr(self._speculation_controller, "invalidate_fork", None)
        if callable(invalidator):
            invalidator()

    def _cancel_fork_runtime(self, fork) -> None:
        """Kill the fork container immediately on the spec-mismatch path
        so its still-running spec_exec stops competing with the oracle
        exec on the active sandbox for host CPU/IO. Cheap synchronous
        runc delete -f; the dataset destroy stays in the deferred
        finalize path."""
        if fork is None or self._speculation_controller is None:
            return
        canceller = getattr(self._speculation_controller, "cancel_fork_runtime", None)
        if callable(canceller):
            canceller(fork)

    def _finalize_speculative_fork(
        self,
        *,
        accepted: bool,
        fork,
        reuse_candidate: bool,
        pair_id: str = "",
    ) -> Future[tuple[bool, bool]] | None:
        """Schedule fork finalization off the critical path.

        Submits the actual state_changed checks and release_fork call to the
        spec executor and returns the future. The next turn's
        `_ensure_speculative_fork` blocks on this future before reading the
        fork-reuse cache, so the reuse decision (which release_fork writes)
        is observed at the right moment without sitting on the current
        turn's critical path.

        Returns None if no spec executor is available (test/fallback path)
        or if there's nothing to release; in that case the synchronous path
        ran inline and the per-turn step gets back the (False, False) pair
        immediately via `_finalize_speculative_fork_sync`.
        """
        if self._speculation_controller is None or fork is None:
            return None
        if self._spec_executor is None:
            # Inline fallback for tests / non-executor configurations.
            self._finalize_speculative_fork_sync(
                accepted=accepted,
                fork=fork,
                reuse_candidate=reuse_candidate,
                pair_id=pair_id,
            )
            return None
        future = self._submit_spec_future(
            self._finalize_speculative_fork_sync,
            accepted=accepted,
            fork=fork,
            reuse_candidate=reuse_candidate,
            pair_id=pair_id,
        )
        with self._lock:
            self._pending_finalize_future = future
        return future

    def _await_pending_finalize(self, *, timeout_s: float | None = None) -> None:
        """Block on the previous turn's deferred finalize so the next
        ensure_fork() observes a settled cache. Hidden behind the next
        turn's draft+oracle wait so the agent loop pays no extra latency
        unless finalize legitimately needed longer than the LLM round trip.
        """
        with self._lock:
            future = self._pending_finalize_future
        if future is None:
            return
        try:
            future.result(timeout=timeout_s)
        except Exception:
            logger.debug(
                "Pending speculative finalize did not complete cleanly sandbox=%s",
                self.sandbox.sandbox_id,
                exc_info=True,
            )
        finally:
            with self._lock:
                if self._pending_finalize_future is future:
                    self._pending_finalize_future = None

    def _finalize_speculative_fork_sync(
        self,
        *,
        accepted: bool,
        fork,
        reuse_candidate: bool,
        pair_id: str = "",
    ) -> tuple[bool, bool]:
        """Release a speculative fork back to the controller and decide
        whether it can be cached for reuse next turn.

        The reuse cache contract from
        docs/speculative-execution-benchmark.md requires both the active
        sandbox AND the fork to report `state_changed=False` on top of the
        caller's own `reuse_candidate` (the controller's "fork-side state
        is well defined" assertion). With our per-sandbox tmux session,
        skipping this gate broke tool-heavy traces: consecutive accepts
        pingponged shell state across the cached pair because each side's
        pane was frozen at the moment it was last active. We now apply the
        same state-changed gating mini_swe uses, so a fork is only kept for
        reuse when no shell-visible work happened on either side.

        Returns the (current_state_changed, fork_state_changed) pair. The
        async wrapper emits a separate telemetry event with this pair so
        the per-turn metrics that don't depend on it can be recorded
        without waiting.
        """
        if self._speculation_controller is None or fork is None:
            return False, False
        try:
            current_state_changed = bool(
                self._speculation_controller.state_changed(self.sandbox)
            )
        except Exception:
            logger.debug(
                "Failed to read state_changed for active sandbox=%s",
                self.sandbox.sandbox_id,
                exc_info=True,
            )
            current_state_changed = True
        try:
            fork_state_changed = bool(
                self._speculation_controller.state_changed(fork)
            )
        except Exception:
            logger.debug(
                "Failed to read state_changed for fork sandbox=%s",
                fork.sandbox_id,
                exc_info=True,
            )
            fork_state_changed = True
        reusable = (
            reuse_candidate
            and not current_state_changed
            and not fork_state_changed
        )
        logger.debug(
            "Finalizing speculative fork sandbox=%s accepted=%s "
            "current_state_changed=%s fork_state_changed=%s "
            "reuse_candidate=%s reusable=%s",
            self.sandbox.sandbox_id,
            accepted,
            current_state_changed,
            fork_state_changed,
            reuse_candidate,
            reusable,
        )
        releaser = getattr(self._speculation_controller, "release_fork", None)
        if callable(releaser):
            releaser(fork, reusable=reusable, accepted=accepted)
        self._emit_spec_finalize_telemetry(
            pair_id=pair_id,
            accepted=accepted,
            reuse_candidate=reuse_candidate,
            reusable=reusable,
            current_state_changed=current_state_changed,
            fork_state_changed=fork_state_changed,
        )
        return current_state_changed, fork_state_changed

    def _emit_spec_finalize_telemetry(
        self,
        *,
        pair_id: str,
        accepted: bool,
        reuse_candidate: bool,
        reusable: bool,
        current_state_changed: bool,
        fork_state_changed: bool,
    ) -> None:
        if self.telemetry is None:
            return
        try:
            self.telemetry.emit_event(
                "spec.finalize",
                {
                    "sandbox_id": str(self.sandbox.sandbox_id),
                    "task_run_id": self._benchmark_task_run_id(),
                    "pair_id": pair_id,
                    "accepted": 1 if accepted else 0,
                    "reuse_candidate": 1 if reuse_candidate else 0,
                    "reusable": 1 if reusable else 0,
                    "current_state_changed": 1 if current_state_changed else 0,
                    "fork_state_changed": 1 if fork_state_changed else 0,
                },
            )
        except Exception:
            logger.debug(
                "Failed to emit spec.finalize telemetry sandbox=%s",
                self.sandbox.sandbox_id,
                exc_info=True,
            )

    def _record_spec_metrics(
        self,
        *,
        accepted: bool,
        pair_id: str = "",
        draft_first: bool = False,
        fork_created: bool = False,
        fork_reused: bool = False,
        fork_restore_ms: float = 0.0,
        speculative_exec_ms: float = 0.0,
        saved_ms: float = 0.0,
        penalty_ms: float = 0.0,
        hidden_penalty_ms: float = 0.0,
        current_state_changed: bool = False,
        fork_state_changed: bool = False,
        reuse_candidate: bool | None = None,
    ) -> None:
        saved_ms = max(0.0, saved_ms)
        penalty_ms = max(0.0, penalty_ms)
        hidden_penalty_ms = max(0.0, hidden_penalty_ms)
        speculative_exec_ms = max(0.0, speculative_exec_ms)
        fork_restore_ms = max(0.0, fork_restore_ms)
        with self._lock:
            self._spec_total_turns += 1
            if accepted:
                self._spec_accept_count += 1
            else:
                self._spec_reject_count += 1
            if fork_created:
                self._spec_fork_create_count += 1
            if fork_reused:
                self._spec_fork_reuse_count += 1
            self._spec_saved_ms += saved_ms
            self._spec_penalty_ms += penalty_ms
            self._spec_hidden_penalty_ms += hidden_penalty_ms
            accept_count = self._spec_accept_count
            reject_count = self._spec_reject_count
        if self.telemetry is None:
            return
        attributes = {
            "sandbox_id": str(self.sandbox.sandbox_id),
            "task_run_id": self._benchmark_task_run_id(),
            "pair_id": pair_id,
            "accepted": 1 if accepted else 0,
            "draft_first": 1 if draft_first else 0,
            "oracle_first": 0 if draft_first else 1,
            "fork_created": 1 if fork_created else 0,
            "fork_reused": 1 if fork_reused else 0,
            "current_state_changed": 1 if current_state_changed else 0,
            "fork_state_changed": 1 if fork_state_changed else 0,
            "fork_finalized": 1 if reuse_candidate is not None else 0,
            "reuse_candidate": 1 if reuse_candidate else 0,
        }
        self.telemetry.emit_event("spec.turn.finish", attributes)
        self.telemetry.emit_metric("benchmark.spec.saved_ms", saved_ms, attributes)
        self.telemetry.emit_metric("benchmark.spec.penalty_ms", penalty_ms, attributes)
        self.telemetry.emit_metric(
            "benchmark.spec.hidden_penalty_ms", hidden_penalty_ms, attributes
        )
        self.telemetry.emit_metric(
            "benchmark.spec.net_gain_ms", saved_ms - penalty_ms, attributes
        )
        self.telemetry.emit_metric(
            "benchmark.spec.fork_restore_ms", fork_restore_ms, attributes
        )
        self.telemetry.emit_metric(
            "benchmark.spec.speculative_exec_ms", speculative_exec_ms, attributes
        )
        decisions = accept_count + reject_count
        self.telemetry.emit_metric(
            "benchmark.spec.accept_rate",
            0.0 if decisions <= 0 else accept_count / decisions,
            attributes,
        )

    @staticmethod
    def _observed_pane_command(session: "_TmuxSandboxSession") -> str:
        """Best-effort `pane_current_command` snapshot for telemetry.

        Returns the foreground process name as tmux reports it via
        `display-message #{pane_current_command}`; on any failure
        returns an empty string so the guard call site stays
        non-fatal.
        """
        try:
            result = session._exec(  # noqa: SLF001 — telemetry-only diagnostic read
                [
                    "tmux",
                    "display-message",
                    "-p",
                    "-t",
                    session._session_name,  # noqa: SLF001
                    "#{pane_current_command}",
                ],
                timeout_s=2.0,
            )
        except Exception:
            return ""
        raw = getattr(result, "stdout", "") or ""
        text = raw if isinstance(raw, str) else raw.decode(errors="replace")
        return text.strip()

    def _record_fast_forward_skip(
        self,
        *,
        command_count: int,
        intended_sleep_ms: float,
        saved_ms: float,
    ) -> None:
        """Per-skip counter + telemetry for the fast-forward path.

        Counterpart to `_record_drain_event` for the slow-replay drain
        guards: together they let the report show how often each
        replay-cadence mechanism kicked in and how much wall time it
        moved (`benchmark.terminus.fast_forward_saved_ms` for skips,
        `benchmark.terminus.guard_drain_ms` for drains).
        """
        with self._lock:
            self._fast_forward_skip_count += 1
            self._fast_forward_saved_ms += saved_ms
        if self.telemetry is None:
            return
        attributes = {
            "sandbox_id": str(self.sandbox.sandbox_id),
            "task_run_id": self._benchmark_task_run_id(),
            "command_count": int(command_count),
            "intended_sleep_ms": float(intended_sleep_ms),
            "saved_ms": float(saved_ms),
        }
        self.telemetry.emit_event("terminus.fast_forward.skip", attributes)
        self.telemetry.emit_metric(
            "benchmark.terminus.fast_forward_saved_ms", saved_ms, attributes
        )

    def _record_drain_event(
        self,
        *,
        site: str,
        busy_cmd: str,
        drain_wait_ms: float,
        drain_resolved: bool,
    ) -> None:
        """Per-drain-event counter + telemetry for the slow-replay guards.

        `site` is one of {"pre_fork", "post_match", "non_spec"}.
        `drain_resolved` is True iff `wait_for_quiescence` returned True
        (pane drained before the timeout); False iff it returned False
        or raised (timeout — we still proceed but accumulate the wait
        as overhead).
        """
        with self._lock:
            if site == "pre_fork":
                self._guard_pre_fork_drain_count += 1
                self._guard_pre_fork_drain_ms += drain_wait_ms
                if not drain_resolved:
                    self._guard_pre_fork_drain_timeout_count += 1
            elif site == "post_match":
                self._guard_post_match_drain_count += 1
                self._guard_post_match_drain_ms += drain_wait_ms
                if not drain_resolved:
                    self._guard_post_match_drain_timeout_count += 1
            elif site == "non_spec":
                self._guard_non_spec_drain_count += 1
                self._guard_non_spec_drain_ms += drain_wait_ms
                if not drain_resolved:
                    self._guard_non_spec_drain_timeout_count += 1
        if self.telemetry is None:
            return
        attributes = {
            "sandbox_id": str(self.sandbox.sandbox_id),
            "task_run_id": self._benchmark_task_run_id(),
            "site": site,
            "busy_cmd": busy_cmd,
            "drain_wait_ms": float(drain_wait_ms),
            "drain_resolved": 1 if drain_resolved else 0,
        }
        if site == "pre_fork":
            event_name = "terminus.guard.pre_fork_drain"
        elif site == "post_match":
            event_name = "terminus.guard.post_match_drain"
        else:
            event_name = "terminus.guard.non_spec_drain"
        self.telemetry.emit_event(event_name, attributes)
        self.telemetry.emit_metric(
            "benchmark.terminus.guard_drain_ms", drain_wait_ms, attributes
        )

    def _benchmark_task_run_id(self) -> str:
        metadata = self.sandbox.launch_metadata.get("benchmark", {})
        if isinstance(metadata, dict):
            raw_task_run_id = metadata.get("task_run_id")
            if isinstance(raw_task_run_id, str) and raw_task_run_id:
                return raw_task_run_id
        return str(self.sandbox.sandbox_id)

    @staticmethod
    def _batches_match(draft: _ParseResult, oracle: _ParseResult) -> bool:
        if draft.is_task_complete != oracle.is_task_complete:
            return False
        if len(draft.commands) != len(oracle.commands):
            return False
        for d, o in zip(draft.commands, oracle.commands):
            if " ".join(d.keystrokes.split()) != " ".join(o.keystrokes.split()):
                return False
        return True

    def _wait_for_tmux_quiescence(
        self, *, timeout_s: float = 1800.0
    ) -> None:
        """Block until the active sandbox's tmux pane drains any
        currently-foreground command. Called at end-of-task so verification
        does not race against an in-progress build/install whose `duration`
        hint underestimated wall-clock cost.

        Default budget is 30 min — long enough for the slowest builds in
        the spec_friendly trace set (compcert/povray/cython post-build
        steps typically clear within ~10 min once the agent stops issuing
        new keystrokes). With non-blocking send_keys the agent's
        task-complete signal fires as soon as the trace's last response
        is consumed, well before bash has finished the queued
        `make all` / `make install` / post-build steps; the previous
        3 min budget was too tight and verification ran against an
        incomplete rootfs.

        Compose-replay mode does not use tmux; bail in that case. Errors
        are logged but never raised — quiescence is best-effort.
        """
        if self._is_compose_replay_mode():
            return
        sandbox = getattr(self, "sandbox", None)
        if sandbox is None:
            return
        sandbox_id = getattr(sandbox, "sandbox_id", None)
        if sandbox_id is None:
            return
        with self._tmux_sessions_lock:
            session = self._tmux_sessions.get(sandbox_id)
        if session is None:
            return
        try:
            ok = session.wait_for_quiescence(timeout_s=timeout_s)
        except Exception as exc:
            logger.warning(
                "tmux quiescence wait raised sandbox=%s err=%s",
                sandbox_id,
                exc,
            )
            return
        if not ok:
            logger.warning(
                "tmux quiescence marker not observed within %.1fs sandbox=%s; "
                "verification may race against an in-progress command",
                timeout_s,
                sandbox_id,
            )

    def _tmux_session_for(self, sandbox_id: Any) -> _TmuxSandboxSession:
        """Return (and lazily create) the persistent tmux session for the
        given sandbox_id. Each TerminusAgent caches one session per sandbox
        so cwd / env / shell variables persist across the trace's separate
        tool_call invocations — matching how the original Terminal-Bench
        Terminus 2 agent runs commands inside a single tmux pane.
        """
        with self._tmux_sessions_lock:
            session = self._tmux_sessions.get(sandbox_id)
            if session is not None:
                return session
            session = _TmuxSandboxSession(
                runtime=self.runtime,
                sandbox_id=sandbox_id,
                cwd=self._process_cwd(),
                user=self._process_user(),
                env=self._process_env(),
                stop_check=self._stop_requested.is_set,
            )
            self._tmux_sessions[sandbox_id] = session
            return session

    def _execute_records_for_sandbox(
        self, records: list[_RecordedCommand], sandbox_id
    ) -> list[Any]:
        """Execute a batch of recorded commands inside the given sandbox's
        persistent tmux session. Used by the spec-replay path to send the
        draft batch to a fork without disturbing the active sandbox."""
        assert self.runtime is not None
        session = self._tmux_session_for(sandbox_id)
        results: list[Any] = []
        for record in records:
            session.send_keys(
                record.keystrokes,
                min_timeout_sec=record.duration_s,
                max_timeout_sec=record.timeout_s,
            )
            results.append(
                _TmuxExecResult(
                    returncode=0,
                    stdout="",
                    stderr="",
                )
            )
        return results

    def _execute_commands(self, commands: list[_ParsedCommand]) -> str:
        if not commands:
            return ""
        if self._maybe_fast_forward_idle_wait(commands):
            return ""
        # Slow-replay drain at the universal execute-commands entry.
        # Parity with the spec pre_fork / post_match guards: if the
        # active pane still has a foreground command running (replay
        # host slower than capture host on this turn — typical for
        # builds), wait for it to drain before issuing the next batch
        # of keystrokes. Otherwise the send_keys would queue behind the
        # in-flight command and inflate every subsequent turn's wall
        # time the same way unbounded queueing did on the spec path
        # before its guards were added.
        #
        # Reached by:
        #   * Non-spec (nofault/fault) execution path — primary motivation.
        #   * Spec rejection paths that didn't go through pre_fork
        #     drain (oracle-first, draft-error, empty-draft fast path).
        #   * Spec rejection paths after the pre_fork or post_match
        #     drain already fired — probe sees idle pane, skips. Cheap.
        # Telemetry attributes events here to site=non_spec; the
        # explicit upstream guards keep their own pre_fork/post_match
        # sites so the report can still distinguish them.
        try:
            session = self._tmux_session_for(self.sandbox.sandbox_id)
            pane_busy = not session.is_pane_idle()
        except Exception:
            pane_busy = False
        if pane_busy:
            busy_cmd = self._observed_pane_command(session)
            drain_started = time.perf_counter()
            drain_resolved = False
            try:
                drain_resolved = bool(
                    session.wait_for_quiescence(
                        timeout_s=_SPEC_PRE_FORK_QUIESCENCE_TIMEOUT_S
                    )
                )
            except Exception:
                logger.debug(
                    "wait_for_quiescence raised in non-spec drain sandbox=%s",
                    self.sandbox.sandbox_id,
                    exc_info=True,
                )
            drain_wait_ms = (time.perf_counter() - drain_started) * 1000.0
            self._record_drain_event(
                site="non_spec",
                busy_cmd=busy_cmd,
                drain_wait_ms=drain_wait_ms,
                drain_resolved=drain_resolved,
            )
        outputs: list[str] = []
        for command in commands:
            recorded = self._build_command_record(command)
            with self._lock:
                self._command_in_flight = True
                self._current_command = recorded
            self._restore_event.clear()
            self._set_status(self.poll_status())
            try:
                result = self._execute_recorded_command(recorded)
            finally:
                with self._lock:
                    self._command_in_flight = False
                    self._current_command = None
                self._set_status(self.poll_status())
            output = self._format_exec_output(result)
            if output:
                outputs.append(output)
        return "\n".join(outputs).strip("\n") if outputs else ""

    def _maybe_fast_forward_idle_wait(self, commands: list[_ParsedCommand]) -> bool:
        """Skip a pure-wait turn (`keystrokes=""`, recorded `duration` only)
        when the active pane is idle and its content is stable.

        The trace's wait turns exist to give an in-flight foreground
        command (typically a build) the wall-clock budget to advance
        between two LLM observations. When the replay's pane finishes
        that command earlier than the trace's pacing — usually because
        the replay host has more CPU/IO than the capture host — the
        sleep produces no new output and just inflates wall time. This
        check absorbs that drift by short-circuiting empty waits when
        the pane already drained.

        Conservative check, in order:
          1. all commands have empty/whitespace keystrokes
          2. `is_pane_idle()` (foreground process is the shell)
          3. `capture_pane()` snapshot stable across `_FAST_FORWARD_SETTLE_S`
          4. `is_pane_idle()` again (no new fg command started during settle)

        Any False answer here returns False and the caller falls back to
        the regular send_keys path, which will sleep the recorded
        `min_timeout_sec` exactly as it did before this knob existed.
        Net cost on a False answer is ~`_FAST_FORWARD_SETTLE_S` plus
        two display-message + capture-pane round-trips.
        """
        if not self._fast_forward_idle_waits:
            return False
        if not commands:
            return False
        if not all(not cmd.keystrokes.strip() for cmd in commands):
            return False
        # Mirror send_keys's per-command 60s cap so the saved_ms accounting
        # matches what the agent would actually have slept.
        intended_sleep_ms = sum(
            min(max(0.0, cmd.duration_s), _DEFAULT_COMMAND_DURATION_CAP_S)
            for cmd in commands
        ) * 1000.0
        if intended_sleep_ms <= 0.0:
            return False
        try:
            session = self._tmux_session_for(self.sandbox.sandbox_id)
            if not session.is_pane_idle():
                return False
            snap_before = session.capture_pane()
            time.sleep(_FAST_FORWARD_SETTLE_S)
            if not session.is_pane_idle():
                return False
            snap_after = session.capture_pane()
        except Exception:
            logger.debug(
                "fast-forward probe raised sandbox=%s; falling through to send_keys sleep",
                self.sandbox.sandbox_id,
                exc_info=True,
            )
            return False
        if snap_before != snap_after:
            return False
        saved_ms = max(0.0, intended_sleep_ms - _FAST_FORWARD_SETTLE_S * 1000.0)
        self._record_fast_forward_skip(
            command_count=len(commands),
            intended_sleep_ms=intended_sleep_ms,
            saved_ms=saved_ms,
        )
        return True

    def _execute_recorded_command(self, command_record: _RecordedCommand) -> Any:
        """Send the keystrokes to the active sandbox's tmux session. Retains
        the existing retry-after-restore loop in case a preempted sandbox
        comes back: on failure we wait for restore completion and replay any
        history the harness expects us to re-emit before the next command.
        """
        assert self.runtime is not None
        while True:
            if self._stop_requested.is_set():
                # The harness asked us to stop — don't keep retrying
                # against a sandbox that may be on its way out.
                # Re-raise so the outer turn loop exits via its
                # normal failure path; the next iteration check at
                # the top of the loop will see _stop_requested too.
                raise RuntimeError(
                    f"stop requested while executing command on sandbox {self.sandbox.sandbox_id}"
                )
            try:
                result = self._run_command_record(command_record, resilient=False)
            except _TmuxProtocolError:
                # tmux itself rejected the command (e.g. payload exceeded
                # the imsg cap). A restore won't fix this, so don't burn
                # _RESTORE_WAIT_TIMEOUT_S waiting for one — let the task
                # fail and surface the real error.
                raise
            except Exception:
                # Container/preemption-style failures: the sandbox is
                # being torn down or restored mid-send. Wait for the
                # restore to complete, replay any history the harness
                # expects, and retry.
                logger.warning(
                    "tmux send failed for sandbox=%s action=%d; awaiting restore",
                    self.sandbox.sandbox_id,
                    command_record.action_index,
                    exc_info=True,
                )
                self._wait_for_restore_completion()
                if self._stop_requested.is_set():
                    raise RuntimeError(
                        f"stop requested while awaiting restore for sandbox {self.sandbox.sandbox_id}"
                    )
                self._drain_pending_restore_replays(
                    max_replay_action_index=command_record.action_index - 1
                )
                continue
            self._record_completed_command(command_record)
            return result

    def _run_command_record(self, command_record: _RecordedCommand, *, resilient: bool) -> Any:
        assert self.runtime is not None
        # `resilient` is ignored in the tmux path: send-keys is a thin
        # wrapper over a `runtime.exec(["tmux", ...])` call, and any
        # preemption surfaces as an exception that the caller already
        # handles by waiting for restore completion. The flag is kept in
        # the signature for compatibility with the surrounding harness.
        del resilient
        session = self._tmux_session_for(self.sandbox.sandbox_id)
        session.send_keys(
            command_record.keystrokes,
            min_timeout_sec=command_record.duration_s,
            max_timeout_sec=command_record.timeout_s,
        )
        return _TmuxExecResult(returncode=0, stdout="", stderr="")

    def _build_command_record(
        self, command: _ParsedCommand, *, action_index: int | None = None
    ) -> _RecordedCommand:
        raw_timeout = self.task_config.options.get("max_agent_timeout_sec", 900.0)
        try:
            base_timeout_s = max(1.0, float(raw_timeout))
        except (TypeError, ValueError):
            base_timeout_s = 900.0
        # The recorded duration is a tmux wait-after-keystrokes hint, not a hard
        # cap. We still want to bound execution by the task's overall timeout.
        # Allow up to 30 min per command (the task as a whole still has its
        # own timeout via base_timeout_s); long builds (e.g. compcert's
        # `make all` or apt-get install of large dep sets) routinely exceed
        # the 600s default and would otherwise time out the wait-for barrier.
        timeout_s = max(command.duration_s + 5.0, min(base_timeout_s, 1800.0))
        return _RecordedCommand(
            action_index=self._next_action_index() if action_index is None else int(action_index),
            keystrokes=command.keystrokes,
            duration_s=command.duration_s,
            cwd=self._process_cwd(),
            env=self._process_env(),
            user=self._process_user(),
            timeout_s=timeout_s,
        )

    def _format_exec_output(self, result: Any) -> str:
        stdout = "" if result.stdout is None else str(result.stdout)
        stderr = "" if result.stderr is None else str(result.stderr)
        if stderr and stdout:
            return stdout + stderr
        return stdout or stderr

    def _is_retryable_exec_failure(self, result: Any) -> bool:
        if int(result.returncode) == 0:
            return False
        stdout = "" if result.stdout is None else str(result.stdout)
        stderr = "" if result.stderr is None else str(result.stderr)
        merged = f"{stdout}\n{stderr}".lower()
        return any(fragment in merged for fragment in _RETRYABLE_EXEC_ERROR_FRAGMENTS)

    def _wait_for_restore_completion(self) -> None:
        # Poll _stop_requested and _restore_event together. The full
        # _RESTORE_WAIT_TIMEOUT_S window is honored only while the
        # task is still active; once teardown calls request_stop(),
        # we must abort promptly so task_executor.shutdown can drain.
        # Pre-fix, this slept the full 300s during teardown, which
        # dovetailed with the "container does not exist" path because
        # runtime.delete_all had already torn the container down by
        # the time the wait finally returned. Observed in benchmark
        # 20260429_063012 where fault-10 burned 300s here while the
        # harness sat at task_executor.shutdown.
        deadline = time.monotonic() + _RESTORE_WAIT_TIMEOUT_S
        while True:
            if self._stop_requested.is_set():
                # Caller's outer loop will see _stop_requested and
                # exit cleanly; we do NOT raise here so the existing
                # "tmux send failed; awaiting restore" branch can
                # unwind without surfacing the timeout.
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"timed out waiting for restore completion for sandbox {self.sandbox.sandbox_id} after "
                    f"{_RESTORE_WAIT_TIMEOUT_S:.1f}s"
                )
            # Cap each wait at 1s so stop signals are picked up
            # quickly without busy-looping; restores normally fire
            # the event within seconds, so this slice doesn't add
            # meaningful latency to the happy path.
            if self._restore_event.wait(timeout=min(1.0, remaining)):
                self._restore_event.clear()
                return

    def _drain_pending_restore_replays(self, *, max_replay_action_index: int | None = None) -> None:
        while True:
            with self._lock:
                pending_generation = self._pending_restore_generation
                if pending_generation <= self._handled_restore_generation:
                    return
                restore_trace_cursor = self._pending_restore_trace_cursor
                replay_target = self._completed_actions
                if max_replay_action_index is not None:
                    replay_target = min(replay_target, max_replay_action_index)
                if restore_trace_cursor is None or replay_target <= restore_trace_cursor:
                    self._handled_restore_generation = pending_generation
                    return
                commands = [
                    record
                    for record in self._command_history
                    if restore_trace_cursor < record.action_index <= replay_target
                ]
            logger.info(
                "Replaying terminus commands after restore sandbox=%s restore_trace_cursor=%s replay_target=%s count=%d",
                self.sandbox.sandbox_id,
                restore_trace_cursor,
                replay_target,
                len(commands),
            )
            for record in commands:
                replay_result = self._run_command_record(record, resilient=True)
                if int(replay_result.returncode) != 0:
                    logger.warning(
                        "Replayed terminus command returned non-zero sandbox=%s action=%d returncode=%d",
                        self.sandbox.sandbox_id,
                        record.action_index,
                        int(replay_result.returncode),
                    )
            with self._lock:
                if self._pending_restore_generation == pending_generation:
                    self._handled_restore_generation = pending_generation
                    return

    def _next_action_index(self) -> int:
        with self._lock:
            return self._completed_actions + 1

    def _record_completed_command(self, command_record: _RecordedCommand) -> None:
        with self._lock:
            self._completed_actions += 1
            self._command_history.append(command_record)

    def _llm_request_timeout_seconds(self) -> float:
        raw_timeout = self.task_config.options.get("max_agent_timeout_sec", _DEFAULT_LLM_REQUEST_TIMEOUT_S)
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError):
            timeout_s = _DEFAULT_LLM_REQUEST_TIMEOUT_S
        return max(60.0, timeout_s)

    def _bundle_process_config(self) -> dict[str, object]:
        config_path = self.sandbox.bundle_dir / "config.json"
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        process = payload.get("process", {})
        if not isinstance(process, dict):
            raise RuntimeError(f"unsupported process config for sandbox {self.sandbox.sandbox_id}")
        return process

    def _process_cwd(self) -> str:
        process = self._bundle_process_config()
        cwd = process.get("cwd")
        return str(cwd) if isinstance(cwd, str) and cwd else "/app"

    def _process_env(self) -> dict[str, object]:
        process = self._bundle_process_config()
        env_items = process.get("env", [])
        defaults = [str(item) for item in env_items] if isinstance(env_items, list) else []
        merged = sandbox_bundle.merge_environment_defaults(
            defaults,
            [
                "PAGER=cat",
                "MANPAGER=cat",
                "LESS=-R",
                "PIP_PROGRESS_BAR=off",
                "TQDM_DISABLE=1",
                "DEBIAN_FRONTEND=noninteractive",
            ],
        )
        env_map: dict[str, object] = {}
        for item in merged:
            key, _, value = item.partition("=")
            env_map[key] = value
        return env_map

    def _process_user(self) -> str | None:
        process = self._bundle_process_config()
        user = process.get("user")
        if not isinstance(user, dict):
            return None
        uid = user.get("uid")
        gid = user.get("gid")
        if uid is None:
            return None
        return f"{uid}:{gid}" if gid is not None else str(uid)

    def _replay_trace_is_complete(self) -> bool:
        return bool(self._replay_router_state().get("is_complete", False))

    def _replay_router_state(self) -> dict[str, Any]:
        if not self.sandbox.llm_control_base_url:
            return {}
        try:
            payload = self.wait_for_http_json(
                f"{self.sandbox.llm_control_base_url}/control/state?sandbox_id={self.sandbox.sandbox_id}",
                timeout_s=5.0,
            )
        except Exception:
            logger.debug(
                "Failed to read terminus replay router state sandbox=%s",
                self.sandbox.sandbox_id,
                exc_info=True,
            )
            return {}
        state = payload.get("state")
        if not isinstance(state, dict):
            return {}
        nested_state = state.get("state")
        if not isinstance(nested_state, dict):
            return {}
        return nested_state

    def _wait_for_action_count(self, target_actions: int) -> None:
        # Callers (e.g. spot scenario's choose_replay_chunk_targets) derive
        # `target_actions` from `trace_response_count`, which is in turn space.
        # Comparing against `total_actions` (command count) is dimensionally
        # wrong for terminus because each turn carries 1-N commands, and
        # additionally caps at `trace_action_count` whenever the trace has any
        # response with empty tool_calls — that gap makes the spot last chunk
        # unreachable and surfaces as "task finished before reaching action
        # count". Align with iflow / claude_code: poll the LLM router's trace
        # cursor (turns consumed) so the comparison is dimensionally
        # consistent with the targets the harness asked for.
        deadline = time.monotonic() + self._action_wait_timeout_seconds(target_actions)
        while time.monotonic() < deadline:
            payload = self.poll_status()
            current_cursor = int(payload.get("replay_trace_cursor", 0))
            if current_cursor >= target_actions:
                return
            if self.sandbox.task_future is not None and self.sandbox.task_future.done():
                self.sandbox.task_future.result()
                raise RuntimeError(
                    f"terminus replay task finished before reaching replay action count {target_actions}; "
                    f"last observed count was {current_cursor}"
                )
            time.sleep(0.2)
        raise RuntimeError(f"timed out waiting for terminus replay action count {target_actions}")

    def _action_wait_timeout_seconds(self, target_actions: int) -> float:
        raw_task_timeout = self.task_config.options.get("max_agent_timeout_sec", 900.0)
        try:
            task_timeout = float(raw_task_timeout)
        except (TypeError, ValueError):
            task_timeout = 900.0
        return max(45.0, task_timeout, target_actions * 10.0)

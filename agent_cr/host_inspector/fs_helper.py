from __future__ import annotations

import itertools
import json
import logging
import os
import queue
import subprocess
import time
from dataclasses import dataclass, field
from threading import Event, Lock, Thread
from typing import Callable

from .protocol import HelperEvent, decode_event, encode_command

logger = logging.getLogger(__name__)


"""Per-sandbox event dispatch removes the prior 8-worker hash pool
entirely. At 10-sandbox scale the hash pool combined with a per-sandbox
sync barrier was sufficient (only one bucket hot at a time). At 54+
sandboxes — measured in benchmark 20260429_031243 — multiple buckets
are simultaneously hot because each bucket carries ~7 sandboxes and any
of them can burst, so the calling sandbox's barrier waits behind any
peer in its own bucket. One queue+thread per registered sandbox
eliminates HOL blocking by construction at any scale; thread overhead
is trivial relative to the ZFS/CRIU/runc cost the host already pays per
sandbox.
"""


@dataclass
class _SyncTiming:
    """Per-sync() instrumentation record.

    `sent_ts` is stamped by sync() before writing the command to stdin.
    `ack_ts` is stamped by the reader thread the moment the matching
    sync_ack line is parsed. `helper_drain_us` / `helper_events` carry
    the counters the helper attached to its sync_ack (see emit_sync_ack
    in fs_monitor.c). `waiter` is set by the per-sandbox worker once it
    has drained past the barrier — at which point every event for that
    sandbox emitted before sync_ack has been applied. sync() reads
    these after waiter.wait() to log a latency breakdown (helper vs.
    Python-worker drain).

    `sandbox_id` carries the caller's target sandbox so the ack handler
    can fence only that sandbox's worker — events for other sandboxes
    are irrelevant to this sync and must not block its return.

    `target_depth` is the calling sandbox's queue size at the moment
    the barrier was enqueued. `peer_depth_max` / `peer_depth_sum` give
    a quick global-shape diagnostic across other sandboxes' queues so
    backlog skew is still visible even though it can no longer block
    this sync.
    """

    waiter: Event
    sandbox_id: str | None = None
    sent_ts: float = 0.0
    ack_ts: float = 0.0
    helper_drain_us: int = 0
    helper_events: int = 0
    target_depth: int = 0
    peer_depth_max: int = 0
    peer_depth_sum: int = 0
    peer_count: int = 0


@dataclass(frozen=True)
class _Barrier:
    """Fence marker enqueued behind all events that preceded a sync_ack.

    Each worker processes events in FIFO order; when a worker reaches its
    barrier it decrements the counter under the lock, and the last worker
    to arrive sets the waiter Event. That guarantees: when sync() returns,
    every event emitted by the helper before the matching sync_ack has
    been applied by its owning worker.

    Sync uses the caller's own Event as the waiter so there's no relay
    thread between the worker and the sync() caller.
    """

    sync_id: int
    counter: list[int]
    waiter: Event
    lock: Lock


class _EventWorkerPool:
    """Per-sandbox event dispatch.

    One queue + worker thread per registered sandbox. Events for the
    same sandbox remain strictly ordered (required: the daemon's state
    updates are not commutative); events for different sandboxes
    process fully in parallel with zero head-of-line blocking.

    Workers are created lazily on the first dispatched event for a
    sandbox (the C helper drops events for unregistered cgroups, so a
    surprise sandbox here only happens in the brief register/
    unregister window). They are torn down in `unregister_sandbox`
    after a poison-pill sentinel.
    """

    def __init__(self, on_event: Callable[[HelperEvent], None]) -> None:
        self._on_event = on_event
        self._lock = Lock()
        self._queues: dict[str, queue.Queue[object]] = {}
        self._threads: dict[str, Thread] = {}

    def register_sandbox(self, sandbox_id: str) -> None:
        """Idempotently spin up the per-sandbox queue + worker. Called
        from LibbpfFilesystemMonitor.upsert_sandbox so the worker is
        ready before the C helper starts emitting events."""
        with self._lock:
            if sandbox_id in self._queues:
                return
            q: queue.Queue[object] = queue.Queue()
            thread = Thread(
                target=self._run_worker,
                args=(sandbox_id, q),
                name=f"host-inspector-fs-event-{sandbox_id}",
                daemon=True,
            )
            self._queues[sandbox_id] = q
            self._threads[sandbox_id] = thread
            thread.start()

    def unregister_sandbox(self, sandbox_id: str) -> None:
        """Drain + tear down a sandbox's worker. Sends a poison pill so
        any events already queued get processed before the thread
        exits — avoids losing late-arriving events that had already
        been dispatched before the unregister."""
        with self._lock:
            q = self._queues.pop(sandbox_id, None)
            thread = self._threads.pop(sandbox_id, None)
        if q is not None:
            q.put(None)
        if thread is not None:
            thread.join(timeout=1.0)

    def dispatch(self, event: HelperEvent) -> None:
        with self._lock:
            q = self._queues.get(event.sandbox_id)
        if q is None:
            # Race: event for a sandbox not yet registered (or already
            # unregistered). Lazily register so the event isn't lost;
            # the C helper's cgroup filter ensures we never see events
            # for sandboxes we never registered, so no leak.
            self.register_sandbox(event.sandbox_id)
            with self._lock:
                q = self._queues.get(event.sandbox_id)
        if q is not None:
            q.put(event)

    def barrier_for_sandbox(
        self, sandbox_id: str, sync_id: int, waiter: Event
    ) -> tuple[int, int, int, int]:
        """Enqueue a fence on `sandbox_id`'s queue; `waiter` fires when
        that worker has drained past the fence. No fence on any other
        sandbox's queue — by construction this sync cannot be blocked
        by a peer sandbox's burst.

        Returns `(target_depth, peer_depth_max, peer_depth_sum,
        peer_count)`. Peer-depth fields are diagnostic only — they
        give a global-shape signal without serializing on peer drain.
        """
        marker = _Barrier(
            sync_id=sync_id,
            counter=[1],
            waiter=waiter,
            lock=Lock(),
        )
        with self._lock:
            target_q = self._queues.get(sandbox_id)
            peer_qs = [q for sid, q in self._queues.items() if sid != sandbox_id]
        if target_q is None:
            # No worker for this sandbox — nothing to wait for. Fire
            # immediately so the caller doesn't block on a sync()
            # that has no in-flight events to drain.
            waiter.set()
            return 0, 0, 0, 0
        # qsize() is approximate under concurrent producers, but this
        # is diagnostic output only; the exact count isn't load-bearing.
        target_depth = target_q.qsize()
        peer_depths = [q.qsize() for q in peer_qs]
        peer_max = max(peer_depths) if peer_depths else 0
        peer_sum = sum(peer_depths)
        target_q.put(marker)
        return target_depth, peer_max, peer_sum, len(peer_qs)

    def stop(self) -> None:
        with self._lock:
            queues = list(self._queues.values())
            threads = list(self._threads.values())
            self._queues.clear()
            self._threads.clear()
        for q in queues:
            q.put(None)
        for thread in threads:
            thread.join(timeout=1.0)

    def _run_worker(self, sandbox_id: str, q: queue.Queue[object]) -> None:
        while True:
            item = q.get()
            if item is None:
                return
            if isinstance(item, _Barrier):
                with item.lock:
                    item.counter[0] -= 1
                    is_last = item.counter[0] == 0
                if is_last:
                    item.waiter.set()
                continue
            try:
                self._on_event(item)  # type: ignore[arg-type]
            except Exception:
                logger.exception("Error applying fs event for sandbox %s", sandbox_id)


class LibbpfFilesystemMonitor:
    def __init__(self, helper_path: str | None = None) -> None:
        default_path = os.path.join(os.path.dirname(__file__), "bpf", "fs_monitor")
        self._helper_path = helper_path or default_path
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: Thread | None = None
        self._stderr_thread: Thread | None = None
        self._stdin_lock = Lock()
        self._on_event: Callable[[HelperEvent], None] | None = None
        self._sync_counter = itertools.count(1)
        self._sync_lock = Lock()
        self._sync_waiters: dict[int, _SyncTiming] = {}
        self._worker_pool: _EventWorkerPool | None = None

    def start(self, on_event: Callable[[HelperEvent], None]) -> None:
        if self._process is not None:
            return
        self._on_event = on_event
        self._worker_pool = _EventWorkerPool(on_event)
        self._process = subprocess.Popen(
            [self._helper_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdout_thread = Thread(target=self._read_stdout, name="host-inspector-fs-monitor", daemon=True)
        self._stderr_thread = Thread(target=self._read_stderr, name="host-inspector-fs-monitor-stderr", daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def stop(self) -> None:
        if self._process is None:
            return
        process = self._process
        self._process = None
        try:
            if process.stdin is not None:
                process.stdin.close()
        except BrokenPipeError:
            pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                logger.warning("filesystem monitor helper did not exit after terminate; killing pid=%s", process.pid)
                process.kill()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    logger.error("filesystem monitor helper did not exit after kill; pid=%s", process.pid)
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1.0)
            self._stdout_thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None
        if self._worker_pool is not None:
            self._worker_pool.stop()
            self._worker_pool = None
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()

    def upsert_sandbox(self, sandbox_id: str, cgroup_id: int) -> None:
        # Spin up the per-sandbox worker BEFORE the C helper starts
        # routing events for this cgroup, so the first event has a
        # destination queue and doesn't take the dispatch fallback path.
        if self._worker_pool is not None:
            self._worker_pool.register_sandbox(sandbox_id)
        self._send({"op": "upsert_sandbox", "sandbox_id": sandbox_id, "cgroup_id": cgroup_id})

    def remove_sandbox(self, sandbox_id: str) -> None:
        self._send({"op": "remove_sandbox", "sandbox_id": sandbox_id})
        # Tear down AFTER the C helper drops the cgroup mapping so
        # late-arriving events for the same sandbox don't recreate
        # a worker that immediately leaks. The pool's poison-pill
        # drain ensures already-queued events still get applied.
        if self._worker_pool is not None:
            self._worker_pool.unregister_sandbox(sandbox_id)

    def add_ignored_pid(self, pid: int) -> None:
        if pid <= 0:
            return
        self._send({"op": "add_ignored_pid", "pid": int(pid)})

    def remove_ignored_pid(self, pid: int) -> None:
        if pid <= 0:
            return
        self._send({"op": "remove_ignored_pid", "pid": int(pid)})

    def set_ignored_path_prefixes(
        self, sandbox_id: str, prefixes: tuple[str, ...]
    ) -> None:
        """Push the per-sandbox path-prefix filter into the C helper so it
        can drop matching events at the user-space C layer, before any
        JSON serialization or Python parsing. Uses tab as the separator
        (paths can't natively contain tabs) so the wire format stays
        plain JSON without nested arrays."""
        joined = "\t".join(p for p in prefixes if p)
        self._send(
            {
                "op": "set_ignored_path_prefixes",
                "sandbox_id": sandbox_id,
                "prefixes": joined,
            }
        )

    def set_ignore_process_rules(
        self,
        sandbox_id: str,
        rules: "tuple[ProcessIgnoreRule, ...]",
    ) -> None:
        """Push the per-sandbox process-ignore-rule filter into the C
        helper. Only scope=all rules are sent — scope=process_only rules
        suppress process_changed but must NOT drop fs events, so they
        stay Python-only (evaluated in `_split_pids`).

        Wire encoding: one `clear_ignore_process_rules` op followed by
        one `add_ignore_process_rule` op per rule. cmdline_contains is
        joined on '|' since the JSON encoder passes it through cleanly
        and the rule values used by terminus are short literal
        substrings (no '|' inside them). Empty fields become empty
        strings on the wire and "no constraint" on the helper side —
        identical to how Python's ProcessIgnoreRule.matches treats
        them."""
        from .process_filter import SCOPE_ALL  # local import: avoid module-level cycle

        self._send({"op": "clear_ignore_process_rules", "sandbox_id": sandbox_id})
        for rule in rules:
            if rule.scope != SCOPE_ALL:
                continue
            self._send(
                {
                    "op": "add_ignore_process_rule",
                    "sandbox_id": sandbox_id,
                    "executable_basename": rule.executable_basename or "",
                    "executable_path_contains": rule.executable_path_contains or "",
                    "cmdline_contains": "|".join(rule.cmdline_contains),
                    "ancestor_executable_basename": rule.ancestor_executable_basename or "",
                }
            )

    def sync(self, sandbox_id: str, timeout_s: float = 2.0) -> bool:
        """Drain events currently in flight for `sandbox_id` and wait
        until each has been delivered to `on_event`.

        Returns True on success, False if the helper is not running or
        the ack did not arrive before `timeout_s` elapsed.

        The kernel submits fs_event records to the ring buffer at
        syscall exit, but the userspace helper consumes them serially.
        The reader thread then dispatches events to a per-sandbox
        worker (one queue + thread per registered sandbox). Because
        per-sandbox ordering must be preserved (state updates are not
        commutative), sync uses a barrier marker: when the reader sees
        the helper's sync_ack it enqueues a fence on this sandbox's
        worker, and sync() returns when that worker has drained past
        the fence. Other sandboxes have their own queues — by
        construction this sync cannot be blocked by a peer's burst.
        """
        if self._process is None or self._process.stdin is None:
            return False
        sync_id = next(self._sync_counter)
        timing = _SyncTiming(waiter=Event(), sandbox_id=sandbox_id)
        with self._sync_lock:
            self._sync_waiters[sync_id] = timing
        try:
            timing.sent_ts = time.monotonic()
            self._send({"op": "sync", "sync_id": sync_id})
            ok = timing.waiter.wait(timeout=timeout_s)
            done_ts = time.monotonic()
            self._log_sync_latency(sync_id, ok, timing, done_ts, timeout_s)
            return ok
        finally:
            with self._sync_lock:
                self._sync_waiters.pop(sync_id, None)

    def _log_sync_latency(
        self,
        sync_id: int,
        ok: bool,
        timing: _SyncTiming,
        done_ts: float,
        timeout_s: float,
    ) -> None:
        """Emit a breakdown of where the sync() wall time went.

        We split total latency into two phases so bottleneck analysis
        is unambiguous:
          - helper_ms: sync command sent → sync_ack line parsed on the
            reader. Dominated by the C helper's ring_buffer__poll /
            consume + per-event /proc syscalls + stdout writes.
          - worker_ms: sync_ack parsed → barrier fires on the caller's
            sandbox queue. Dominated by Python per-event daemon work
            and the backlog on that one queue.

        peer_max / peer_sum / peer_count summarize backlog on OTHER
        sandboxes' queues at fence time. They no longer affect this
        sync's drain time (per-sandbox queues), but a sustained peer
        backlog still flags an event-rate-vs-Python-throughput problem
        that's worth surfacing.
        """
        total_ms = (done_ts - timing.sent_ts) * 1000.0
        if timing.ack_ts > 0.0:
            helper_ms_str = f"{(timing.ack_ts - timing.sent_ts) * 1000.0:.1f}"
            worker_ms_str = f"{(done_ts - timing.ack_ts) * 1000.0:.1f}"
        else:
            helper_ms_str = "n/a"
            worker_ms_str = "n/a"
        drain_ms_str = f"{timing.helper_drain_us / 1000.0:.1f}" if timing.ack_ts > 0.0 else "n/a"
        sandbox_str = timing.sandbox_id or "?"
        if ok:
            logger.debug(
                "fs_sync id=%d ok=1 sandbox=%s target_depth=%d "
                "total_ms=%.1f helper_ms=%s helper_drain_ms=%s "
                "helper_events=%d worker_ms=%s peer_max=%d peer_sum=%d peer_n=%d",
                sync_id,
                sandbox_str,
                timing.target_depth,
                total_ms,
                helper_ms_str,
                drain_ms_str,
                timing.helper_events,
                worker_ms_str,
                timing.peer_depth_max,
                timing.peer_depth_sum,
                timing.peer_count,
            )
        else:
            logger.warning(
                "fs_sync id=%d TIMEOUT sandbox=%s target_depth=%d "
                "timeout_s=%.1f total_ms=%.1f helper_ms=%s helper_drain_ms=%s "
                "helper_events=%d worker_ms=%s peer_max=%d peer_sum=%d peer_n=%d",
                sync_id,
                sandbox_str,
                timing.target_depth,
                timeout_s,
                total_ms,
                helper_ms_str,
                drain_ms_str,
                timing.helper_events,
                worker_ms_str,
                timing.peer_depth_max,
                timing.peer_depth_sum,
                timing.peer_count,
            )

    def _send(self, payload: dict[str, object]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("filesystem monitor helper is not running")
        with self._stdin_lock:
            self._process.stdin.write(encode_command(payload))
            self._process.stdin.flush()

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        try:
            for line in self._process.stdout:
                line = line.strip()
                if not line:
                    continue
                if self._handle_control_line(line):
                    continue
                try:
                    event = decode_event(line)
                except Exception:
                    logger.exception("Failed to decode fs helper event: %s", line)
                    continue
                if self._worker_pool is not None:
                    self._worker_pool.dispatch(event)
                elif self._on_event is not None:
                    self._on_event(event)
        except ValueError:
            logger.debug("filesystem monitor stdout closed during shutdown")

    def _handle_control_line(self, line: str) -> bool:
        if '"kind":"sync_ack"' not in line:
            return False
        try:
            payload = json.loads(line)
        except Exception:
            return False
        if not isinstance(payload, dict) or payload.get("kind") != "sync_ack":
            return False
        raw_id = payload.get("sync_id")
        if raw_id is None:
            return True
        try:
            sync_id = int(raw_id)
        except (TypeError, ValueError):
            return True
        # sync_ack is the fence marker: every event that was emitted
        # before it is already ahead of us on this thread, and we've
        # already dispatched each to its per-sandbox worker queue.
        # Enqueue a barrier ONLY on the bucket owning the caller's
        # sandbox; that's the only worker whose drain affects this
        # sync's correctness. Fencing every worker (the original
        # design) made an unrelated sandbox's burst on a different
        # bucket block this sync until its 5s timeout.
        with self._sync_lock:
            timing = self._sync_waiters.get(sync_id)
        if timing is None:
            return True
        # Stamp ack_ts and carry the helper-side counters across to
        # sync() for its latency breakdown. ack_ts must be stamped
        # before the barrier is scheduled so sync() observes the right
        # helper-vs-worker split even on very fast drains.
        timing.ack_ts = time.monotonic()
        raw_drain = payload.get("drain_us")
        raw_events = payload.get("events")
        try:
            if raw_drain is not None:
                timing.helper_drain_us = int(raw_drain)
            if raw_events is not None:
                timing.helper_events = int(raw_events)
        except (TypeError, ValueError):
            pass
        if self._worker_pool is not None and timing.sandbox_id is not None:
            target_depth, peer_max, peer_sum, peer_count = self._worker_pool.barrier_for_sandbox(
                timing.sandbox_id, sync_id, timing.waiter
            )
            timing.target_depth = target_depth
            timing.peer_depth_max = peer_max
            timing.peer_depth_sum = peer_sum
            timing.peer_count = peer_count
        else:
            timing.waiter.set()
        return True

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        try:
            for line in self._process.stderr:
                logger.debug("fs helper: %s", line.rstrip())
        except ValueError:
            logger.debug("filesystem monitor stderr closed during shutdown")

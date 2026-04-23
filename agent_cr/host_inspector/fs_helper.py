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


# Size of the per-sandbox worker pool. Event processing requires the
# sandbox-specific lock in the daemon, so parallelism is bounded by the
# number of distinct sandboxes — 8 workers comfortably covers 128
# sandboxes while keeping context-switch overhead low.
_DEFAULT_EVENT_WORKER_COUNT = 8


@dataclass
class _SyncTiming:
    """Per-sync() instrumentation record.

    `sent_ts` is stamped by sync() before writing the command to stdin.
    `ack_ts` is stamped by the reader thread the moment the matching
    sync_ack line is parsed. `helper_drain_us` / `helper_events` carry
    the counters the helper attached to its sync_ack (see emit_sync_ack
    in fs_monitor.c). `waiter` is set by the last worker to reach the
    barrier — at which point every event emitted before sync_ack has
    been applied. sync() reads these after waiter.wait() to log a
    latency breakdown (helper vs. Python-worker drain).

    `queue_depths` captures each worker queue's size at the moment the
    barrier was enqueued — the smoking gun for "one worker has a huge
    backlog, all others idle" shape.
    """

    waiter: Event
    sent_ts: float = 0.0
    ack_ts: float = 0.0
    helper_drain_us: int = 0
    helper_events: int = 0
    queue_depths: list[int] = field(default_factory=list)


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
    """Partition-parallel event dispatch.

    The reader thread no longer calls on_event inline — it parses each
    event and drops it into a per-sandbox queue keyed by
    hash(sandbox_id) % N. Workers consume from their own queues so that
    events for different sandboxes process concurrently, and events for
    the same sandbox remain strictly ordered (required: the daemon's
    state updates are not commutative).
    """

    def __init__(
        self,
        on_event: Callable[[HelperEvent], None],
        *,
        worker_count: int = _DEFAULT_EVENT_WORKER_COUNT,
    ) -> None:
        self._on_event = on_event
        self._worker_count = max(1, int(worker_count))
        self._queues: list[queue.Queue[object]] = [queue.Queue() for _ in range(self._worker_count)]
        self._threads: list[Thread] = []
        for idx in range(self._worker_count):
            thread = Thread(
                target=self._run_worker,
                args=(idx,),
                name=f"host-inspector-fs-event-worker-{idx}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def dispatch(self, event: HelperEvent) -> None:
        idx = hash(event.sandbox_id) % self._worker_count
        self._queues[idx].put(event)

    def barrier(self, sync_id: int, waiter: Event) -> list[int]:
        """Enqueue a fence on every worker queue; `waiter` is set once
        every worker has reached it. Returns the queue depths snapshot
        (in worker-index order) at the moment the barriers were
        enqueued — callers use this to diagnose backlog skew that
        causes the barrier to wait past the sync timeout.
        """
        marker = _Barrier(
            sync_id=sync_id,
            counter=[self._worker_count],
            waiter=waiter,
            lock=Lock(),
        )
        depths: list[int] = []
        for q in self._queues:
            # qsize() is approximate under concurrent producers, but
            # this is diagnostic output only; the exact count isn't
            # load-bearing.
            depths.append(q.qsize())
            q.put(marker)
        return depths

    def stop(self) -> None:
        for q in self._queues:
            q.put(None)
        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads = []

    def _run_worker(self, idx: int) -> None:
        q = self._queues[idx]
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
                logger.exception("Error applying fs event in worker %d", idx)


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
        self._send({"op": "upsert_sandbox", "sandbox_id": sandbox_id, "cgroup_id": cgroup_id})

    def remove_sandbox(self, sandbox_id: str) -> None:
        self._send({"op": "remove_sandbox", "sandbox_id": sandbox_id})

    def add_ignored_pid(self, pid: int) -> None:
        if pid <= 0:
            return
        self._send({"op": "add_ignored_pid", "pid": int(pid)})

    def remove_ignored_pid(self, pid: int) -> None:
        if pid <= 0:
            return
        self._send({"op": "remove_ignored_pid", "pid": int(pid)})

    def sync(self, timeout_s: float = 2.0) -> bool:
        """Drain events currently in flight in the kernel ring buffer and
        wait until each has been delivered to `on_event`.

        Returns True on success, False if the helper is not running or the
        ack did not arrive before `timeout_s` elapsed.

        The kernel submits fs_event records to the ring buffer at syscall
        exit, but the userspace helper consumes them serially. The reader
        thread then dispatches events to a per-sandbox worker pool so that
        events for different sandboxes process concurrently. Because
        per-sandbox ordering must be preserved (the daemon state updates
        are not commutative), sync uses a cross-worker barrier: when the
        reader sees the helper's sync_ack it enqueues a fence behind all
        events already dispatched; sync() returns when every worker has
        reached its fence.
        """
        if self._process is None or self._process.stdin is None:
            return False
        sync_id = next(self._sync_counter)
        timing = _SyncTiming(waiter=Event())
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
          - worker_ms: sync_ack parsed → last barrier fires. Dominated
            by Python per-event daemon work and any queue backlog on
            the slowest worker's queue.
        """
        total_ms = (done_ts - timing.sent_ts) * 1000.0
        if timing.ack_ts > 0.0:
            helper_ms_str = f"{(timing.ack_ts - timing.sent_ts) * 1000.0:.1f}"
            worker_ms_str = f"{(done_ts - timing.ack_ts) * 1000.0:.1f}"
        else:
            helper_ms_str = "n/a"
            worker_ms_str = "n/a"
        drain_ms_str = f"{timing.helper_drain_us / 1000.0:.1f}" if timing.ack_ts > 0.0 else "n/a"
        if timing.queue_depths:
            depths_str = "[" + ",".join(str(d) for d in timing.queue_depths) + "]"
            depths_max = max(timing.queue_depths)
            depths_sum = sum(timing.queue_depths)
        else:
            depths_str = "n/a"
            depths_max = 0
            depths_sum = 0
        if ok:
            logger.debug(
                "fs_sync id=%d ok=1 total_ms=%.1f helper_ms=%s helper_drain_ms=%s "
                "helper_events=%d worker_ms=%s q_max=%d q_sum=%d q=%s",
                sync_id,
                total_ms,
                helper_ms_str,
                drain_ms_str,
                timing.helper_events,
                worker_ms_str,
                depths_max,
                depths_sum,
                depths_str,
            )
        else:
            logger.warning(
                "fs_sync id=%d TIMEOUT timeout_s=%.1f total_ms=%.1f helper_ms=%s "
                "helper_drain_ms=%s helper_events=%d worker_ms=%s q_max=%d q_sum=%d q=%s",
                sync_id,
                timeout_s,
                total_ms,
                helper_ms_str,
                drain_ms_str,
                timing.helper_events,
                worker_ms_str,
                depths_max,
                depths_sum,
                depths_str,
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
        # Enqueue a barrier on every worker queue; when all workers have
        # processed it, they set the waiter — meaning all events emitted
        # before sync_ack have been applied.
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
        if self._worker_pool is not None:
            timing.queue_depths = self._worker_pool.barrier(sync_id, timing.waiter)
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

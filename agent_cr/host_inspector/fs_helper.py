from __future__ import annotations

import logging
import os
import subprocess
from threading import Lock, Thread
from typing import Callable

from .protocol import HelperEvent, decode_event, encode_command

logger = logging.getLogger(__name__)


class LibbpfFilesystemMonitor:
    def __init__(self, helper_path: str | None = None) -> None:
        default_path = os.path.join(os.path.dirname(__file__), "bpf", "fs_monitor")
        self._helper_path = helper_path or default_path
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: Thread | None = None
        self._stderr_thread: Thread | None = None
        self._stdin_lock = Lock()
        self._on_event: Callable[[HelperEvent], None] | None = None

    def start(self, on_event: Callable[[HelperEvent], None]) -> None:
        if self._process is not None:
            return
        self._on_event = on_event
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
        process.terminate()
        process.wait(timeout=5.0)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1.0)
            self._stdout_thread = None
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1.0)
            self._stderr_thread = None

    def upsert_sandbox(self, sandbox_id: str, cgroup_id: int) -> None:
        self._send({"op": "upsert_sandbox", "sandbox_id": sandbox_id, "cgroup_id": cgroup_id})

    def remove_sandbox(self, sandbox_id: str) -> None:
        self._send({"op": "remove_sandbox", "sandbox_id": sandbox_id})

    def _send(self, payload: dict[str, object]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("filesystem monitor helper is not running")
        with self._stdin_lock:
            self._process.stdin.write(encode_command(payload))
            self._process.stdin.flush()

    def _read_stdout(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        for line in self._process.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = decode_event(line)
            except Exception:
                logger.exception("Failed to decode fs helper event: %s", line)
                continue
            if self._on_event is not None:
                self._on_event(event)

    def _read_stderr(self) -> None:
        assert self._process is not None and self._process.stderr is not None
        for line in self._process.stderr:
            logger.debug("fs helper: %s", line.rstrip())

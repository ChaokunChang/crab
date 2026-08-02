from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Callable

from crab import SandboxId, TelemetrySink

logger = logging.getLogger(__name__)


def _safe_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _parse_key_value_lines(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, _, raw_value = line.partition(":")
        if not key or not raw_value:
            continue
        number = raw_value.strip().split()[0]
        try:
            values[key.strip()] = int(number)
        except ValueError:
            continue
    return values


def _parse_proc_net_dev(text: str) -> tuple[int, int]:
    rx_total = 0
    tx_total = 0
    for line in text.splitlines()[2:]:
        iface, _, payload = line.partition(":")
        if not payload:
            continue
        iface = iface.strip()
        if iface == "lo":
            continue
        fields = payload.split()
        if len(fields) < 16:
            continue
        try:
            rx_total += int(fields[0])
            tx_total += int(fields[8])
        except ValueError:
            continue
    return rx_total, tx_total


def _parse_cpu_stat(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("usage_usec "):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def _parse_io_stat(text: str) -> tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    for line in text.splitlines():
        parts = line.split()
        for part in parts[1:]:
            key, _, raw_value = part.partition("=")
            if key == "rbytes":
                try:
                    read_bytes += int(raw_value)
                except ValueError:
                    continue
            elif key == "wbytes":
                try:
                    write_bytes += int(raw_value)
                except ValueError:
                    continue
    return read_bytes, write_bytes


def _parse_proc_io(text: str) -> tuple[int, int]:
    values = _parse_key_value_lines(text)
    return int(values.get("rchar", 0)), int(values.get("wchar", 0))


def _parse_proc_cgroup_path(text: str) -> str | None:
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        path = parts[2].strip()
        if path and path != "/":
            return path
    return None


def _parse_proc_stat_cpu(text: str) -> tuple[int, int] | None:
    for line in text.splitlines():
        if not line.startswith("cpu "):
            continue
        fields = line.split()[1:]
        try:
            values = [int(item) for item in fields]
        except ValueError:
            return None
        if len(values) < 4:
            return None
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return total - idle, total
    return None


def _parse_diskstats(text: str) -> tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 14:
            continue
        device = fields[2]
        if device.startswith(("loop", "ram", "fd")):
            continue
        try:
            read_bytes += int(fields[5]) * 512
            write_bytes += int(fields[9]) * 512
        except ValueError:
            continue
    return read_bytes, write_bytes


class BenchmarkResourceMonitor:
    def __init__(
        self,
        *,
        telemetry: TelemetrySink,
        runtime,
        sandboxes: Callable[[], list],
        sample_interval_ms: int = 1000,
        include_host: bool = True,
        include_sandboxes: bool = True,
    ) -> None:
        self._telemetry = telemetry
        self._runtime = runtime
        self._sandboxes = sandboxes
        self._sample_interval_s = max(0.1, float(sample_interval_ms) / 1000.0)
        self._include_host = include_host
        self._include_sandboxes = include_sandboxes
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_host_cpu: tuple[float, int, int] | None = None
        self._last_sandbox_cpu: dict[str, tuple[float, int]] = {}
        self._sandbox_cgroup_paths: dict[str, str | None] = {}

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="benchmark-resource-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._sample_interval_s * 4.0))
            self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._sample_interval_s):
            try:
                self.sample_once()
            except Exception:
                logger.exception("Resource monitor sample failed")

    def sample_once(self) -> None:
        if self._include_host:
            self._sample_host()
        if self._include_sandboxes:
            for sandbox in self._sandboxes():
                self._sample_sandbox(sandbox)

    def _emit_metric(self, name: str, value: float, *, sandbox_id: SandboxId | None = None, extra: dict[str, object] | None = None) -> None:
        attributes: dict[str, object] = {"component": "monitoring"}
        if sandbox_id is not None:
            attributes["sandbox_id"] = str(sandbox_id)
        if extra:
            attributes.update(extra)
        self._telemetry.emit_metric(name, float(value), attributes)

    def _sample_host(self) -> None:
        proc_stat = _safe_read_text(Path("/proc/stat"))
        meminfo = _safe_read_text(Path("/proc/meminfo"))
        diskstats = _safe_read_text(Path("/proc/diskstats"))
        net_dev = _safe_read_text(Path("/proc/net/dev"))
        if proc_stat:
            current = _parse_proc_stat_cpu(proc_stat)
            now = time.monotonic()
            if current is not None:
                busy, total = current
                if self._last_host_cpu is not None:
                    _, last_busy, last_total = self._last_host_cpu
                    total_delta = total - last_total
                    busy_delta = busy - last_busy
                    usage_percent = 0.0 if total_delta <= 0 else (busy_delta / total_delta) * 100.0
                    self._emit_metric("resource.host.cpu.usage_percent", usage_percent)
                self._last_host_cpu = (now, busy, total)
        if meminfo:
            values = _parse_key_value_lines(meminfo)
            total_kb = values.get("MemTotal")
            available_kb = values.get("MemAvailable")
            if total_kb is not None:
                self._emit_metric("resource.host.memory.total_bytes", total_kb * 1024)
            if available_kb is not None and total_kb is not None:
                self._emit_metric("resource.host.memory.used_bytes", (total_kb - available_kb) * 1024)
        if diskstats:
            read_bytes, write_bytes = _parse_diskstats(diskstats)
            self._emit_metric("resource.host.disk.read_bytes", read_bytes)
            self._emit_metric("resource.host.disk.write_bytes", write_bytes)
        if net_dev:
            rx_bytes, tx_bytes = _parse_proc_net_dev(net_dev)
            self._emit_metric("resource.host.network.rx_bytes", rx_bytes)
            self._emit_metric("resource.host.network.tx_bytes", tx_bytes)

    def _sample_sandbox(self, sandbox) -> None:
        sandbox_id = sandbox.sandbox_id
        cgroup_path = self._sandbox_cgroup_path(sandbox)
        if cgroup_path and not self._cgroup_path_is_readable(cgroup_path):
            fallback_cgroup_path = self._runtime_cgroup_path(sandbox_id)
            if fallback_cgroup_path:
                cgroup_path = fallback_cgroup_path
                self._sandbox_cgroup_paths[str(sandbox_id)] = fallback_cgroup_path
        if not cgroup_path:
            return
        self._sample_sandbox_cgroup_metrics(sandbox_id, cgroup_path)
        pids = self._sandbox_process_ids(cgroup_path)
        if not pids:
            return
        self._sample_sandbox_network_metrics(sandbox_id, pids)
        self._sample_sandbox_filesystem_metrics(sandbox_id, pids)

    def _sandbox_cgroup_path(self, sandbox) -> str | None:
        cache_key = str(sandbox.sandbox_id)
        if cache_key in self._sandbox_cgroup_paths:
            return self._sandbox_cgroup_paths[cache_key]
        config_path = Path(sandbox.bundle_dir) / "config.json"
        raw = _safe_read_text(config_path)
        if not raw:
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None
        linux_cfg = payload.get("linux", {})
        if not isinstance(linux_cfg, dict):
            return None
        cgroup_path = linux_cfg.get("cgroupsPath")
        resolved = cgroup_path if isinstance(cgroup_path, str) and cgroup_path else None
        self._sandbox_cgroup_paths[cache_key] = resolved
        return resolved

    def _sandbox_process_ids(self, cgroup_path: str) -> list[int]:
        procs_text = _safe_read_text(self._cgroup_root(cgroup_path) / "cgroup.procs")
        if not procs_text:
            return []
        seen: set[int] = set()
        pids: list[int] = []
        for raw_pid in procs_text.splitlines():
            raw_pid = raw_pid.strip()
            if not raw_pid:
                continue
            try:
                pid = int(raw_pid)
            except ValueError:
                continue
            if pid <= 0 or pid in seen:
                continue
            seen.add(pid)
            pids.append(pid)
        return pids

    def _cgroup_root(self, cgroup_path: str) -> Path:
        return Path("/sys/fs/cgroup") / cgroup_path.lstrip("/")

    def _cgroup_path_is_readable(self, cgroup_path: str) -> bool:
        root = self._cgroup_root(cgroup_path)
        for relative_path in ("cgroup.procs", "cpu.stat", "memory.current", "memory.peak", "io.stat"):
            if _safe_read_text(root / relative_path) is not None:
                return True
        return False

    def _runtime_cgroup_path(self, sandbox_id: SandboxId) -> str | None:
        inspect_runtime = getattr(self._runtime, "inspect_runtime", None)
        if not callable(inspect_runtime):
            return None
        try:
            runtime_state = inspect_runtime(sandbox_id)
        except Exception:
            logger.debug("Failed to inspect runtime while resolving cgroup path", exc_info=True)
            return None
        pid = getattr(runtime_state, "pid", None)
        if not isinstance(pid, int) or pid <= 0:
            return None
        proc_cgroup = _safe_read_text(Path(f"/proc/{pid}/cgroup"))
        if not proc_cgroup:
            return None
        return _parse_proc_cgroup_path(proc_cgroup)

    def _sample_sandbox_cgroup_metrics(self, sandbox_id: SandboxId, cgroup_path: str) -> None:
        root = self._cgroup_root(cgroup_path)
        cpu_stat = _safe_read_text(root / "cpu.stat")
        if cpu_stat:
            usage_usec = _parse_cpu_stat(cpu_stat)
            now = time.monotonic()
            if usage_usec is not None:
                last = self._last_sandbox_cpu.get(str(sandbox_id))
                if last is not None:
                    last_ts, last_usage_usec = last
                    elapsed_usec = (now - last_ts) * 1_000_000.0
                    usage_delta = usage_usec - last_usage_usec
                    usage_percent = 0.0 if elapsed_usec <= 0 else (usage_delta / elapsed_usec) * 100.0
                    self._emit_metric("resource.sandbox.cpu.usage_percent", usage_percent, sandbox_id=sandbox_id)
                self._last_sandbox_cpu[str(sandbox_id)] = (now, usage_usec)
                self._emit_metric("resource.sandbox.cpu.usage_usec", usage_usec, sandbox_id=sandbox_id)
        current_text = _safe_read_text(root / "memory.current")
        if current_text:
            try:
                self._emit_metric("resource.sandbox.memory.current_bytes", int(current_text.strip()), sandbox_id=sandbox_id)
            except ValueError:
                pass
        peak_text = _safe_read_text(root / "memory.peak")
        if peak_text:
            try:
                self._emit_metric("resource.sandbox.memory.peak_bytes", int(peak_text.strip()), sandbox_id=sandbox_id)
            except ValueError:
                pass
        io_stat = _safe_read_text(root / "io.stat")
        if io_stat:
            read_bytes, write_bytes = _parse_io_stat(io_stat)
            self._emit_metric("resource.sandbox.disk.read_bytes", read_bytes, sandbox_id=sandbox_id)
            self._emit_metric("resource.sandbox.disk.write_bytes", write_bytes, sandbox_id=sandbox_id)

    def _sample_sandbox_network_metrics(self, sandbox_id: SandboxId, pids: list[int]) -> None:
        for pid in pids:
            net_text = _safe_read_text(Path(f"/proc/{int(pid)}/net/dev"))
            if not net_text:
                continue
            rx_bytes, tx_bytes = _parse_proc_net_dev(net_text)
            self._emit_metric("resource.sandbox.network.rx_bytes", rx_bytes, sandbox_id=sandbox_id)
            self._emit_metric("resource.sandbox.network.tx_bytes", tx_bytes, sandbox_id=sandbox_id)
            return

    def _sample_sandbox_filesystem_metrics(self, sandbox_id: SandboxId, pids: list[int]) -> None:
        read_bytes = 0
        write_bytes = 0
        for pid in pids:
            io_text = _safe_read_text(Path(f"/proc/{pid}/io"))
            if not io_text:
                continue
            proc_read_bytes, proc_write_bytes = _parse_proc_io(io_text)
            read_bytes += proc_read_bytes
            write_bytes += proc_write_bytes
        self._emit_metric("resource.sandbox.filesystem.read_bytes", read_bytes, sandbox_id=sandbox_id)
        self._emit_metric("resource.sandbox.filesystem.write_bytes", write_bytes, sandbox_id=sandbox_id)

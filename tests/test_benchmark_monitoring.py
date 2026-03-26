from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent_cr import InMemoryTelemetrySink, SandboxId
from benchmarks.monitoring import resource


class BenchmarkMonitoringTests(unittest.TestCase):
    def test_parse_io_stat_sums_devices(self) -> None:
        read_bytes, write_bytes = resource._parse_io_stat("8:0 rbytes=10 wbytes=20\n8:16 rbytes=30 wbytes=40\n")
        self.assertEqual(read_bytes, 40)
        self.assertEqual(write_bytes, 60)

    def test_parse_proc_net_dev_ignores_loopback(self) -> None:
        rx_bytes, tx_bytes = resource._parse_proc_net_dev(
            "\n".join(
                [
                    "Inter-|   Receive                                                |  Transmit",
                    " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed",
                    " lo: 100 1 0 0 0 0 0 0 200 1 0 0 0 0 0 0",
                    " eth0: 300 1 0 0 0 0 0 0 400 1 0 0 0 0 0 0",
                ]
            )
        )
        self.assertEqual(rx_bytes, 300)
        self.assertEqual(tx_bytes, 400)

    def test_sample_once_emits_host_and_sandbox_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_monitoring_") as tmp:
            root = Path(tmp)
            bundle_dir = root / "bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            config_path = bundle_dir / "config.json"
            config_path.write_text('{"linux":{"cgroupsPath":"demo.scope"}}', encoding="utf-8")

            telemetry = InMemoryTelemetrySink()
            sandbox = SimpleNamespace(sandbox_id=SandboxId("sbx-1"), bundle_dir=bundle_dir)
            runtime = SimpleNamespace(inspect_runtime=Mock(side_effect=AssertionError("inspect_runtime should not be called")))
            monitor = resource.BenchmarkResourceMonitor(
                telemetry=telemetry,
                runtime=runtime,
                sandboxes=lambda: [sandbox],
                sample_interval_ms=1000,
                include_host=True,
                include_sandboxes=True,
            )
            config_reads = 0

            fake_files = {
                "/proc/stat": "cpu  1 2 3 4 5 6 7 8 9 10\n",
                "/proc/meminfo": "MemTotal: 2048 kB\nMemAvailable: 1024 kB\n",
                "/proc/diskstats": "8 0 sda 1 0 2 0 3 0 4 0 0 0 0 0 0 0 0 0 0 0\n",
                "/proc/net/dev": "\n".join(
                    [
                        "Inter-|   Receive                                                |  Transmit",
                        " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed",
                        " eth0: 300 1 0 0 0 0 0 0 400 1 0 0 0 0 0 0",
                    ]
                ),
                "/sys/fs/cgroup/demo.scope/cpu.stat": "usage_usec 50\n",
                "/sys/fs/cgroup/demo.scope/memory.current": "4096\n",
                "/sys/fs/cgroup/demo.scope/memory.peak": "8192\n",
                "/sys/fs/cgroup/demo.scope/io.stat": "8:0 rbytes=64 wbytes=128\n",
                "/sys/fs/cgroup/demo.scope/cgroup.procs": "999\n123\n123\n",
                "/proc/123/net/dev": "\n".join(
                    [
                        "Inter-|   Receive                                                |  Transmit",
                        " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed",
                        " eth0: 500 1 0 0 0 0 0 0 600 1 0 0 0 0 0 0",
                    ]
                ),
                "/proc/999/io": "rchar: 1\nwchar: 2\n",
                "/proc/123/io": "rchar: 700\nwchar: 900\n",
            }

            def _fake_read_text(path: Path) -> str | None:
                nonlocal config_reads
                if path == config_path:
                    config_reads += 1
                    return config_path.read_text(encoding="utf-8")
                return fake_files.get(str(path))

            with patch("benchmarks.monitoring.resource._safe_read_text", side_effect=_fake_read_text):
                monitor.sample_once()
                monitor.sample_once()

            metric_names = [name for name, _, _ in telemetry.metrics]
            self.assertIn("resource.host.cpu.usage_percent", metric_names)
            self.assertIn("resource.host.memory.used_bytes", metric_names)
            self.assertIn("resource.sandbox.cpu.usage_percent", metric_names)
            self.assertIn("resource.sandbox.disk.write_bytes", metric_names)
            self.assertIn("resource.sandbox.filesystem.write_bytes", metric_names)
            self.assertIn("resource.sandbox.network.tx_bytes", metric_names)
            self.assertEqual(config_reads, 1)
            self.assertEqual(
                [value for name, value, _ in telemetry.metrics if name == "resource.sandbox.network.tx_bytes"][-1],
                600.0,
            )
            self.assertEqual(
                [value for name, value, _ in telemetry.metrics if name == "resource.sandbox.filesystem.write_bytes"][-1],
                902.0,
            )
            runtime.inspect_runtime.assert_not_called()

    def test_sample_once_gracefully_handles_missing_sandbox_proc_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_monitoring_missing_") as tmp:
            root = Path(tmp)
            bundle_dir = root / "bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            config_path = bundle_dir / "config.json"
            config_path.write_text('{"linux":{"cgroupsPath":"demo.scope"}}', encoding="utf-8")

            telemetry = InMemoryTelemetrySink()
            sandbox = SimpleNamespace(sandbox_id=SandboxId("sbx-1"), bundle_dir=bundle_dir)
            monitor = resource.BenchmarkResourceMonitor(
                telemetry=telemetry,
                runtime=SimpleNamespace(),
                sandboxes=lambda: [sandbox],
                sample_interval_ms=1000,
                include_host=False,
                include_sandboxes=True,
            )

            def _fake_read_text(path: Path) -> str | None:
                if path == config_path:
                    return config_path.read_text(encoding="utf-8")
                if str(path).endswith("cpu.stat"):
                    return "usage_usec 10\n"
                if str(path).endswith("memory.current"):
                    return "10\n"
                if str(path).endswith("memory.peak"):
                    return "11\n"
                if str(path).endswith("io.stat"):
                    return "8:0 rbytes=1 wbytes=2\n"
                if str(path).endswith("cgroup.procs"):
                    return "123\n"
                return None

            with patch("benchmarks.monitoring.resource._safe_read_text", side_effect=_fake_read_text):
                monitor.sample_once()

            metric_names = [name for name, _, _ in telemetry.metrics]
            self.assertIn("resource.sandbox.memory.current_bytes", metric_names)
            self.assertNotIn("resource.sandbox.network.tx_bytes", metric_names)

    def test_sample_once_falls_back_to_runtime_pid_cgroup_when_bundle_cgroup_is_unreadable(self) -> None:
        with tempfile.TemporaryDirectory(prefix="agent_cr_monitoring_fallback_") as tmp:
            root = Path(tmp)
            bundle_dir = root / "bundle"
            bundle_dir.mkdir(parents=True, exist_ok=True)
            config_path = bundle_dir / "config.json"
            config_path.write_text('{"linux":{"cgroupsPath":"missing.scope"}}', encoding="utf-8")

            telemetry = InMemoryTelemetrySink()
            sandbox = SimpleNamespace(sandbox_id=SandboxId("sbx-1"), bundle_dir=bundle_dir)
            runtime = SimpleNamespace(inspect_runtime=Mock(return_value=SimpleNamespace(pid=321)))
            monitor = resource.BenchmarkResourceMonitor(
                telemetry=telemetry,
                runtime=runtime,
                sandboxes=lambda: [sandbox],
                sample_interval_ms=1000,
                include_host=False,
                include_sandboxes=True,
            )

            def _fake_read_text(path: Path) -> str | None:
                if path == config_path:
                    return config_path.read_text(encoding="utf-8")
                fake_files = {
                    "/proc/321/cgroup": "0::/resolved.scope\n",
                    "/sys/fs/cgroup/resolved.scope/cgroup.procs": "321\n",
                    "/sys/fs/cgroup/resolved.scope/cpu.stat": "usage_usec 50\n",
                    "/sys/fs/cgroup/resolved.scope/memory.current": "4096\n",
                    "/sys/fs/cgroup/resolved.scope/memory.peak": "8192\n",
                    "/sys/fs/cgroup/resolved.scope/io.stat": "8:0 rbytes=64 wbytes=128\n",
                    "/proc/321/net/dev": "\n".join(
                        [
                            "Inter-|   Receive                                                |  Transmit",
                            " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed",
                            " eth0: 500 1 0 0 0 0 0 0 600 1 0 0 0 0 0 0",
                        ]
                    ),
                    "/proc/321/io": "rchar: 700\nwchar: 900\n",
                }
                return fake_files.get(str(path))

            with patch("benchmarks.monitoring.resource._safe_read_text", side_effect=_fake_read_text):
                monitor.sample_once()
                monitor.sample_once()

            metric_names = [name for name, _, _ in telemetry.metrics]
            self.assertIn("resource.sandbox.cpu.usage_percent", metric_names)
            self.assertIn("resource.sandbox.memory.current_bytes", metric_names)
            self.assertIn("resource.sandbox.filesystem.write_bytes", metric_names)
            runtime.inspect_runtime.assert_called_once_with(SandboxId("sbx-1"))


if __name__ == "__main__":
    unittest.main()

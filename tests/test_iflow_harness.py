from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from crab import InMemoryTelemetrySink
from integrations.sandboxes.iflow.harness import prepare_iflow_runtime, prepare_iflow_state


class IFlowHarnessTests(unittest.TestCase):
    def test_prepare_iflow_runtime_reuses_shared_runtime_cache(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_iflow_runtime_") as tmp:
            root = Path(tmp)
            cache_dir = root / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "node-v22.18.0-linux-x64.tar.xz").write_bytes(b"node-cache")
            (cache_dir / "iflow-ai-iflow-cli-for-roll-0-4-4-v4.tgz").write_bytes(b"iflow-cli")
            alternate_node = root / "alternate-node"
            (alternate_node / "bin").mkdir(parents=True, exist_ok=True)
            (alternate_node / "bin" / "node").write_text("#!/bin/sh\n", encoding="utf-8")
            (alternate_node / "bin" / "npm").write_text("#!/bin/sh\n", encoding="utf-8")
            telemetry = InMemoryTelemetrySink()
            install_calls = 0

            def _fake_run(command, *, check, env, stdout, stderr):
                del check, env, stdout, stderr
                nonlocal install_calls
                install_calls += 1
                global_prefix = Path(command[4])
                (global_prefix / "bin").mkdir(parents=True, exist_ok=True)
                (global_prefix / "bin" / "iflow").write_text("#!/bin/sh\n", encoding="utf-8")
                entrypoint = (
                    global_prefix
                    / "lib"
                    / "node_modules"
                    / "@iflow-ai"
                    / "iflow-cli"
                    / "bundle"
                    / "entry.js"
                )
                entrypoint.parent.mkdir(parents=True, exist_ok=True)
                entrypoint.write_text("console.log('ok');\n", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            with patch("integrations.sandboxes.iflow.harness.subprocess.run", side_effect=_fake_run):
                runtime_a = prepare_iflow_runtime(
                    work_root=root / "sandbox-a",
                    cache_dir=cache_dir,
                    alternate_node_runtime_dir=alternate_node,
                    telemetry=telemetry,
                    sandbox_id="sbx-a",
                )
                runtime_b = prepare_iflow_runtime(
                    work_root=root / "sandbox-b",
                    cache_dir=cache_dir,
                    alternate_node_runtime_dir=alternate_node,
                    telemetry=telemetry,
                    sandbox_id="sbx-b",
                )

        self.assertEqual(install_calls, 1)
        self.assertEqual(runtime_a.root, runtime_b.root)
        self.assertEqual(runtime_a.node_root, runtime_b.node_root)
        self.assertEqual(runtime_a.global_prefix, runtime_b.global_prefix)
        self.assertEqual(runtime_a.root.parent.name, "prepared-runtimes")
        event_names = [name for name, _ in telemetry.events]
        metric_names = [name for name, _, _ in telemetry.metrics]
        self.assertIn("iflow.runtime.cache_miss", event_names)
        self.assertIn("iflow.runtime.cache_hit", event_names)
        self.assertIn("iflow.runtime.prepare.duration_ms", metric_names)
        self.assertIn("iflow.runtime.stage_node.duration_ms", metric_names)
        self.assertIn("iflow.runtime.install_cli.duration_ms", metric_names)
        self.assertIn("iflow.runtime.cache_lock_wait_ms", metric_names)

    def test_prepare_iflow_state_keeps_writable_state_per_sandbox(self) -> None:
        with tempfile.TemporaryDirectory(prefix="crab_iflow_state_") as tmp:
            root = Path(tmp)
            telemetry = InMemoryTelemetrySink()

            state_a = prepare_iflow_state(
                work_root=root / "sandbox-a",
                base_url="http://127.0.0.1:43123/v1",
                model_name="model-a",
                telemetry=telemetry,
                sandbox_id="sbx-a",
            )
            state_b = prepare_iflow_state(
                work_root=root / "sandbox-b",
                base_url="http://127.0.0.1:43123/v1",
                model_name="model-a",
                telemetry=telemetry,
                sandbox_id="sbx-b",
            )

            self.assertNotEqual(state_a.root, state_b.root)
            self.assertTrue((state_a.iflow_home / "settings.json").is_file())
            self.assertTrue((state_b.iflow_home / "settings.json").is_file())
            metric_names = [name for name, _, _ in telemetry.metrics]
            self.assertIn("iflow.state.prepare.duration_ms", metric_names)


if __name__ == "__main__":
    unittest.main()

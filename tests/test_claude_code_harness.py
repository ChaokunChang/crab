from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_cr import InMemoryTelemetrySink
from integrations.sandboxes.claude_code.harness import (
    CLAUDE_CODE_WRAPPER_ARG,
    prepare_claude_code_runtime,
    prepare_claude_code_state,
)


class _DummyHTTPResponse(io.BytesIO):
    def __enter__(self) -> "_DummyHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class ClaudeCodeHarnessTests(unittest.TestCase):
    def _write_binary(self, path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o755)

    def test_prepare_runtime_uses_requested_trace_agent_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions_dir = root / "versions"
            work_root = root / "work"
            self._write_binary(versions_dir / "2.1.34", b"trace-version")
            self._write_binary(versions_dir / "2.1.86", b"newer-version")

            with patch("integrations.sandboxes.claude_code.harness._claude_versions_dir", return_value=versions_dir):
                prepared = prepare_claude_code_runtime(work_root=work_root, requested_version="2.1.34")
                self.assertEqual(prepared.resolved_version, "2.1.34")
                self.assertEqual(prepared.runtime_strategy, "version_cache")
                self.assertEqual(prepared.source_binary.read_bytes(), b"trace-version")
                self.assertEqual(prepared.claude_bin.read_bytes(), b"trace-version")

    def test_prepare_runtime_env_version_override_wins_over_trace_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions_dir = root / "versions"
            work_root = root / "work"
            self._write_binary(versions_dir / "2.1.34", b"trace-version")
            self._write_binary(versions_dir / "2.1.86", b"env-version")

            with patch("integrations.sandboxes.claude_code.harness._claude_versions_dir", return_value=versions_dir), patch.dict(
                os.environ,
                {"AGENT_CR_CLAUDE_CODE_VERSION": "2.1.86"},
                clear=False,
            ):
                prepared = prepare_claude_code_runtime(work_root=work_root, requested_version="2.1.34")
                self.assertEqual(prepared.resolved_version, "2.1.86")
                self.assertEqual(prepared.source_binary.read_bytes(), b"env-version")
                self.assertEqual(prepared.claude_bin.read_bytes(), b"env-version")

    def test_prepare_runtime_downloads_missing_version_from_url_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions_dir = root / "versions"
            work_root = root / "work"

            with patch("integrations.sandboxes.claude_code.harness._claude_versions_dir", return_value=versions_dir), patch.dict(
                os.environ,
                {"AGENT_CR_CLAUDE_CODE_BINARY_URL_TEMPLATE": "https://example.test/{version}/claude"},
                clear=False,
            ), patch(
                "integrations.sandboxes.claude_code.harness.urllib.request.urlopen",
                return_value=_DummyHTTPResponse(b"downloaded-version"),
            ):
                prepared = prepare_claude_code_runtime(work_root=work_root, requested_version="2.1.34")
                self.assertEqual(prepared.resolved_version, "2.1.34")
                self.assertEqual(prepared.runtime_strategy, "downloaded_version")
                self.assertTrue((versions_dir / "2.1.34").exists())
                self.assertEqual((versions_dir / "2.1.34").read_bytes(), b"downloaded-version")
                self.assertEqual(prepared.claude_bin.read_bytes(), b"downloaded-version")
                self.assertTrue(os.access(versions_dir / "2.1.34", os.X_OK))

    def test_prepare_runtime_raises_clear_error_when_version_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions_dir = root / "versions"
            work_root = root / "work"

            with patch("integrations.sandboxes.claude_code.harness._claude_versions_dir", return_value=versions_dir):
                with self.assertRaises(FileNotFoundError) as ctx:
                    prepare_claude_code_runtime(work_root=work_root, requested_version="2.1.34")

        self.assertIn("2.1.34", str(ctx.exception))

    def test_prepare_runtime_emits_setup_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions_dir = root / "versions"
            work_root = root / "work"
            telemetry = InMemoryTelemetrySink()
            self._write_binary(versions_dir / "2.1.34", b"trace-version")

            with patch("integrations.sandboxes.claude_code.harness._claude_versions_dir", return_value=versions_dir):
                prepare_claude_code_runtime(
                    work_root=work_root,
                    requested_version="2.1.34",
                    telemetry=telemetry,
                    sandbox_id="sbx-claude",
                )

        event_names = [name for name, _ in telemetry.events]
        metric_names = [name for name, _, _ in telemetry.metrics]
        self.assertIn("claude_code.runtime.prepare.start", event_names)
        self.assertIn("claude_code.runtime.prepare.finish", event_names)
        self.assertIn("claude_code.runtime.prepare.duration_ms", metric_names)
        finish_payload = next(
            attributes for name, attributes in telemetry.events if name == "claude_code.runtime.prepare.finish"
        )
        self.assertEqual(finish_payload["sandbox_id"], "sbx-claude")
        self.assertEqual(finish_payload["runtime_strategy"], "version_cache")
        self.assertEqual(finish_payload["resolved_version"], "2.1.34")

    def test_prepare_state_creates_mounted_home_and_logs_under_host_work_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_root = root / "work"

            prepared = prepare_claude_code_state(
                work_root=work_root,
                base_url="http://127.0.0.1:8080",
                model_name="claude-opus-4-6",
            )

            self.assertEqual(prepared.root, work_root / "claude-code-state")
            self.assertEqual(prepared.home_root, work_root / "claude-code-state" / "home")
            self.assertEqual(prepared.claude_home, work_root / "claude-code-state" / "home" / ".claude")
            self.assertEqual(prepared.logs_dir, work_root / "claude-code-state" / "logs")
            self.assertTrue((prepared.claude_home / "settings.json").is_file())

    def test_prepare_state_emits_setup_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work_root = root / "work"
            telemetry = InMemoryTelemetrySink()

            prepare_claude_code_state(
                work_root=work_root,
                base_url="http://127.0.0.1:8080",
                model_name="claude-opus-4-6",
                telemetry=telemetry,
                sandbox_id="sbx-claude",
            )

        event_names = [name for name, _ in telemetry.events]
        metric_names = [name for name, _, _ in telemetry.metrics]
        self.assertIn("claude_code.state.prepare.start", event_names)
        self.assertIn("claude_code.state.prepare.finish", event_names)
        self.assertIn("claude_code.state.prepare.duration_ms", metric_names)
        finish_payload = next(
            attributes for name, attributes in telemetry.events if name == "claude_code.state.prepare.finish"
        )
        self.assertEqual(finish_payload["sandbox_id"], "sbx-claude")
        self.assertEqual(finish_payload["model_name"], "claude-opus-4-6")

    def test_prepare_runtime_ignores_wrapper_shell_and_claude_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            versions_dir = root / "versions"
            work_root = root / "work"
            self._write_binary(versions_dir / "2.1.34", b"trace-version")

            with patch("integrations.sandboxes.claude_code.harness._claude_versions_dir", return_value=versions_dir):
                prepared = prepare_claude_code_runtime(work_root=work_root, requested_version="2.1.34")

            self.assertEqual(
                prepared.ignore_process_rules,
                [
                    {
                        "executable_basename": "claude",
                        "cmdline_contains": [prepared.mounted_claude_bin],
                    },
                    {
                        "cmdline_contains": [CLAUDE_CODE_WRAPPER_ARG],
                    },
                ],
            )


if __name__ == "__main__":
    unittest.main()

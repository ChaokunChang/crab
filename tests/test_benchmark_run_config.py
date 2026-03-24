from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmarks.config import BenchmarkConfig, load_config
from benchmarks.core import resolve_task_records
from benchmarks.run import run_benchmark_config
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition
from integrations.agents import TaskConfig, TaskDescription


class BenchmarkConfigTests(unittest.TestCase):
    def test_load_config_resolves_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "task_dataset: datasets/tasks.jsonl",
                        "sandboxes: 4",
                        "output: results/out.csv",
                        "log_file: results/out.log",
                        "max_workers: 3",
                        "log_file_mode: write",
                        "benchmark_root: benchmark-runs",
                        "zpool_size: 32G",
                        "zpool_name: benchcache",
                        "zpool_image: cache/bench.zpool.img",
                        "reuse_zpool: true",
                        "telemetry_output: results/out.telemetry.jsonl",
                        "image_cache_root: cache/images",
                        "work_dir_host_root: workdirs",
                        "scenario_options:",
                        "  injection_rate: 0.25",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(config.scenario, "fault")
            self.assertEqual(config.mode, "auto")
            self.assertEqual(config.task_dataset, (root / "datasets" / "tasks.jsonl").resolve())
            self.assertEqual(config.output, (root / "results" / "out.csv").resolve())
            self.assertEqual(config.log_file, (root / "results" / "out.log").resolve())
            self.assertEqual(config.max_workers, 3)
            self.assertEqual(config.effective_max_workers, 3)
            self.assertEqual(config.effective_phase_workers.as_dict(), {"build": 3, "prepare": 3, "run": 3, "verification": 3})
            self.assertEqual(config.log_file_mode, "write")
            self.assertEqual(config.benchmark_root, (root / "benchmark-runs").resolve())
            self.assertEqual(config.zpool_size, "32G")
            self.assertEqual(config.zpool_name, "benchcache")
            self.assertEqual(config.zpool_image, (root / "cache" / "bench.zpool.img").resolve())
            self.assertTrue(config.reuse_zpool)
            self.assertEqual(config.telemetry_output, (root / "results" / "out.telemetry.jsonl").resolve())
            self.assertEqual(config.image_cache_root, (root / "cache" / "images").resolve())
            self.assertEqual(config.work_dir_host_root, (root / "workdirs").resolve())
            self.assertEqual(config.scenario_options["injection_rate"], 0.25)

    def test_load_config_supports_nested_telemetry_block(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "telemetry:",
                        "  output: results/nested.telemetry.jsonl",
                        "  detail_level: detailed",
                        "  capture_command_output: true",
                        "  max_text_attribute_bytes: 512",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                config.telemetry_output,
                (root / "results" / "nested.telemetry.jsonl").resolve(),
            )
            self.assertEqual(config.telemetry_detail_level, "detailed")
            self.assertTrue(config.telemetry_capture_command_output)
            self.assertEqual(config.telemetry_max_text_attribute_bytes, 512)

    def test_load_config_nested_telemetry_output_overrides_legacy_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "telemetry_output: results/legacy.telemetry.jsonl",
                        "telemetry:",
                        "  output: results/nested.telemetry.jsonl",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

            self.assertEqual(
                config.telemetry_output,
                (root / "results" / "nested.telemetry.jsonl").resolve(),
            )

    def test_resolve_task_records_merges_llm_service_options_without_overriding_dataset_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_path = root / "tasks.jsonl"
            dataset_path.write_text(
                "\n".join(
                    [
                        (
                            '{"agent_type":"iflow","llm_service_type":"iflow_trace_replay",'
                            '"task_description":{"prompt":"task-a"},"task_config":{},'
                            '"llm_service_config":{"trace_path":"trace-a.log","response_delay_policy":"fixed"}}'
                        )
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config_path = root / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "agent: iflow",
                        "llm_service: iflow_trace_replay",
                        "task_dataset: tasks.jsonl",
                        "llm_service_options:",
                        "  response_delay_policy: trace_replay",
                        "  response_delay_scaling_factor: 0.5",
                        "  response_delay_ms: 77",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)
            records = resolve_task_records(
                config,
                default_task_description=TaskDescription("ignored"),
                default_task_config=TaskConfig(),
            )

        self.assertEqual(
            config.llm_service_options,
            {
                "response_delay_policy": "trace_replay",
                "response_delay_scaling_factor": 0.5,
                "response_delay_ms": 77,
            },
        )
        self.assertEqual(
            records[0].llm_service_config,
            {
                "trace_path": str((root / "trace-a.log").resolve()),
                "response_delay_policy": "fixed",
                "response_delay_scaling_factor": 0.5,
                "response_delay_ms": 77,
            },
        )

    def test_load_config_rejects_invalid_log_file_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "log_file_mode: rotate",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "log_file_mode"):
                load_config(config_path)

    def test_load_config_rejects_non_positive_max_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "max_workers: 0",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "max_workers"):
                load_config(config_path)

    def test_load_config_parses_phase_workers_with_fallbacks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "sandboxes: 8",
                        "max_workers: 4",
                        "phase_workers:",
                        "  build: 2",
                        "  run: 3",
                    ]
                ),
                encoding="utf-8",
            )

            config = load_config(config_path)

        self.assertEqual(config.effective_phase_workers.as_dict(), {"build": 2, "prepare": 4, "run": 3, "verification": 4})

    def test_load_config_rejects_unknown_phase_worker_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "phase_workers:",
                        "  build: 1",
                        "  deploy: 2",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "unknown phases"):
                load_config(config_path)

    def test_load_config_rejects_non_positive_phase_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: fault",
                        "mode: auto",
                        "phase_workers:",
                        "  verification: 0",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "phase_workers.verification"):
                load_config(config_path)

    def test_load_config_rejects_invalid_scenario_mode_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bench.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "scenario: e2e",
                        "mode: auto",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "does not support"):
                load_config(config_path)


class BenchmarkRunDispatchTests(unittest.TestCase):
    def _config(self, mode: str) -> BenchmarkConfig:
        return BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode=mode,
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            task_dataset=None,
            sandboxes=1,
            iterations=1,
            output=None,
            telemetry_output=None,
            telemetry_detail_level="basic",
            telemetry_capture_command_output=False,
            telemetry_max_text_attribute_bytes=2048,
            log_file=None,
            log_file_mode="append",
            benchmark_root=None,
            zpool_size="10G",
            zpool_name=None,
            zpool_image=None,
            reuse_zpool=False,
            image_cache_root=None,
            log_level="info",
            transfer_delay_ms=0.0,
            work_dir_host_root=None,
            scenario_options={},
        )

    def test_run_benchmark_config_dispatches_selected_mode(self) -> None:
        calls: list[str] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

            def __enter__(self):
                return {"kind": "fake-harness", "kwargs": self.kwargs}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual", "auto"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: calls.append(f"manual:{config.mode}:{harness['kind']}") or [],
            run_auto=lambda config, harness: calls.append(f"auto:{config.mode}:{harness['kind']}") or [],
            summarize=lambda config, rows: {"success_ratio": float(len(rows))},
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                run_benchmark_config(self._config("manual"))
                run_benchmark_config(self._config("auto"))

        self.assertEqual(calls, ["manual:manual:fake-harness", "auto:auto:fake-harness"])
        self.assertIn("success_ratio_avg: 0.000", buffer.getvalue())

    def test_run_benchmark_config_defaults_telemetry_output_path(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode="manual",
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            output=Path("/tmp/out.csv"),
            telemetry_output=None,
            log_file=Path("/tmp/out.log"),
            log_file_mode="append",
            benchmark_root=Path("/tmp/bench-root"),
            zpool_size="10G",
            zpool_name=None,
            zpool_image=None,
            reuse_zpool=False,
            image_cache_root=None,
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertEqual(calls[0]["telemetry_output"], Path("/tmp/out.telemetry.jsonl"))
        self.assertEqual(calls[0]["telemetry_detail_level"], "basic")
        self.assertFalse(calls[0]["telemetry_capture_command_output"])
        self.assertEqual(calls[0]["telemetry_max_text_attribute_bytes"], 2048)
        self.assertEqual(calls[0]["benchmark_root"], Path("/tmp/bench-root"))

    def test_run_benchmark_config_passes_telemetry_yaml_settings_to_harness(self) -> None:
        calls: list[dict[str, object]] = []

        class _HarnessContext:
            def __init__(self, **kwargs) -> None:
                calls.append(kwargs)

            def __enter__(self):
                return {"kind": "fake-harness"}

            def __exit__(self, exc_type, exc, tb) -> None:
                _ = (exc_type, exc, tb)

        scenario = ScenarioDefinition(
            name="fake",
            supported_modes=frozenset({"manual"}),
            build_harness_settings=lambda config: HarnessSettings(
                scheduler_config={"mode": config.mode},
                scheduler_policy=None,
                checkpoint_manager_factory=lambda base: base,
                max_workers=1,
            ),
            run_manual=lambda config, harness: [],
            run_auto=None,
            summarize=lambda config, rows: {},
        )
        config = BenchmarkConfig(
            config_path=Path("/tmp/bench.yaml"),
            scenario="fake",
            mode="manual",
            provider="openai",
            agent="simulated",
            llm_service="simulated",
            output=None,
            telemetry_output=Path("/tmp/out.telemetry.jsonl"),
            telemetry_detail_level="detailed",
            telemetry_capture_command_output=True,
            telemetry_max_text_attribute_bytes=512,
            log_file=None,
            log_file_mode="append",
            benchmark_root=None,
            zpool_size="10G",
            zpool_name=None,
            zpool_image=None,
            reuse_zpool=False,
            image_cache_root=None,
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertEqual(calls[0]["telemetry_detail_level"], "detailed")
        self.assertTrue(calls[0]["telemetry_capture_command_output"])
        self.assertEqual(calls[0]["telemetry_max_text_attribute_bytes"], 512)

    def test_run_benchmark_config_logs_run_markers_and_summary_to_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_file = root / "benchmark.log"
            output = root / "out.csv"
            telemetry_output = root / "out.telemetry.jsonl"

            class _HarnessContext:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs

                def __enter__(self):
                    return {"kind": "fake-harness"}

                def __exit__(self, exc_type, exc, tb) -> None:
                    _ = (exc_type, exc, tb)

            scenario = ScenarioDefinition(
                name="fake",
                supported_modes=frozenset({"manual"}),
                build_harness_settings=lambda config: HarnessSettings(
                    scheduler_config={"mode": config.mode},
                    scheduler_policy=None,
                    checkpoint_manager_factory=lambda base: base,
                    max_workers=1,
                ),
                run_manual=lambda config, harness: [{"success_ratio": 1.0}],
                run_auto=None,
                summarize=lambda config, rows: {"success_ratio": 1.0},
            )
            config = BenchmarkConfig(
                config_path=root / "bench.yaml",
                scenario="fake",
                mode="manual",
                provider="openai",
                agent="simulated",
                llm_service="simulated",
                output=output,
                telemetry_output=telemetry_output,
                log_file=log_file,
                log_file_mode="write",
                benchmark_root=root / "bench-root",
                zpool_size="10G",
                zpool_name=None,
                zpool_image=None,
                reuse_zpool=False,
                image_cache_root=None,
            )

            with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
                "benchmarks.run.SCENARIOS",
                {"fake": scenario},
                clear=True,
            ):
                buffer = io.StringIO()
                with redirect_stdout(buffer):
                    run_benchmark_config(config)

            log_text = log_file.read_text(encoding="utf-8")
            self.assertIn("benchmark.run start", log_text)
            self.assertIn("max_workers=1", log_text)
            self.assertIn("success_ratio_avg: 1.000", log_text)
            self.assertIn(f"output: {output.resolve()}", log_text)
            self.assertIn("benchmark.run end status=completed", log_text)

    def test_example_yaml_files_load(self) -> None:
        examples = sorted((Path("/root/workspace/agent-cr/benchmarks/examples")).glob("*.yaml"))
        self.assertTrue(examples)
        for path in examples:
            with self.subTest(path=path.name):
                config = load_config(path)
                self.assertEqual(config.config_path, path.resolve())


if __name__ == "__main__":
    unittest.main()

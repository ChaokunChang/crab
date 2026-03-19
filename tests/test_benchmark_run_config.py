from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmarks.config import BenchmarkConfig, load_config
from benchmarks.run import run_benchmark_config
from benchmarks.scenarios import HarnessSettings, ScenarioDefinition


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
                        "output: results/out.csv",
                        "log_file: results/out.log",
                        "log_file_mode: write",
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
            self.assertEqual(config.log_file_mode, "write")
            self.assertEqual(config.telemetry_output, (root / "results" / "out.telemetry.jsonl").resolve())
            self.assertEqual(config.image_cache_root, (root / "cache" / "images").resolve())
            self.assertEqual(config.work_dir_host_root, (root / "workdirs").resolve())
            self.assertEqual(config.scenario_options["injection_rate"], 0.25)

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
            log_file=None,
            log_file_mode="append",
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
            image_cache_root=None,
        )

        with patch("benchmarks.run.RealHostScenarioHarness", _HarnessContext), patch.dict(
            "benchmarks.run.SCENARIOS",
            {"fake": scenario},
            clear=True,
        ):
            run_benchmark_config(config)

        self.assertEqual(calls[0]["telemetry_output"], Path("/tmp/out.telemetry.jsonl"))

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

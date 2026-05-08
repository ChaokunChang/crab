from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from benchmarks.run import _replicate_artifacts_to_benchmark_run_root


class BenchmarkRunArtifactReplicationTests(unittest.TestCase):
    def test_replicates_config_to_benchmark_run_root(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            benchmark_run_root = root / "benchmark-run"
            config_path = root / "example.yaml"
            output_path = root / "results.csv"
            config_path.write_text("scenario: fault\n", encoding="utf-8")
            output_path.write_text("sandbox_id,success_ratio\n", encoding="utf-8")
            context = type(
                "HarnessContext",
                (),
                {"root": benchmark_run_root, "uses_temporary_root": False},
            )()

            _replicate_artifacts_to_benchmark_run_root(
                harness_context=context,
                config_path=config_path,
                log_file=None,
                output=output_path,
                telemetry_output=None,
                telemetry_report_output_dir=None,
            )

            self.assertEqual(
                (benchmark_run_root / "example.yaml").read_text(encoding="utf-8"),
                "scenario: fault\n",
            )
            self.assertEqual(
                (benchmark_run_root / "results.csv").read_text(encoding="utf-8"),
                "sandbox_id,success_ratio\n",
            )


if __name__ == "__main__":
    unittest.main()

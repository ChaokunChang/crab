from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.generate_swebench_sweagent_replay_dataset import generate_dataset
from integrations.sandboxes import swebench as swebench_support


def _instance() -> dict[str, object]:
    return {
        "instance_id": "django__django-13820",
        "repo": "django/django",
        "version": "3.2",
        "base_commit": "98ad327864aed8df245fd19ea9d2743279e11643",
        "problem_statement": "Permit migrations in non-namespace packages that don't have __file__",
        "patch": "",
        "test_patch": (
            "diff --git a/tests/migrations/test_loader.py b/tests/migrations/test_loader.py\n"
            "--- a/tests/migrations/test_loader.py\n"
            "+++ b/tests/migrations/test_loader.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-print('a')\n"
            "+print('b')\n"
        ),
        "FAIL_TO_PASS": '["test_loading_package_without__file__ (migrations.test_loader.LoaderTests)"]',
        "PASS_TO_PASS": '["test_apply (migrations.test_loader.RecorderTests)"]',
    }


def _astropy_instance() -> dict[str, object]:
    return {
        "instance_id": "astropy__astropy-14096",
        "repo": "astropy/astropy",
        "version": "5.1",
        "base_commit": "1a4462d72eb03f30dc83a879b1dd57aac8b2c18b",
        "test_patch": (
            "diff --git a/astropy/coordinates/tests/test_sky_coord.py b/astropy/coordinates/tests/test_sky_coord.py\n"
            "--- a/astropy/coordinates/tests/test_sky_coord.py\n"
            "+++ b/astropy/coordinates/tests/test_sky_coord.py\n"
            "@@ -2165,3 +2165,21 @@ def test_match_to_catalog_3d_and_sky():\n"
            "     npt.assert_array_equal(idx, [0, 1, 2, 3])\n"
            "     assert_allclose(angle, 0 * u.deg, atol=1e-14 * u.deg, rtol=0)\n"
            "     assert_allclose(distance, 0 * u.kpc, atol=1e-14 * u.kpc, rtol=0)\n"
            "+\n"
            "+\n"
            "+def test_subclass_property_exception_error():\n"
            "+    \"\"\"Regression test for gh-8340.\n"
            "+\n"
            "+    Non-existing attribute access inside a property should give attribute\n"
            "+    error for the attribute, not for the property.\n"
            "+    \"\"\"\n"
            "+\n"
            "+    class custom_coord(SkyCoord):\n"
            "+        @property\n"
            "+        def prop(self):\n"
            "+            return self.random_attr\n"
            "+\n"
            "+    c = custom_coord(\"00h42m30s\", \"+41d12m00s\", frame=\"icrs\")\n"
            "+    with pytest.raises(AttributeError, match=\"random_attr\"):\n"
            "+        # Before this matched \"prop\" rather than \"random_attr\"\n"
            "+        c.prop\n"
        ),
        "FAIL_TO_PASS": '["astropy/coordinates/tests/test_sky_coord.py::test_subclass_property_exception_error"]',
        "PASS_TO_PASS": '["astropy/coordinates/tests/test_sky_coord.py::test_transform_to"]',
    }


class SWEbenchSupportTests(unittest.TestCase):
    def test_write_task_assets_renders_compose_and_exit_code_aware_run_tests(self) -> None:
        instance = _instance()
        with tempfile.TemporaryDirectory() as tmp:
            task_root = Path(tmp) / instance["instance_id"]
            swebench_support.write_task_assets(task_root=task_root, instance=instance)

            compose_text = (task_root / "docker-compose.yaml").read_text(encoding="utf-8")
            run_tests_text = (task_root / "run-tests.sh").read_text(encoding="utf-8")

        self.assertIn("exec tail -f /dev/null >/dev/null 2>/dev/null", compose_text)
        self.assertIn("swebench/sweb.eval.x86_64.django_1776_django-13820:latest", compose_text)
        self.assertIn("test_rc=0", run_tests_text)
        self.assertIn("echo '>>>>> Start Test Output'", run_tests_text)
        self.assertIn("echo '>>>>> End Test Output'", run_tests_text)
        self.assertIn('if [ "$test_rc" -ne 0 ]; then exit "$test_rc"; fi', run_tests_text)
        self.assertIn("migrations.test_loader", run_tests_text)

    def test_grade_verification_log_uses_targeted_swebench_resolution(self) -> None:
        instance = _astropy_instance()
        log_text = "\n".join(
            [
                ">>>>> Start Test Output",
                "PASSED astropy/coordinates/tests/test_sky_coord.py::test_transform_to",
                "PASSED astropy/coordinates/tests/test_sky_coord.py::test_subclass_property_exception_error",
                "FAILED astropy/coordinates/tests/test_sky_coord.py::test_repr_altaz - AstropyWarning: leap-second auto-update failed",
                ">>>>> End Test Output",
            ]
        )

        result = swebench_support.grade_verification_log(instance=instance, log_text=log_text)

        self.assertTrue(result["resolved"])
        self.assertTrue(result["patch_successfully_applied"])
        report = result["tests_status"]
        self.assertEqual(report["FAIL_TO_PASS"]["failure"], [])
        self.assertEqual(report["PASS_TO_PASS"]["failure"], [])

    def test_generate_dataset_builds_mini_swe_rows_and_task_assets(self) -> None:
        instance = _instance()
        trace_payload = {
            "messages": [
                {"role": "assistant", "content": "THOUGHT: one\n<mswea_bash_command>echo one</mswea_bash_command>"},
                {"role": "assistant", "content": "THOUGHT: two\n<mswea_bash_command>echo two</mswea_bash_command>"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traces_root = root / "traces"
            trace_dir = traces_root / str(instance["instance_id"])
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"{instance['instance_id']}.traj.json").write_text(json.dumps(trace_payload), encoding="utf-8")
            tasks_root = root / "tasks"
            output_path = root / "dataset.jsonl"

            with patch(
                "benchmarks.generate_swebench_sweagent_replay_dataset.swebench_support.load_verified_dataset_rows",
                return_value={str(instance["instance_id"]): instance},
            ):
                rows = generate_dataset(traces_root=traces_root, tasks_root=tasks_root, output_path=output_path)

            row = rows[0]
            task_root = tasks_root / str(instance["instance_id"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(row["agent_type"], "mini_swe")
            self.assertEqual(row["llm_service_type"], "mini_swe_trace_replay")
            self.assertEqual(row["trace_response_count"], 2)
            self.assertEqual(row["service_name"], "client")
            self.assertTrue((task_root / "docker-compose.yaml").exists())
            self.assertTrue((task_root / "run-tests.sh").exists())


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import yaml
from pyarrow import ipc
from swebench.harness.constants import (
    END_TEST_OUTPUT,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    START_TEST_OUTPUT,
)
from swebench.harness.docker_build import build_instance_images
from swebench.harness.grading import get_eval_report
from swebench.harness.test_spec.create_scripts import make_eval_script_list
from swebench.harness.test_spec.test_spec import make_test_spec

from integrations.sandboxes.runtime import image as sandbox_image

logger = logging.getLogger(__name__)

DEFAULT_DATASET_NAME = "princeton-nlp/SWE-Bench_Verified"
DEFAULT_DATASET_SPLIT = "test"
DEFAULT_TEST_TIMEOUT_S = 1800
DEFAULT_AGENT_TIMEOUT_S = 900
_DEFAULT_NAMESPACE = "swebench"
_DEFAULT_ARCH = "x86_64"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_HF_CACHE = _REPO_ROOT / ".cache" / "huggingface"
_HF_ARROW_GLOBS = (
    "datasets/princeton-nlp___swe-bench_verified/default/0.0.0/*/swe-bench_verified-test.arrow",
    "datasets/princeton-nlp___SWE-Bench_Verified/default/0.0.0/*/SWE-Bench_Verified-test.arrow",
)
_DATASET_ROW_CACHE: dict[str, dict[str, object]] | None = None


def remote_instance_image_ref(
    instance_id: str,
    *,
    arch: str = _DEFAULT_ARCH,
    namespace: str = _DEFAULT_NAMESPACE,
    tag: str = "latest",
) -> str:
    image_name = f"sweb.eval.{arch}.{instance_id.lower()}:{tag}"
    return f"{namespace}/{image_name}".replace("__", "_1776_")


def locate_verified_dataset_arrow(
    *,
    cache_roots: Iterable[Path] | None = None,
) -> Path | None:
    roots = list(cache_roots or [])
    if not roots:
        roots = [_LOCAL_HF_CACHE, Path.home() / ".cache" / "huggingface"]
    for root in roots:
        for pattern in _HF_ARROW_GLOBS:
            matches = sorted(root.glob(pattern))
            if matches:
                return matches[0]
    return None


def _load_rows_from_arrow(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    with path.open("rb") as handle:
        reader = ipc.open_stream(handle)
        for batch in reader:
            for row in batch.to_pylist():
                instance_id = row.get("instance_id")
                if isinstance(instance_id, str) and instance_id:
                    rows[instance_id] = dict(row)
    return rows


def load_verified_dataset_rows(
    *,
    cache_roots: Iterable[Path] | None = None,
) -> dict[str, dict[str, object]]:
    global _DATASET_ROW_CACHE
    if _DATASET_ROW_CACHE is not None:
        return dict(_DATASET_ROW_CACHE)
    arrow_path = locate_verified_dataset_arrow(cache_roots=cache_roots)
    if arrow_path is None:
        try:
            from datasets import load_dataset
        except Exception as exc:
            raise RuntimeError(
                "unable to locate cached SWE-Bench Verified dataset and datasets.load_dataset is unavailable"
            ) from exc
        dataset = load_dataset(
            DEFAULT_DATASET_NAME,
            split=DEFAULT_DATASET_SPLIT,
            cache_dir=str(_LOCAL_HF_CACHE),
        )
        rows = {
            str(row["instance_id"]): dict(row)
            for row in dataset
            if isinstance(row, dict) and isinstance(row.get("instance_id"), str)
        }
    else:
        rows = _load_rows_from_arrow(arrow_path)
    _DATASET_ROW_CACHE = dict(rows)
    return dict(rows)


def load_verified_dataset_row(instance_id: str) -> dict[str, object]:
    rows = load_verified_dataset_rows()
    try:
        return dict(rows[instance_id])
    except KeyError as exc:
        raise KeyError(f"SWE-Bench Verified instance {instance_id!r} was not found in the cached dataset") from exc


def _run_tests_script_lines(eval_commands: list[str]) -> list[str]:
    try:
        start_index = eval_commands.index(f": '{START_TEST_OUTPUT}'")
        end_index = eval_commands.index(f": '{END_TEST_OUTPUT}'")
    except ValueError as exc:
        raise ValueError("SWE-bench eval commands did not contain expected test output markers") from exc
    if end_index <= start_index:
        raise ValueError("SWE-bench eval commands contained malformed test output markers")
    pre_commands = eval_commands[:start_index]
    test_commands = eval_commands[start_index + 1 : end_index]
    post_commands = eval_commands[end_index + 1 :]
    lines = [
        "#!/bin/bash",
        "set -uxo pipefail",
        "test_rc=0",
        "post_rc=0",
        *pre_commands,
        f"echo '{START_TEST_OUTPUT}'",
        "set +e",
    ]
    for command in test_commands:
        lines.extend(
            [
                command,
                "cmd_rc=$?",
                'if [ "$cmd_rc" -ne 0 ] && [ "$test_rc" -eq 0 ]; then test_rc=$cmd_rc; fi',
            ]
        )
    lines.extend(
        [
            "set -e",
            f"echo '{END_TEST_OUTPUT}'",
        ]
    )
    if post_commands:
        lines.append("set +e")
        for command in post_commands:
            lines.extend(
                [
                    command,
                    "cmd_rc=$?",
                    'if [ "$cmd_rc" -ne 0 ] && [ "$post_rc" -eq 0 ]; then post_rc=$cmd_rc; fi',
                ]
            )
        lines.append("set -e")
    lines.extend(
        [
            'if [ "$test_rc" -ne 0 ]; then exit "$test_rc"; fi',
            'if [ "$post_rc" -ne 0 ]; then exit "$post_rc"; fi',
        ]
    )
    return lines


def render_run_tests_script(instance: dict[str, object]) -> str:
    repo = str(instance["repo"])
    version = str(instance["version"])
    specs = MAP_REPO_VERSION_TO_SPECS[repo][version]
    eval_commands = make_eval_script_list(
        instance,
        specs,
        "testbed",
        "/testbed",
        str(instance["base_commit"]),
        str(instance["test_patch"]),
    )
    return "\n".join(_run_tests_script_lines(eval_commands)) + "\n"


def grade_verification_log(
    *,
    instance: dict[str, object],
    log_text: str,
) -> dict[str, object]:
    instance_id = str(instance["instance_id"])
    test_spec = make_test_spec(instance, arch=_DEFAULT_ARCH)
    prediction = {
        KEY_INSTANCE_ID: instance_id,
        KEY_PREDICTION: "agent-cr-mini-swe-replay",
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".log", delete=False) as handle:
        handle.write(log_text)
        log_path = Path(handle.name)
    try:
        report_map = get_eval_report(
            test_spec,
            prediction,
            str(log_path),
            include_tests_status=True,
        )
    finally:
        log_path.unlink(missing_ok=True)
    report = dict(report_map.get(instance_id, {}))
    return {
        "resolved": bool(report.get("resolved", False)),
        "patch_successfully_applied": bool(report.get("patch_successfully_applied", False)),
        "tests_status": report.get("tests_status", {}),
    }


def write_task_assets(
    *,
    task_root: Path,
    instance: dict[str, object],
    image_ref: str | None = None,
) -> None:
    resolved_image_ref = image_ref or remote_instance_image_ref(str(instance["instance_id"]))
    task_root.mkdir(parents=True, exist_ok=True)
    compose_path = task_root / "docker-compose.yaml"
    run_tests_path = task_root / "run-tests.sh"
    instance_path = task_root / "instance.json"
    compose_payload = {
        "services": {
            "client": {
                "image": resolved_image_ref,
                "working_dir": "/testbed",
                "command": "exec tail -f /dev/null >/dev/null 2>/dev/null",
            }
        }
    }
    compose_path.write_text(yaml.safe_dump(compose_payload, sort_keys=False), encoding="utf-8")
    run_tests_path.write_text(render_run_tests_script(instance), encoding="utf-8")
    instance_path.write_text(json.dumps(instance, sort_keys=True, indent=2), encoding="utf-8")


def ensure_instance_image_available(
    instance_id: str,
    *,
    arch: str = _DEFAULT_ARCH,
    namespace: str = _DEFAULT_NAMESPACE,
    tag: str = "latest",
) -> str:
    image_ref = remote_instance_image_ref(instance_id, arch=arch, namespace=namespace, tag=tag)
    if sandbox_image.image_exists(tag=image_ref):
        return image_ref
    pull_result = subprocess.run(
        ["docker", "pull", image_ref],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if pull_result.returncode == 0 and sandbox_image.image_exists(tag=image_ref):
        return image_ref
    logger.info("Falling back to local SWE-bench image build for %s", instance_id)
    instance = load_verified_dataset_row(instance_id)
    try:
        import docker
    except Exception as exc:
        raise RuntimeError(f"docker Python SDK is required to build SWE-bench image {image_ref}") from exc
    client = docker.from_env()
    successful, failed = build_instance_images(
        client=client,
        dataset=[instance],
        force_rebuild=False,
        max_workers=1,
        namespace=namespace,
        tag=tag,
        env_image_tag=tag,
    )
    _ = successful
    if failed:
        raise RuntimeError(f"failed to build SWE-bench image {image_ref}: {failed}")
    if not sandbox_image.image_exists(tag=image_ref):
        raise RuntimeError(f"SWE-bench image {image_ref} is still unavailable after local build")
    return image_ref

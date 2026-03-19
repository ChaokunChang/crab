from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from integrations.agents import SandboxHandle, TaskConfig, TaskDescription

from benchmarks.config import BenchmarkConfig
from benchmarks.support import (
    BenchmarkTaskRecord,
    is_replay_llm_service_type,
    task_timeout_seconds,
    verification_timeout_seconds,
)


T = TypeVar("T")
R = TypeVar("R")


def _resolve_dataset_service_config(
    dataset_root: Path,
    raw_value: object,
) -> dict[str, object] | None:
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError(f"llm_service_config must be an object, got {raw_value!r}")
    config = dict(raw_value)
    trace_path = config.get("trace_path")
    if isinstance(trace_path, str):
        config["trace_path"] = str((dataset_root / trace_path).resolve())
    return config


def load_task_dataset(path: Path) -> list[BenchmarkTaskRecord]:
    dataset_path = path.expanduser().resolve()
    dataset_root = dataset_path.parent
    records: list[BenchmarkTaskRecord] = []
    for line_number, raw_line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"dataset row {line_number} in {dataset_path} must be an object")
        compose_file = payload.get("docker_compose_file")
        env_file = payload.get("env_file")
        task_root = payload.get("task_root")
        records.append(
            BenchmarkTaskRecord(
                agent_type=str(payload.get("agent_type", "simulated")),
                task_description=TaskDescription.from_json_value(payload.get("task_description", "")),
                task_config=TaskConfig.from_json_value(payload.get("task_config")),
                task_id=None if payload.get("task_id") is None else str(payload["task_id"]),
                llm_service_type=None if payload.get("llm_service_type") is None else str(payload["llm_service_type"]),
                docker_compose_file=None if compose_file is None else (dataset_root / str(compose_file)).resolve(),
                env_file=None if env_file is None else (dataset_root / str(env_file)).resolve(),
                service_name=None if payload.get("service_name") is None else str(payload["service_name"]),
                task_root=None if task_root is None else (dataset_root / str(task_root)).resolve(),
                llm_service_config=_resolve_dataset_service_config(dataset_root, payload.get("llm_service_config")),
                trace_response_count=None
                if payload.get("trace_response_count") is None
                else int(payload["trace_response_count"]),
                trace_malformed_line_count=None
                if payload.get("trace_malformed_line_count") is None
                else int(payload["trace_malformed_line_count"]),
            )
        )
    if not records:
        raise ValueError(f"dataset {dataset_path} did not contain any task rows")
    return records


def select_task_record(
    dataset: list[BenchmarkTaskRecord] | None,
    *,
    sandbox_index: int,
    default_agent_type: str,
    default_llm_service_type: str | None,
    default_task_description: TaskDescription,
    default_task_config: TaskConfig,
) -> BenchmarkTaskRecord:
    if dataset:
        return dataset[sandbox_index % len(dataset)]
    return BenchmarkTaskRecord(
        agent_type=default_agent_type,
        task_description=default_task_description,
        task_config=default_task_config,
        llm_service_type=default_llm_service_type,
        task_id=None,
        trace_response_count=None,
        trace_malformed_line_count=None,
    )


def resolve_task_records(
    config: BenchmarkConfig,
    *,
    default_task_description: TaskDescription,
    default_task_config: TaskConfig,
) -> list[BenchmarkTaskRecord]:
    dataset = load_task_dataset(config.task_dataset) if config.task_dataset is not None else None
    return [
        select_task_record(
            dataset,
            sandbox_index=index,
            default_agent_type=config.agent,
            default_llm_service_type=config.llm_service,
            default_task_description=default_task_description,
            default_task_config=default_task_config,
        )
        for index in range(config.sandboxes)
    ]


def parallel_map(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    max_workers: int,
) -> list[R]:
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        return list(executor.map(fn, items))


def launch_task_records(
    harness,
    *,
    sandbox_name_prefix: str,
    records: Sequence[BenchmarkTaskRecord],
    max_workers: int,
) -> list[SandboxHandle]:
    indexed_records = list(enumerate(records))
    return parallel_map(
        indexed_records,
        lambda item: harness.launch_task_record(f"{sandbox_name_prefix}-{item[0]}", item[1]),
        max_workers=max_workers,
    )


def sandbox_benchmark_metadata(sandbox: SandboxHandle) -> dict[str, object]:
    metadata = sandbox.launch_metadata.get("benchmark", {})
    return metadata if isinstance(metadata, dict) else {}


def task_id_for_sandbox(sandbox: SandboxHandle) -> str:
    metadata = sandbox_benchmark_metadata(sandbox)
    raw_task_id = metadata.get("task_id")
    if isinstance(raw_task_id, str) and raw_task_id:
        return raw_task_id
    if sandbox.task_config is not None:
        raw_task_id = sandbox.task_config.options.get("task_id")
        if isinstance(raw_task_id, str) and raw_task_id:
            return raw_task_id
    return str(sandbox.sandbox_id)


def trace_response_count_for_sandbox(sandbox: SandboxHandle) -> int:
    metadata = sandbox_benchmark_metadata(sandbox)
    raw_value = metadata.get("trace_response_count")
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def task_completion_timeout_seconds(sandbox: SandboxHandle) -> float:
    timeout_s = task_timeout_seconds(sandbox.task_config or TaskConfig())
    if not is_replay_llm_service_type(sandbox.llm_service_type):
        return timeout_s
    trace_response_count = trace_response_count_for_sandbox(sandbox)
    if trace_response_count <= 0:
        return timeout_s
    # Replay benchmarks can legitimately exceed the original agent timeout after
    # a recovery because they still need to execute the remaining trace steps.
    return max(timeout_s, 900.0, trace_response_count * 20.0)


def poll_sandbox_status(sandbox: SandboxHandle) -> dict[str, object]:
    status = dict(sandbox.last_status)
    if sandbox.task_run is not None:
        try:
            status = sandbox.task_run.poll_status()
        except Exception:
            status = dict(sandbox.last_status)
    sandbox.last_status = dict(status)
    return status


def verify_task_accuracy(harness, sandbox: SandboxHandle) -> tuple[str, dict[str, object]]:
    task_error = ""
    verification = {
        "verification_status": "task_failed",
        "verification_exit_code": -1,
        "verification_ms": 0.0,
    }
    try:
        harness.wait_for_task_completion(
            sandbox,
            timeout_s=task_completion_timeout_seconds(sandbox),
        )
    except Exception as exc:
        task_error = str(exc) or exc.__class__.__name__
    else:
        try:
            verification = harness.verify_task_accuracy(
                sandbox,
                timeout_s=verification_timeout_seconds(sandbox.task_config or TaskConfig()),
            )
        except Exception as exc:
            verification = {
                "verification_status": "verification_error",
                "verification_exit_code": -1,
                "verification_ms": 0.0,
                "verification_stdout": "",
                "verification_stderr": str(exc),
                "verification_command": "",
            }
    return task_error, verification


def build_core_row(
    config: BenchmarkConfig,
    sandbox: SandboxHandle,
    *,
    iteration: int,
    success_ratio: float = 0.0,
    task_error: str = "",
) -> dict[str, object]:
    return {
        "scenario": config.scenario,
        "mode": config.mode,
        "provider": config.provider,
        "agent": sandbox.agent_type or config.agent,
        "llm_service": sandbox.llm_service_type or config.llm_service or "",
        "sandbox_id": str(sandbox.sandbox_id),
        "task_id": task_id_for_sandbox(sandbox),
        "iteration": iteration,
        "success_ratio": success_ratio,
        "task_error": task_error,
    }


def annotate_row(
    config: BenchmarkConfig,
    sandbox: SandboxHandle,
    *,
    iteration: int,
    row: dict[str, object],
    success_ratio: float = 0.0,
    task_error: str = "",
) -> dict[str, object]:
    return {
        **build_core_row(
            config,
            sandbox,
            iteration=iteration,
            success_ratio=success_ratio,
            task_error=task_error,
        ),
        **row,
    }


def replay_enabled(records: Iterable[BenchmarkTaskRecord]) -> bool:
    return any(is_replay_llm_service_type(record.llm_service_type) for record in records)

from __future__ import annotations

import dataclasses
import json
import logging
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from agent_cr.telemetry import start_operation
from integrations.agents import SandboxHandle, TaskConfig, TaskDescription

from benchmarks.config import BenchmarkConfig
from benchmarks.support import (
    BenchmarkTaskRecord,
    is_replay_llm_service_type,
    task_timeout_seconds,
    verification_timeout_seconds,
)


T = TypeVar("T")
U = TypeVar("U")
R = TypeVar("R")

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class BenchmarkSandboxSpec:
    sandbox_name: str
    task_record: BenchmarkTaskRecord


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


def _apply_llm_service_options(
    record: BenchmarkTaskRecord,
    llm_service_options: dict[str, object],
) -> BenchmarkTaskRecord:
    if not llm_service_options:
        return record
    if record.llm_service_config is None:
        return record
    merged = dict(record.llm_service_config)
    for key, value in llm_service_options.items():
        if key not in merged:
            merged[key] = value
    return dataclasses.replace(record, llm_service_config=merged)


def _apply_task_timeout_scales(
    record: BenchmarkTaskRecord,
    *,
    max_agent_timeout_scale: float,
    max_test_timeout_scale: float,
) -> BenchmarkTaskRecord:
    options = dict(record.task_config.options)
    changed = False
    for key, scale in (
        ("max_agent_timeout_sec", max_agent_timeout_scale),
        ("max_test_timeout_sec", max_test_timeout_scale),
    ):
        if key not in options:
            continue
        try:
            options[key] = float(options[key]) * scale
        except (TypeError, ValueError):
            continue
        changed = True
    if not changed:
        return record
    return dataclasses.replace(record, task_config=TaskConfig(options=options))


def resolve_task_records(
    config: BenchmarkConfig,
    *,
    default_task_description: TaskDescription,
    default_task_config: TaskConfig,
) -> list[BenchmarkTaskRecord]:
    dataset = load_task_dataset(config.task_dataset) if config.task_dataset is not None else None
    records = [
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
    if config.llm_service_options:
        records = [_apply_llm_service_options(r, config.llm_service_options) for r in records]
    if config.max_agent_timeout_scale != 1.0 or config.max_test_timeout_scale != 1.0:
        records = [
            _apply_task_timeout_scales(
                r,
                max_agent_timeout_scale=config.max_agent_timeout_scale,
                max_test_timeout_scale=config.max_test_timeout_scale,
            )
            for r in records
        ]
    return records


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


def sandbox_name_for_index(prefix: str, index: int) -> str:
    return f"{prefix}-{index}"


def make_benchmark_sandbox_specs(
    *,
    sandbox_name_prefix: str,
    records: Sequence[BenchmarkTaskRecord],
) -> list[BenchmarkSandboxSpec]:
    return [
        BenchmarkSandboxSpec(
            sandbox_name=sandbox_name_for_index(sandbox_name_prefix, index),
            task_record=record,
        )
        for index, record in enumerate(records)
    ]


def benchmark_phase_run_attributes(
    *,
    phase: str,
    sandbox_count: int,
    configured_max_workers: int,
) -> dict[str, object]:
    return {
        "component": "benchmark",
        "phase": phase,
        "phase_scope": "run",
        "sandbox_count": int(sandbox_count),
        "configured_max_workers": int(configured_max_workers),
    }


def benchmark_phase_item_attributes(
    harness,
    *,
    phase: str,
    sandbox_name: str,
    task_record: BenchmarkTaskRecord | None = None,
    sandbox: SandboxHandle | None = None,
) -> dict[str, object]:
    if sandbox is not None and hasattr(harness, "benchmark_telemetry_attributes"):
        attributes = harness.benchmark_telemetry_attributes(sandbox)
    else:
        task_id = sandbox_name
        agent_type = ""
        llm_service_type = ""
        if task_record is not None:
            if task_record.task_id:
                task_id = task_record.task_id
            agent_type = task_record.agent_type
            llm_service_type = task_record.llm_service_type or ""
        attributes = {
            "component": "benchmark",
            "sandbox_id": sandbox_name,
            "task_id": task_id,
            "agent_type": agent_type,
            "llm_service_type": llm_service_type,
            "task_attempt": 0,
        }
    return {
        **attributes,
        "phase": phase,
        "phase_scope": "sandbox",
    }


def emit_benchmark_phase_progress(
    *,
    phase: str,
    status: str,
    sandbox_count: int,
    configured_max_workers: int,
    duration_seconds: float | None = None,
) -> None:
    line = (
        f"benchmark.phase.{phase} {status} "
        f"sandboxes={int(sandbox_count)} max_workers={int(configured_max_workers)}"
    )
    if duration_seconds is not None:
        line = f"{line} duration_s={duration_seconds:.3f}"
    print(line, flush=True)


def emit_benchmark_phase_skipped(
    *,
    phase: str,
    sandbox_count: int,
    configured_max_workers: int,
) -> None:
    emit_benchmark_phase_progress(
        phase=phase,
        status="skipped",
        sandbox_count=sandbox_count,
        configured_max_workers=configured_max_workers,
    )


def _begin_benchmark_phase(
    *,
    phase: str,
    sandbox_count: int,
    worker_count: int,
    telemetry,
):
    run_attributes = benchmark_phase_run_attributes(
        phase=phase,
        sandbox_count=sandbox_count,
        configured_max_workers=worker_count,
    )
    phase_started_at = time.perf_counter()
    emit_benchmark_phase_progress(
        phase=phase,
        status="start",
        sandbox_count=sandbox_count,
        configured_max_workers=worker_count,
    )
    phase_operation = None
    if telemetry is not None:
        telemetry.emit_metric(
            f"benchmark.phase.{phase}.configured_max_workers",
            float(worker_count),
            run_attributes,
        )
        phase_operation = start_operation(
            telemetry,
            f"benchmark.phase.{phase}",
            run_attributes,
        )
    return phase_started_at, phase_operation


def _finish_benchmark_phase(
    *,
    phase: str,
    sandbox_count: int,
    worker_count: int,
    phase_started_at: float,
    phase_operation,
    status: str,
) -> None:
    if phase_operation is not None:
        phase_operation.finish(status="succeeded" if status == "end" else "failed")
    emit_benchmark_phase_progress(
        phase=phase,
        status=status,
        sandbox_count=sandbox_count,
        configured_max_workers=worker_count,
        duration_seconds=max(0.0, time.perf_counter() - phase_started_at),
    )


def _finish_phase_item_operation(item_operation, *, success: bool) -> None:
    if item_operation is not None:
        item_operation.finish(status="succeeded" if success else "failed")


def _benchmark_setup_run_pipeline_with_separate_pools(
    items: Sequence[T],
    *,
    setup_fn: Callable[[T], U],
    run_fn: Callable[[T, U], R],
    setup_max_workers: int,
    run_max_workers: int,
    harness,
    setup_item_attributes: Callable[[T], dict[str, object]] | None = None,
    run_item_attributes: Callable[[T, U], dict[str, object]] | None = None,
) -> list[R]:
    if not items:
        return []
    setup_worker_count = max(1, setup_max_workers)
    run_worker_count = max(1, run_max_workers)
    telemetry = getattr(harness, "telemetry", None)
    setup_started_at, setup_operation = _begin_benchmark_phase(
        phase="setup",
        sandbox_count=len(items),
        worker_count=setup_worker_count,
        telemetry=telemetry,
    )
    run_started_at: float | None = None
    run_operation = None
    run_started = False
    setup_finished = False
    results: list[R | None] = [None] * len(items)

    def _invoke_setup(item: T) -> U:
        item_operation = None
        if telemetry is not None and setup_item_attributes is not None:
            item_operation = start_operation(
                telemetry,
                "benchmark.phase.setup.item",
                setup_item_attributes(item),
            )
        try:
            result = setup_fn(item)
        except Exception:
            _finish_phase_item_operation(item_operation, success=False)
            raise
        _finish_phase_item_operation(item_operation, success=True)
        return result

    def _invoke_run(item: T, prepared: U) -> R:
        item_operation = None
        if telemetry is not None and run_item_attributes is not None:
            item_operation = start_operation(
                telemetry,
                "benchmark.phase.run.item",
                run_item_attributes(item, prepared),
            )
        try:
            result = run_fn(item, prepared)
        except Exception:
            _finish_phase_item_operation(item_operation, success=False)
            raise
        _finish_phase_item_operation(item_operation, success=True)
        return result

    with (
        ThreadPoolExecutor(max_workers=setup_worker_count) as setup_executor,
        ThreadPoolExecutor(max_workers=run_worker_count) as run_executor,
    ):
        setup_futures: dict[Future[U], tuple[int, T]] = {
            setup_executor.submit(_invoke_setup, item): (index, item)
            for index, item in enumerate(items)
        }
        run_futures: dict[Future[R], tuple[int, T]] = {}
        try:
            while setup_futures or run_futures:
                done, _ = wait(
                    set(setup_futures) | set(run_futures),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    if future in setup_futures:
                        index, item = setup_futures.pop(future)
                        prepared = future.result()
                        if not run_started:
                            run_started_at, run_operation = _begin_benchmark_phase(
                                phase="run",
                                sandbox_count=len(items),
                                worker_count=run_worker_count,
                                telemetry=telemetry,
                            )
                            run_started = True
                        run_futures[run_executor.submit(_invoke_run, item, prepared)] = (index, item)
                        continue
                    index, _item = run_futures.pop(future)
                    results[index] = future.result()
                if not setup_finished and not setup_futures:
                    _finish_benchmark_phase(
                        phase="setup",
                        sandbox_count=len(items),
                        worker_count=setup_worker_count,
                        phase_started_at=setup_started_at,
                        phase_operation=setup_operation,
                        status="end",
                    )
                    setup_finished = True
        except Exception:
            for future in (*setup_futures, *run_futures):
                future.cancel()
            if not setup_finished:
                _finish_benchmark_phase(
                    phase="setup",
                    sandbox_count=len(items),
                    worker_count=setup_worker_count,
                    phase_started_at=setup_started_at,
                    phase_operation=setup_operation,
                    status="failed",
                )
            if run_started and run_started_at is not None:
                _finish_benchmark_phase(
                    phase="run",
                    sandbox_count=len(items),
                    worker_count=run_worker_count,
                    phase_started_at=run_started_at,
                    phase_operation=run_operation,
                    status="failed",
                )
            raise
    if not setup_finished:
        _finish_benchmark_phase(
            phase="setup",
            sandbox_count=len(items),
            worker_count=setup_worker_count,
            phase_started_at=setup_started_at,
            phase_operation=setup_operation,
            status="end",
        )
    if not run_started:
        run_started_at, run_operation = _begin_benchmark_phase(
            phase="run",
            sandbox_count=len(items),
            worker_count=run_worker_count,
            telemetry=telemetry,
        )
    _finish_benchmark_phase(
        phase="run",
        sandbox_count=len(items),
        worker_count=run_worker_count,
        phase_started_at=run_started_at if run_started_at is not None else time.perf_counter(),
        phase_operation=run_operation,
        status="end",
    )
    return [result for result in results]


def _benchmark_setup_run_pipeline_with_shared_pool(
    items: Sequence[T],
    *,
    setup_fn: Callable[[T], U],
    run_fn: Callable[[T, U], R],
    setup_max_workers: int,
    run_max_workers: int,
    harness,
    setup_item_attributes: Callable[[T], dict[str, object]] | None = None,
    run_item_attributes: Callable[[T, U], dict[str, object]] | None = None,
) -> list[R]:
    if not items:
        return []
    shared_worker_count = max(1, min(setup_max_workers, run_max_workers))
    telemetry = getattr(harness, "telemetry", None)
    setup_started_at, setup_operation = _begin_benchmark_phase(
        phase="setup",
        sandbox_count=len(items),
        worker_count=shared_worker_count,
        telemetry=telemetry,
    )
    run_started_at: float | None = None
    run_operation = None
    run_started = False
    setup_finished = False
    completed_setups = 0
    phase_state_lock = threading.Lock()
    results: list[R | None] = [None] * len(items)

    def _invoke_setup(item: T) -> U:
        item_operation = None
        if telemetry is not None and setup_item_attributes is not None:
            item_operation = start_operation(
                telemetry,
                "benchmark.phase.setup.item",
                setup_item_attributes(item),
            )
        try:
            result = setup_fn(item)
        except Exception:
            _finish_phase_item_operation(item_operation, success=False)
            raise
        _finish_phase_item_operation(item_operation, success=True)
        return result

    def _invoke_run(item: T, prepared: U) -> R:
        item_operation = None
        if telemetry is not None and run_item_attributes is not None:
            item_operation = start_operation(
                telemetry,
                "benchmark.phase.run.item",
                run_item_attributes(item, prepared),
            )
        try:
            result = run_fn(item, prepared)
        except Exception:
            _finish_phase_item_operation(item_operation, success=False)
            raise
        _finish_phase_item_operation(item_operation, success=True)
        return result

    def _invoke_setup_and_run(item: T) -> R:
        nonlocal run_started_at, run_operation, run_started, setup_finished, completed_setups

        prepared = _invoke_setup(item)
        with phase_state_lock:
            completed_setups += 1
            if not run_started:
                run_started_at, run_operation = _begin_benchmark_phase(
                    phase="run",
                    sandbox_count=len(items),
                    worker_count=shared_worker_count,
                    telemetry=telemetry,
                )
                run_started = True
            if not setup_finished and completed_setups == len(items):
                _finish_benchmark_phase(
                    phase="setup",
                    sandbox_count=len(items),
                    worker_count=shared_worker_count,
                    phase_started_at=setup_started_at,
                    phase_operation=setup_operation,
                    status="end",
                )
                setup_finished = True
        return _invoke_run(item, prepared)

    with ThreadPoolExecutor(max_workers=shared_worker_count) as executor:
        futures: dict[Future[R], int] = {
            executor.submit(_invoke_setup_and_run, item): index
            for index, item in enumerate(items)
        }
        try:
            while futures:
                done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
                for future in done:
                    index = futures.pop(future)
                    results[index] = future.result()
        except Exception:
            for future in futures:
                future.cancel()
            if not setup_finished:
                _finish_benchmark_phase(
                    phase="setup",
                    sandbox_count=len(items),
                    worker_count=shared_worker_count,
                    phase_started_at=setup_started_at,
                    phase_operation=setup_operation,
                    status="failed",
                )
            if run_started and run_started_at is not None:
                _finish_benchmark_phase(
                    phase="run",
                    sandbox_count=len(items),
                    worker_count=shared_worker_count,
                    phase_started_at=run_started_at,
                    phase_operation=run_operation,
                    status="failed",
                )
            raise
    if not setup_finished:
        _finish_benchmark_phase(
            phase="setup",
            sandbox_count=len(items),
            worker_count=shared_worker_count,
            phase_started_at=setup_started_at,
            phase_operation=setup_operation,
            status="end",
        )
    if not run_started:
        run_started_at, run_operation = _begin_benchmark_phase(
            phase="run",
            sandbox_count=len(items),
            worker_count=shared_worker_count,
            telemetry=telemetry,
        )
    _finish_benchmark_phase(
        phase="run",
        sandbox_count=len(items),
        worker_count=shared_worker_count,
        phase_started_at=run_started_at if run_started_at is not None else time.perf_counter(),
        phase_operation=run_operation,
        status="end",
    )
    return [result for result in results]


def benchmark_setup_run_pipeline(
    items: Sequence[T],
    *,
    setup_fn: Callable[[T], U],
    run_fn: Callable[[T, U], R],
    setup_max_workers: int,
    run_max_workers: int,
    harness,
    setup_item_attributes: Callable[[T], dict[str, object]] | None = None,
    run_item_attributes: Callable[[T, U], dict[str, object]] | None = None,
    executor_pool: str = "separate",
) -> list[R]:
    if executor_pool == "shared":
        return _benchmark_setup_run_pipeline_with_shared_pool(
            items,
            setup_fn=setup_fn,
            run_fn=run_fn,
            setup_max_workers=setup_max_workers,
            run_max_workers=run_max_workers,
            harness=harness,
            setup_item_attributes=setup_item_attributes,
            run_item_attributes=run_item_attributes,
        )
    if executor_pool != "separate":
        raise ValueError(f"unsupported merged setup/run executor pool {executor_pool!r}")
    return _benchmark_setup_run_pipeline_with_separate_pools(
        items,
        setup_fn=setup_fn,
        run_fn=run_fn,
        setup_max_workers=setup_max_workers,
        run_max_workers=run_max_workers,
        harness=harness,
        setup_item_attributes=setup_item_attributes,
        run_item_attributes=run_item_attributes,
    )


def benchmark_phase_map(
    items: Sequence[T],
    fn: Callable[[T], R],
    *,
    phase: str,
    max_workers: int,
    harness,
    item_attributes: Callable[[T], dict[str, object]] | None = None,
) -> list[R]:
    if not items:
        return []
    worker_count = max(1, max_workers)
    telemetry = getattr(harness, "telemetry", None)
    phase_started_at, phase_operation = _begin_benchmark_phase(
        phase=phase,
        sandbox_count=len(items),
        worker_count=worker_count,
        telemetry=telemetry,
    )

    def _invoke(item: T) -> R:
        item_operation = None
        if telemetry is not None and item_attributes is not None:
            item_operation = start_operation(
                telemetry,
                f"benchmark.phase.{phase}.item",
                item_attributes(item),
            )
        try:
            result = fn(item)
        except Exception:
            _finish_phase_item_operation(item_operation, success=False)
            raise
        _finish_phase_item_operation(item_operation, success=True)
        return result

    try:
        results = parallel_map(items, _invoke, max_workers=worker_count)
    except Exception:
        _finish_benchmark_phase(
            phase=phase,
            sandbox_count=len(items),
            worker_count=worker_count,
            phase_started_at=phase_started_at,
            phase_operation=phase_operation,
            status="failed",
        )
        raise
    _finish_benchmark_phase(
        phase=phase,
        sandbox_count=len(items),
        worker_count=worker_count,
        phase_started_at=phase_started_at,
        phase_operation=phase_operation,
        status="end",
    )
    return results


def setup_task_records_phase(
    harness,
    *,
    specs: Sequence[BenchmarkSandboxSpec],
    max_workers: int,
):
    setup_task_record = getattr(harness, "setup_task_record", None)
    if not callable(setup_task_record):
        raise AttributeError(f"{type(harness).__name__} must define setup_task_record(...)")
    return benchmark_phase_map(
        specs,
        lambda spec: setup_task_record(spec.sandbox_name, spec.task_record),
        phase="setup",
        max_workers=max_workers,
        harness=harness,
        item_attributes=lambda spec: benchmark_phase_item_attributes(
            harness,
            phase="setup",
            sandbox_name=spec.sandbox_name,
            task_record=spec.task_record,
        ),
    )


def start_prepared_task_record(harness, prepared) -> SandboxHandle:
    run_prepared_task_record = getattr(harness, "run_prepared_task_record", None)
    if callable(run_prepared_task_record):
        return run_prepared_task_record(prepared)
    return prepared.handle


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


def replay_trace_cursor(status: dict[str, object]) -> int:
    raw_value = status.get("replay_trace_cursor", status.get("total_actions", 0))
    try:
        return max(0, int(raw_value))
    except (TypeError, ValueError):
        return 0


def replay_status_is_complete(status: dict[str, object], *, trace_response_count: int) -> bool:
    if bool(status.get("replay_is_complete", False)):
        return True
    if trace_response_count <= 0:
        return False
    return replay_trace_cursor(status) >= trace_response_count


def replay_action_count_wait_error(task_error: str) -> bool:
    lowered = str(task_error).strip().lower()
    if not lowered or "replay action count" not in lowered:
        return False
    return "timed out waiting" in lowered or "finished before reaching" in lowered


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
    completion_timeout = task_completion_timeout_seconds(sandbox)
    wait_error = ""
    try:
        logger.info(
            "Benchmark verification waiting for task completion sandbox=%s timeout_s=%.3f",
            sandbox.sandbox_id,
            completion_timeout,
        )
        harness.wait_for_task_completion(
            sandbox,
            timeout_s=completion_timeout,
        )
    except Exception as exc:
        wait_error = str(exc) or exc.__class__.__name__
        status = poll_sandbox_status(sandbox)
        trace_response_count = trace_response_count_for_sandbox(sandbox)
        if is_replay_llm_service_type(sandbox.llm_service_type) and replay_status_is_complete(
            status,
            trace_response_count=trace_response_count,
        ):
            logger.info(
                "Benchmark verification continuing after task completion wait failed because replay is complete "
                "sandbox=%s error=%s replay_final_trace_cursor=%d trace_response_count=%d",
                sandbox.sandbox_id,
                wait_error,
                replay_trace_cursor(status),
                trace_response_count,
            )
        else:
            task_error = wait_error
            logger.warning(
                "Benchmark verification skipped after task completion wait failed sandbox=%s error=%s",
                sandbox.sandbox_id,
                task_error,
            )
    if not task_error:
        verification_timeout = verification_timeout_seconds(sandbox.task_config or TaskConfig())
        try:
            logger.info(
                "Benchmark verification starting sandbox=%s timeout_s=%.3f",
                sandbox.sandbox_id,
                verification_timeout,
            )
            verification = harness.verify_task_accuracy(
                sandbox,
                timeout_s=verification_timeout,
            )
        except Exception as exc:
            logger.warning(
                "Benchmark verification raised an exception sandbox=%s error=%s",
                sandbox.sandbox_id,
                exc,
            )
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


def emit_row_telemetry(
    harness,
    sandbox: SandboxHandle,
    row: dict[str, object],
    *,
    iteration: int,
) -> None:
    emit_metric = getattr(harness, "emit_benchmark_metric", None)
    if not callable(emit_metric):
        return
    field_to_metric = {
        "checkpoint_ms": "benchmark.checkpoint_ms",
        "restore_ms": "benchmark.restore_ms",
        "recovery_ms": "benchmark.recovery_ms",
        "readiness_ms": "benchmark.readiness_ms",
        "end_to_end_recovery_ms": "benchmark.end_to_end_recovery_ms",
        "workload_resume_ms": "benchmark.workload_resume_ms",
        "migration_ms": "benchmark.migration_ms",
        "budget_slack_ms": "benchmark.budget_slack_ms",
        "checkpoint_batch_ms": "benchmark.checkpoint_batch_ms",
        "restore_batch_ms": "benchmark.restore_batch_ms",
        "replay_progress_ms": "benchmark.replay_progress_ms",
        "fanout_ms": "benchmark.fanout_ms",
        "lost_actions": "benchmark.lost_actions",
        "success_ratio": "benchmark.task.success_ratio",
    }
    event_type = None if row.get("event_type") is None else str(row.get("event_type"))
    checkpoint_id = None
    raw_checkpoint_id = row.get("checkpoint_id")
    if isinstance(raw_checkpoint_id, str) and raw_checkpoint_id:
        checkpoint_id = raw_checkpoint_id
    for field_name, metric_name in field_to_metric.items():
        if field_name not in row:
            continue
        try:
            value = float(row[field_name])
        except (TypeError, ValueError):
            continue
        extra = {"source_row_field": field_name}
        if row.get("event_injected") is not None:
            extra["event_injected"] = int(row["event_injected"])
        if row.get("recovery_status") is not None:
            extra["recovery_status"] = str(row["recovery_status"])
        if checkpoint_id is not None:
            extra["checkpoint_id"] = checkpoint_id
        emit_metric(metric_name, value, sandbox, iteration=iteration, event_type=event_type, extra=extra)


def replay_enabled(records: Iterable[BenchmarkTaskRecord]) -> bool:
    return any(is_replay_llm_service_type(record.llm_service_type) for record in records)

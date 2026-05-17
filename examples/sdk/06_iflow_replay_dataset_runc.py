"""Run the first two iFlow replay dataset rows through the SDK on runc.

This mirrors the old benchmark smoke:

    python3 -m benchmarks.run --config benchmarks/examples/iflow/iflow.fault.auto.madoka.mini.yaml

but keeps the user-facing sandbox path small: each task is just a
DockerComposeTemplate-backed `Sandbox(...)` plus an `IFlowAgent().bind(...)`.
Start the replay router in another shell first; see docs/sdk-iflow-replay.md.

    AGENT_CR_REPLAY_BASE_URL=http://127.0.0.1:18080 \
      PYTHONPATH=. python3 examples/sdk/06_iflow_replay_dataset_runc.py
"""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agent_cr import Engine, EngineConfig, Sandbox
from agent_cr.agents_builtin.iflow import IFlowAgent
from agent_cr.templates import DockerComposeTemplate
from integrations.llm_services.router import BenchmarkLLMRouterClient


DATASET = Path(os.environ.get(
    "AGENT_CR_REPLAY_DATASET",
    "/root/workspace/agent-cr/results/datasets/termnius_iflow_replay_128tasks_light.jsonl",
))
REPLAY_BASE_URL = os.environ.get("AGENT_CR_REPLAY_BASE_URL", "http://127.0.0.1:18080")
ENGINE_CONFIG = Path(os.environ.get(
    "AGENT_CR_ENGINE_CONFIG",
    Path(__file__).resolve().parent / "configs" / "iflow_replay_engine.runc.yaml",
))


def _load_rows(dataset_path: Path, *, count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
        if raw_line.strip():
            rows.append(json.loads(raw_line))
        if len(rows) >= count:
            break
    if len(rows) < count:
        raise RuntimeError(f"expected {count} rows in {dataset_path}, found {len(rows)}")
    return rows


def _trace_config(dataset_path: Path, row: dict[str, Any]) -> dict[str, object]:
    dataset_root = dataset_path.parent
    config = dict(row.get("llm_service_config") or {})
    trace_path = config.get("trace_path")
    if isinstance(trace_path, str):
        config["trace_path"] = str((dataset_root / trace_path).resolve())
    config.setdefault("response_delay_policy", "trace_replay")
    config.setdefault("response_delay_scaling_factor", 1.0)
    return config


def _sandbox_name(index: int, row: dict[str, Any]) -> str:
    return f"sdk-iflow-replay-{index}-{row['task_id']}"


def _prompt(row: dict[str, Any]) -> str:
    task_description = row.get("task_description")
    if isinstance(task_description, dict) and isinstance(task_description.get("prompt"), str):
        return task_description["prompt"]
    if isinstance(task_description, str):
        return task_description
    raise ValueError(f"dataset row has no prompt: {row!r}")


def _timeout(row: dict[str, Any], key: str, default: float) -> float:
    options = ((row.get("task_config") or {}).get("options") or {})
    try:
        return max(1.0, float(options.get(key, default)))
    except (TypeError, ValueError):
        return default


def _verify(sbx: Sandbox, row: dict[str, Any]) -> tuple[int, str, str]:
    result = sbx.commands.run(
        argv=["/bin/bash", "-lc", "bash /tests/run-tests.sh"],
        cwd=sbx.process_cwd,
        env={
            "TEST_DIR": "/tests",
            "PATH": "/root/.local/agent-cr-verification/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "OMP_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "NUMEXPR_NUM_THREADS": "4",
            "BLIS_NUM_THREADS": "4",
            "VECLIB_MAXIMUM_THREADS": "4",
            "LOKY_MAX_CPU_COUNT": "4",
        },
        timeout=_timeout(row, "max_test_timeout_sec", 180.0),
        capture_output=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


def main() -> None:
    rows = _load_rows(DATASET, count=2)
    names = [_sandbox_name(index, row) for index, row in enumerate(rows)]
    replay = BenchmarkLLMRouterClient(REPLAY_BASE_URL, timeout_seconds=30.0)
    sandboxes: list[tuple[Sandbox, IFlowAgent, dict[str, Any], str]] = []
    try:
        for name, row in zip(names, rows, strict=True):
            replay.register_sandbox(
                sandbox_id=name,
                llm_service_type="iflow_trace_replay",
                llm_service_config=_trace_config(DATASET, row),
            )

        engine_config = EngineConfig.from_file(ENGINE_CONFIG)
        with Engine.start(engine_config) as engine:
            print(f"engine config: {ENGINE_CONFIG}")
            print(f"engine storage_root: {engine.storage_root}")
            print(f"engine runtime_root: {engine.runtime_root}")
            if engine.config.telemetry_config and engine.config.telemetry_config.jsonl_path:
                print(f"engine telemetry: {engine.config.telemetry_config.jsonl_path}")
            if engine.config.log_file:
                print(f"engine log: {engine.config.log_file}")
            for name, row in zip(names, rows, strict=True):
                agent = IFlowAgent(timeout=_timeout(row, "max_agent_timeout_sec", 900.0))
                template = DockerComposeTemplate.from_dataset_row(DATASET, row)
                sbx = Sandbox(
                    template=template,
                    engine=engine,
                    name=name,
                )
                agent.bind(sbx, llm_url=REPLAY_BASE_URL)
                sandboxes.append((sbx, agent, row, name))
                print(f"launched {name}: cwd={sbx.process_cwd}")

            with ThreadPoolExecutor(max_workers=2) as pool:
                task_futures = {
                    pool.submit(
                        agent.run,
                        _prompt(row),
                        timeout=_timeout(row, "max_agent_timeout_sec", 900.0),
                    ): name
                    for _, agent, row, name in sandboxes
                }
                for future in as_completed(task_futures):
                    name = task_futures[future]
                    result = future.result()
                    print(f"agent {name}: exit={result.exit_code} output={result.output.strip()!r}")

            with ThreadPoolExecutor(max_workers=2) as pool:
                verify_futures = {
                    pool.submit(_verify, sbx, row): name
                    for sbx, _, row, name in sandboxes
                }
                for future in as_completed(verify_futures):
                    name = verify_futures[future]
                    code, stdout, stderr = future.result()
                    state = replay.snapshot(name) or {}
                    nested_state = state.get("state") if isinstance(state, dict) else {}
                    if not isinstance(nested_state, dict):
                        nested_state = {}
                    print(
                        f"verify {name}: exit={code} "
                        f"trace={nested_state.get('consumed_response_count')}/{nested_state.get('total_responses')}"
                    )
                    if stdout.strip():
                        print(f"stdout {name}:\n{stdout.rstrip()}")
                    if stderr.strip():
                        print(f"stderr {name}:\n{stderr.rstrip()}")
                    if code != 0:
                        raise SystemExit(f"verification failed for {name}")
    finally:
        for sbx, _, _, _ in sandboxes:
            sbx.kill()
        for name in names:
            try:
                replay.unregister_sandbox(name)
            except Exception:
                pass
        replay.close()


if __name__ == "__main__":
    main()

"""Run iFlow against a replayed LLM trace inside an Agent-CR sandbox.

Setup (three terminals):

    # terminal 1 — daemon
    agentcr daemon start --foreground \\
      --config examples/sdk/configs/iflow_replay_engine.runc.yaml

    # terminal 2 — replay router (provides /v1/chat/completions from a trace)
    PYTHONPATH=. python3 -m integrations.llm_services.router \\
      --host 127.0.0.1 --port 18080 \\
      --telemetry-jsonl /tmp/agentcr-iflow-router.telemetry.jsonl

    # terminal 3 — this script
    PYTHONPATH=. python3 examples/sdk/02_iflow_replay.py
"""
from __future__ import annotations

import json
from pathlib import Path

from agent_cr import Engine, Sandbox
from agent_cr.agents_builtin.iflow import IFlowAgent
from agent_cr.templates import DockerComposeTemplate
from integrations.llm_services.router import BenchmarkLLMRouterClient


DATASET = Path("/root/workspace/agent-cr/results/datasets/termnius_iflow_replay_128tasks_light.jsonl")
ROW_INDEX = 1  # crack-7z-hash — a multi-turn task with shell + write_file calls.
REPLAY_URL = "http://127.0.0.1:18080"


def _load_row(index: int) -> dict:
    for current, line in enumerate(DATASET.read_text().splitlines()):
        if current == index:
            return json.loads(line)
    raise IndexError(f"dataset row {index} not found")


def _trace_config(row: dict) -> dict:
    config = dict(row["llm_service_config"])
    config["trace_path"] = str((DATASET.parent / config["trace_path"]).resolve())
    config["response_delay_policy"] = "trace_replay"
    config["response_delay_scaling_factor"] = 1.0
    return config


def main() -> None:
    row = _load_row(ROW_INDEX)
    name = f"sdk-iflow-replay-{row['task_id']}"
    prompt = row["task_description"]["prompt"]

    replay = BenchmarkLLMRouterClient(REPLAY_URL, timeout_seconds=30.0)
    replay.register_sandbox(
        sandbox_id=name,
        llm_service_type="iflow_trace_replay",
        llm_service_config=_trace_config(row),
    )

    try:
        with Engine.connect() as engine:
            sbx = Sandbox(
                template=DockerComposeTemplate.from_dataset_row(DATASET, row),
                engine=engine,
                name=name,
            )
            try:
                agent = IFlowAgent(timeout=900.0).bind(sbx, llm_url=REPLAY_URL)
                result = agent.run(prompt)
                print("agent exit:", result.exit_code)
                print(result.output.rstrip())
            finally:
                sbx.kill()
    finally:
        replay.unregister_sandbox(name)
        replay.close()


if __name__ == "__main__":
    main()

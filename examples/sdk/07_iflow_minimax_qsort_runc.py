"""Run iFlow in an Ubuntu 22.04 runc sandbox against real MiniMax.

The task asks iFlow to create `/work/qsort` with:

- `input.txt`: 100 randomly generated integers, one per line.
- `qsort.py`: a quicksort implementation that reads `input.txt` and writes
  sorted numbers to `sorted.txt`.

Run from the repository root:

    PYTHONPATH=. python3 examples/sdk/07_iflow_minimax_qsort_runc.py

Credentials and endpoint defaults match the MiniMax values used for this
example, and can be overridden with environment variables:

    MINIMAX_MODEL=MiniMax-M2.7
    MINIMAX_API_BASE=https://api.minimax.chat/v1
    MINIMAX_API_KEY_FILE=/root/workspace/agent-os/.minimax
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from agent_cr import Engine, EngineConfig, Sandbox
from agent_cr.agents_builtin.iflow import IFlowAgent


MODEL = os.environ.get("MINIMAX_MODEL", "MiniMax-M2.7")
API_BASE = os.environ.get("MINIMAX_API_BASE", "https://api.minimax.chat/v1")
API_KEY_FILE = Path(os.environ.get("MINIMAX_API_KEY_FILE", "/root/workspace/agent-os/.minimax"))
ENGINE_CONFIG = Path(
    os.environ.get(
        "AGENT_CR_ENGINE_CONFIG",
        Path(__file__).resolve().parent / "configs" / "iflow_minimax_qsort_engine.runc.yaml",
    )
)
WORK_DIR = Path(
    os.environ.get(
        "AGENT_CR_QSORT_WORK_DIR",
        "/root/workspace/agent-cr/data/agent_cr/sdk/iflow-minimax-qsort-runc/work/qsort-demo",
    )
)


TASK = """\
Create a directory named /work/qsort.

Inside /work/qsort:
1. Create input.txt containing exactly 100 randomly generated integers, one integer per line.
2. Create qsort.py. It must read input.txt from the same directory, implement quicksort itself,
   sort the integers numerically, and write the sorted result to sorted.txt with one integer per line.

Do not put extra prose in the files. Make qsort.py runnable with python3, but do not install packages.
"""


def _read_api_key() -> str:
    try:
        api_key = API_KEY_FILE.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"MiniMax API key file not found: {API_KEY_FILE}") from exc
    if not api_key:
        raise RuntimeError(f"MiniMax API key file is empty: {API_KEY_FILE}")
    return api_key


def _verify(work_dir: Path) -> str:
    qsort_dir = work_dir / "qsort"
    subprocess.run(
        [sys.executable, "qsort.py"],
        cwd=qsort_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=True,
    )
    input_numbers = [int(item) for item in (qsort_dir / "input.txt").read_text().split()]
    sorted_numbers = [int(item) for item in (qsort_dir / "sorted.txt").read_text().split()]
    if len(input_numbers) != 100:
        raise AssertionError(f"expected 100 input numbers, got {len(input_numbers)}")
    if sorted_numbers != sorted(input_numbers):
        raise AssertionError("sorted.txt is not sorted(input.txt)")
    return "\n".join(
        [
            f"verified {len(input_numbers)} numbers",
            "first10: " + " ".join(map(str, sorted_numbers[:10])),
            "last10: " + " ".join(map(str, sorted_numbers[-10:])),
        ]
    )


def main() -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["AGENT_CR_IFLOW_API_KEY"] = _read_api_key()

    config = EngineConfig.from_file(ENGINE_CONFIG)
    with Engine.start(config) as engine:
        print(f"engine config: {ENGINE_CONFIG}")
        print(f"engine storage_root: {engine.storage_root}")
        print(f"engine runtime_root: {engine.runtime_root}")
        print(f"work dir host: {WORK_DIR}")
        if engine.config.telemetry_config and engine.config.telemetry_config.jsonl_path:
            print(f"engine telemetry: {engine.config.telemetry_config.jsonl_path}")
        if engine.config.log_file:
            print(f"engine log: {engine.config.log_file}")

        sbx = Sandbox(
            image="agentos-ubuntu:base",
            work_dir=WORK_DIR,
            engine=engine,
            name="sdk-iflow-minimax-qsort",
        )
        try:
            sbx.commands.run("rm -rf /work/qsort", check=True)
            agent = IFlowAgent(model=MODEL, timeout=900.0, max_session_turns=32)
            agent.bind(sbx, llm_url=API_BASE)

            result = agent.run(TASK)
            print("agent exit:", result.exit_code)
            if result.output.strip():
                print("agent stdout:")
                print(result.output.rstrip())
            stderr = result.extra.get("stderr")
            if isinstance(stderr, str) and stderr.strip():
                print("agent stderr:")
                print(stderr.rstrip())
            if result.exit_code != 0:
                raise SystemExit(result.exit_code)

            print("verification:")
            print(_verify(WORK_DIR).rstrip())
            print("created files:")
            print(sbx.commands.run("find /work/qsort -maxdepth 1 -type f -printf '%f\n' | sort", check=True).stdout.rstrip())
            while True:
                try:
                    pass
                except KeyboardInterrupt:
                    print("Exit")
                    break
        finally:
            sbx.kill()


if __name__ == "__main__":
    main()

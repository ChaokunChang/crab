"""Run iFlow against the real MiniMax API: write a quicksort program.

Before running, start the daemon in another terminal:

    crab daemon start --foreground \\
      --config examples/sdk/configs/iflow_minimax_qsort_engine.runc.yaml

Then run this script (the API key file path can be overridden with
`MINIMAX_API_KEY_FILE`):

    PYTHONPATH=. python3 examples/sdk/03_iflow_minimax_qsort.py
"""
from __future__ import annotations

import os
from pathlib import Path

from crab import Engine, Sandbox
from crab.agents_builtin.iflow import IFlowAgent


API_BASE = "https://api.minimax.chat/v1"
API_KEY_FILE = Path(os.environ.get("MINIMAX_API_KEY_FILE", "/root/workspace/agent-os/.minimax"))
MODEL = "MiniMax-M2.7"
WORK_DIR = Path("/root/workspace/crab/data/crab/sdk/iflow-minimax-qsort-runc/work/qsort-demo")

TASK = """\
Create a directory /work/qsort.

Inside /work/qsort:
1. Create input.txt with 100 random integers, one per line.
2. Create qsort.py that reads input.txt, sorts the numbers with quicksort,
   and writes the result to sorted.txt (one number per line).

Do not install packages and do not add prose to the files.
"""


def main() -> None:
    os.environ["CRAB_IFLOW_API_KEY"] = API_KEY_FILE.read_text().strip()
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    with Engine.connect() as engine:
        sbx = Sandbox(
            image="agentos-ubuntu:base",
            work_dir=WORK_DIR,
            engine=engine,
            name="sdk-iflow-minimax-qsort",
        )
        try:
            agent = IFlowAgent(model=MODEL, timeout=900.0).bind(sbx, llm_url=API_BASE)
            result = agent.run(TASK)
            print("agent exit:", result.exit_code)
            print(sbx.commands.run("ls /work/qsort").stdout.rstrip())
            print("input.txt  (first 10):", sbx.commands.run("head -n 10 /work/qsort/input.txt").stdout.rstrip())
            print("sorted.txt (first 10):", sbx.commands.run("head -n 10 /work/qsort/sorted.txt").stdout.rstrip())
        finally:
            sbx.kill()


if __name__ == "__main__":
    main()

"""Bring-your-own agent demo.

Shows the minimal contract for integrating a custom agent into Agent-CR:
declare `llm_protocol`, optionally `install()`, required `execute()`. The engine
handles LLM-traffic interception, sandbox identity tagging, and
semantic-aware C/R automatically — the agent code itself doesn't know any of
that is happening.

This example uses a fake-LLM agent that just echoes the task. Run with:

    python3 examples/sdk/02_byo_agent.py
"""
from __future__ import annotations

from agent_cr import Agent, Engine, EngineConfig, Sandbox, TaskResult


class EchoAgent(Agent):
    """A no-LLM agent: returns the task string verbatim. Useful for SDK demos."""

    name = "echo"
    llm_protocol = "openai"
    version = "demo-1"

    def install(self, sbx: Sandbox) -> None:
        # No-op for this fake agent. Real agents would `sbx.commands.run(...)`
        # to install their CLI here.
        print(f"[install] sandbox={sbx.sandbox_id} llm_base_url={sbx.llm_base_url}")

    def execute(self, sbx: Sandbox, task: str) -> TaskResult:
        print(f"[execute] sandbox={sbx.sandbox_id} task={task!r}")
        return TaskResult(exit_code=0, output=f"echo: {task}")


def main() -> None:
    with Engine.start(EngineConfig(runtime="docker")) as engine:
        sbx = Sandbox(engine=engine)
        agent = EchoAgent().bind(sbx, llm_url="https://api.openai.com")
        try:
            print("task 1 result:", agent.run("first task"))
            print("task 2 result:", agent.run("follow-up task"))
        finally:
            sbx.kill()


if __name__ == "__main__":
    main()

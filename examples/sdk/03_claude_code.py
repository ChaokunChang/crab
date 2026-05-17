"""Claude Code via the SDK.

Requires:
  - Engine configured with runtime="runc" + host-side runc/ZFS/CRIU/network
  - An image with `node` (or apt-get available) so claude-code can install
  - ANTHROPIC_API_KEY set in the sandbox env so the SDK propagates it to claude

The example is intentionally short: create a sandbox, attach Claude Code as
an agent, then iterate with multiple `agent.run(...)` calls on the same
sandbox.

    python3 examples/sdk/03_claude_code.py
"""
from __future__ import annotations

import os

from agent_cr import Engine, EngineConfig, Sandbox
from agent_cr.agents_builtin.claude_code import ClaudeCodeAgent


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise SystemExit("set ANTHROPIC_API_KEY in your environment")

    with Engine.start(EngineConfig(runtime="runc")) as engine:
        sbx = Sandbox(
            image="ubuntu:22.04",
            env={"ANTHROPIC_API_KEY": api_key},
            engine=engine,
        )
        agent = ClaudeCodeAgent().bind(sbx, llm_url="https://api.anthropic.com")
        try:
            r1 = agent.run(
                "Create /work/hello.py that prints 'hello from claude'"
            )
            print("task 1:", r1)

            # Sandbox is still alive; inspect what claude did.
            ls = sbx.commands.run("ls -la /work")
            print(ls.stdout)

            # Iterate with a second task — same sandbox, same context.
            r2 = agent.run("Now make hello.py also print the current date")
            print("task 2:", r2)

            # Snapshot the state, just because we can.
            ckpt = sbx.checkpoint(label="after-second-task")
            print("checkpoint id:", ckpt)
        finally:
            sbx.kill()


if __name__ == "__main__":
    main()

"""Built-in agent profiles for the user-facing SDK.

These adapters wrap the existing harness-internal `BaseAgent` infrastructure
(under `integrations/agents/`) so users can write:

    agent = ClaudeCodeAgent().bind(sbx, llm_url="https://api.anthropic.com")

without depending on the benchmark harness or touching the one-shot
`BaseAgent` contract that was designed for the harness flow.

The split exists because the harness's `BaseAgent` bakes the task into the
OCI bundle at construction time (one-shot semantics), while the SDK exposes
multi-task per sandbox. The adapters in this package implement the SDK's
`install()`/`execute()` contract by either:
  (a) re-using the task-independent prepare helpers from
      `integrations/sandboxes/<agent>/harness.py`, or
  (b) calling the agent's CLI directly via `sbx.commands.run(...)` for each
      `run(task)` invocation.

Registering these built-ins on import keeps `resolve_agent("claude-code")`
working without operators having to call `register_agent()` themselves.
"""
from __future__ import annotations

from ..agent import register_agent
from .claude_code import ClaudeCodeAgent
from .iflow import IFlowAgent


register_agent("claude-code", ClaudeCodeAgent)
register_agent("claude_code", ClaudeCodeAgent)  # underscore alias for the harness convention
register_agent("iflow", IFlowAgent)


__all__ = ["ClaudeCodeAgent", "IFlowAgent"]

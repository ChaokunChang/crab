from __future__ import annotations

from integrations.agents.base import BaseAgent, TaskConfig, TaskDescription
from integrations.agents.claude_code import ClaudeCodeAgent
from integrations.agents.contracts import SandboxHandle
from integrations.agents.iflow import IFlowAgent
from integrations.agents.mini_swe import MiniSweAgent
from integrations.agents.simulated import SimulatedAgent
from integrations.agents.terminus import TerminusAgent


def build_agent_registry() -> dict[str, type[BaseAgent]]:
    return {
        SimulatedAgent.agent_type: SimulatedAgent,
        IFlowAgent.agent_type: IFlowAgent,
        MiniSweAgent.agent_type: MiniSweAgent,
        ClaudeCodeAgent.agent_type: ClaudeCodeAgent,
        TerminusAgent.agent_type: TerminusAgent,
    }


__all__ = [
    "BaseAgent",
    "ClaudeCodeAgent",
    "IFlowAgent",
    "MiniSweAgent",
    "SandboxHandle",
    "SimulatedAgent",
    "TaskConfig",
    "TaskDescription",
    "TerminusAgent",
    "build_agent_registry",
]

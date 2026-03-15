from __future__ import annotations

from integrations.agents.base import BaseAgent, TaskConfig, TaskDescription
from integrations.agents.contracts import SandboxHandle
from integrations.agents.iflow import IFlowAgent
from integrations.agents.simulated import SimulatedAgent


def build_agent_registry() -> dict[str, type[BaseAgent]]:
    return {
        SimulatedAgent.agent_type: SimulatedAgent,
        IFlowAgent.agent_type: IFlowAgent,
    }


__all__ = [
    "BaseAgent",
    "IFlowAgent",
    "SandboxHandle",
    "SimulatedAgent",
    "TaskConfig",
    "TaskDescription",
    "build_agent_registry",
]

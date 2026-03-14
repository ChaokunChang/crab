from __future__ import annotations

from benchmarks.agents.base import BaseAgent, TaskConfig, TaskDescription
from benchmarks.agents.iflow import IFlowAgent
from benchmarks.agents.simulated import SimulatedAgent


def build_agent_registry() -> dict[str, type[BaseAgent]]:
    return {
        SimulatedAgent.agent_type: SimulatedAgent,
        IFlowAgent.agent_type: IFlowAgent,
    }


__all__ = [
    "BaseAgent",
    "IFlowAgent",
    "SimulatedAgent",
    "TaskConfig",
    "TaskDescription",
    "build_agent_registry",
]

"""
Agents module for CoScientist.

This module contains the agent components for the CoScientist system.
"""

from coscientist.agents.agent_manager import AgentManager
from coscientist.agents.base_agent import BaseAgent
from coscientist.agents.default_agents import (
    DefaultAgent,
    GenerationAgent,
    ReflectionAgent,
    EvolutionAgent,
    RankingAgent,
    MetaReviewAgent,
)

__all__ = [
    "AgentManager",
    "BaseAgent",
    "DefaultAgent",
    "GenerationAgent",
    "ReflectionAgent",
    "EvolutionAgent",
    "RankingAgent",
    "MetaReviewAgent",
]

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
from coscientist.agents.specialized_agents import (
    ManagerAgent,
    LiteratureAgent,
    ResearchAgent,
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
    "ManagerAgent",
    "LiteratureAgent",
    "ResearchAgent",
]

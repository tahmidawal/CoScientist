"""
Core module for CoScientist.

This module contains the core components of the CoScientist system.
"""

from coscientist.core.coscientist import CoScientist
from coscientist.core.hypothesis import Hypothesis
from coscientist.core.workflow import (
    ResearchWorkflow,
    StandardResearchWorkflow,
    CustomResearchWorkflow,
    IterativeResearchWorkflow,
    create_workflow,
)

__all__ = [
    "CoScientist",
    "Hypothesis",
    "ResearchWorkflow",
    "StandardResearchWorkflow",
    "CustomResearchWorkflow",
    "IterativeResearchWorkflow",
    "create_workflow",
]

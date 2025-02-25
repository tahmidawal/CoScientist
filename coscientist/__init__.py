"""
CoScientist: AI Co-Scientist for Autonomous Research

CoScientist is an advanced AI system capable of autonomously generating novel 
and high-impact research ideas in Machine Learning and related fields.
"""

__version__ = "0.1.0"

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

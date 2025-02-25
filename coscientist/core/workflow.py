from abc import ABC, abstractmethod
from typing import Dict, List, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

class ResearchWorkflow(ABC):
    """
    Abstract base class for research workflows.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Initialize a research workflow.
        
        Args:
            name (str): Name of the workflow
            description (str, optional): Description of the workflow. Defaults to "".
        """
        self.name = name
        self.description = description
    
    @abstractmethod
    def execute(self, co_scientist) -> Dict[str, Any]:
        """
        Execute the workflow.
        
        Args:
            co_scientist: The CoScientist instance
            
        Returns:
            Dict[str, Any]: Result of the workflow
        """
        pass

class StandardResearchWorkflow(ResearchWorkflow):
    """
    Standard research workflow implementation.
    """
    
    def __init__(self, 
                 name: str = "Standard Research Workflow", 
                 description: str = "Generate, refine, evolve, and rank hypotheses.",
                 num_hypotheses: int = 5,
                 generations: int = 3):
        """
        Initialize a standard research workflow.
        
        Args:
            name (str, optional): Name of the workflow. Defaults to "Standard Research Workflow".
            description (str, optional): Description of the workflow. 
                Defaults to "Generate, refine, evolve, and rank hypotheses.".
            num_hypotheses (int, optional): Number of hypotheses to generate. Defaults to 5.
            generations (int, optional): Number of evolution generations. Defaults to 3.
        """
        super().__init__(name, description)
        self.num_hypotheses = num_hypotheses
        self.generations = generations
    
    def execute(self, co_scientist) -> Dict[str, Any]:
        """
        Execute the standard research workflow.
        
        Args:
            co_scientist: The CoScientist instance
            
        Returns:
            Dict[str, Any]: Result of the workflow
        """
        return co_scientist.run_full_workflow(
            num_hypotheses=self.num_hypotheses,
            generations=self.generations
        )

class CustomResearchWorkflow(ResearchWorkflow):
    """
    Custom research workflow with configurable steps.
    """
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 steps: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize a custom research workflow.
        
        Args:
            name (str): Name of the workflow
            description (str, optional): Description of the workflow. Defaults to "".
            steps (List[Dict[str, Any]], optional): List of steps. Defaults to None.
        """
        super().__init__(name, description)
        self.steps = steps or []
    
    def add_step(self, method_name: str, params: Dict[str, Any] = None):
        """
        Add a step to the workflow.
        
        Args:
            method_name (str): Name of the method to call
            params (Dict[str, Any], optional): Parameters for the method. Defaults to None.
        """
        self.steps.append({
            "method": method_name,
            "params": params or {}
        })
    
    def execute(self, co_scientist) -> Dict[str, Any]:
        """
        Execute the custom research workflow.
        
        Args:
            co_scientist: The CoScientist instance
            
        Returns:
            Dict[str, Any]: Result of the workflow
        """
        results = {}
        
        for i, step in enumerate(self.steps):
            method_name = step["method"]
            params = step["params"]
            
            logger.info(f"Executing step {i+1}/{len(self.steps)}: {method_name}")
            
            if not hasattr(co_scientist, method_name):
                logger.error(f"Method {method_name} not found in CoScientist")
                continue
            
            method = getattr(co_scientist, method_name)
            result = method(**params)
            results[method_name] = result
        
        return results

class IterativeResearchWorkflow(ResearchWorkflow):
    """
    Iterative research workflow that runs until a condition is met.
    """
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 max_iterations: int = 10,
                 convergence_threshold: float = 0.01,
                 steps_per_iteration: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize an iterative research workflow.
        
        Args:
            name (str): Name of the workflow
            description (str, optional): Description of the workflow. Defaults to "".
            max_iterations (int, optional): Maximum number of iterations. Defaults to 10.
            convergence_threshold (float, optional): Convergence threshold. Defaults to 0.01.
            steps_per_iteration (List[Dict[str, Any]], optional): Steps to execute per iteration. Defaults to None.
        """
        super().__init__(name, description)
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.steps_per_iteration = steps_per_iteration or []
    
    def add_step(self, method_name: str, params: Dict[str, Any] = None):
        """
        Add a step to each iteration.
        
        Args:
            method_name (str): Name of the method to call
            params (Dict[str, Any], optional): Parameters for the method. Defaults to None.
        """
        self.steps_per_iteration.append({
            "method": method_name,
            "params": params or {}
        })
    
    def execute(self, co_scientist) -> Dict[str, Any]:
        """
        Execute the iterative research workflow.
        
        Args:
            co_scientist: The CoScientist instance
            
        Returns:
            Dict[str, Any]: Result of the workflow
        """
        results = {
            "iterations": [],
            "final_result": None,
            "convergence_achieved": False,
            "num_iterations": 0
        }
        
        prev_best_score = 0.0
        
        for iteration in range(self.max_iterations):
            logger.info(f"Iteration {iteration+1}/{self.max_iterations}")
            
            iteration_results = {}
            
            for i, step in enumerate(self.steps_per_iteration):
                method_name = step["method"]
                params = step["params"]
                
                logger.info(f"Executing step {i+1}/{len(self.steps_per_iteration)}: {method_name}")
                
                if not hasattr(co_scientist, method_name):
                    logger.error(f"Method {method_name} not found in CoScientist")
                    continue
                
                method = getattr(co_scientist, method_name)
                result = method(**params)
                iteration_results[method_name] = result
            
            results["iterations"].append(iteration_results)
            
            # Check for convergence
            if co_scientist.ranked_hypotheses:
                best_score = max(h.score for h in co_scientist.ranked_hypotheses)
                score_improvement = best_score - prev_best_score
                
                logger.info(f"Best score: {best_score}, Improvement: {score_improvement}")
                
                if score_improvement < self.convergence_threshold and iteration > 0:
                    logger.info("Convergence achieved")
                    results["convergence_achieved"] = True
                    break
                
                prev_best_score = best_score
        
        results["num_iterations"] = len(results["iterations"])
        
        if co_scientist.ranked_hypotheses:
            # Get the final result
            results["final_result"] = co_scientist.generate_research_summary()
        
        return results

def create_workflow(workflow_type: str, **kwargs) -> ResearchWorkflow:
    """
    Factory function to create a research workflow.
    
    Args:
        workflow_type (str): Type of workflow to create
        **kwargs: Additional arguments for the workflow
        
    Returns:
        ResearchWorkflow: Created research workflow
    """
    if workflow_type == "standard":
        return StandardResearchWorkflow(**kwargs)
    elif workflow_type == "custom":
        return CustomResearchWorkflow(**kwargs)
    elif workflow_type == "iterative":
        return IterativeResearchWorkflow(**kwargs)
    else:
        raise ValueError(f"Unknown workflow type: {workflow_type}") 
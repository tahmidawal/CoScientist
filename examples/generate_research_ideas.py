#!/usr/bin/env python3
"""
Example script demonstrating how to use the CoScientist library programmatically.
"""

import os
import sys
import json
import logging

# Add the parent directory to the Python path to import the coscientist package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coscientist.core.coscientist import CoScientist
from coscientist.core.workflow import create_workflow, CustomResearchWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def example_standard_workflow():
    """
    Example using the standard workflow.
    """
    print("\n===== STANDARD WORKFLOW EXAMPLE =====")
    
    # Create a CoScientist instance
    research_goal = "Develop deep learning methods for solving partial differential equations"
    co_scientist = CoScientist(research_goal=research_goal, verbose=True)
    
    # Run the standard workflow
    result = co_scientist.run_full_workflow(num_hypotheses=3, generations=2)
    
    # Print the result
    print("\nResearch Summary:")
    print(result.get("summary", "No summary available"))
    
    # Save the result to a file
    with open("standard_workflow_result.json", "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\nFull result saved to: standard_workflow_result.json")

def example_custom_workflow():
    """
    Example using a custom workflow.
    """
    print("\n===== CUSTOM WORKFLOW EXAMPLE =====")
    
    # Create a CoScientist instance
    research_goal = "Design a quantum algorithm for drug discovery"
    co_scientist = CoScientist(research_goal=research_goal, verbose=True)
    
    # Create a custom workflow
    workflow = CustomResearchWorkflow(name="Quantum Research Workflow")
    
    # Add steps to the workflow
    workflow.add_step("generate_hypotheses", {"num_hypotheses": 4})
    workflow.add_step("refine_hypotheses", {})
    workflow.add_step("rank_hypotheses", {})
    
    # Run the custom workflow
    result = co_scientist.run_custom_workflow(workflow)
    
    # Print the result
    print("\nWorkflow Results:")
    for step_name, step_result in result.items():
        print(f"\n{step_name.upper()}:")
        if step_name == "rank_hypotheses" and isinstance(step_result, list):
            for i, hypothesis in enumerate(step_result[:3], 1):
                print(f"{i}. {hypothesis.description} (Score: {hypothesis.score:.2f})")
    
    # Save the result to a file
    with open("custom_workflow_result.json", "w") as f:
        # Convert hypothesis objects to dictionaries
        serialized_result = {}
        for step_name, step_result in result.items():
            if isinstance(step_result, list) and step_result and hasattr(step_result[0], 'to_dict'):
                serialized_result[step_name] = [h.to_dict() for h in step_result]
            else:
                serialized_result[step_name] = step_result
        
        json.dump(serialized_result, f, indent=2)
    
    print(f"\nFull result saved to: custom_workflow_result.json")

def example_iterative_workflow():
    """
    Example using an iterative workflow.
    """
    print("\n===== ITERATIVE WORKFLOW EXAMPLE =====")
    
    # Create a CoScientist instance
    research_goal = "Develop a novel architecture for neural network pruning"
    co_scientist = CoScientist(research_goal=research_goal, verbose=True)
    
    # Create an iterative workflow
    workflow = create_workflow(
        "iterative",
        name="Neural Architecture Optimization",
        max_iterations=3,
        convergence_threshold=0.05
    )
    
    # Add steps to each iteration
    workflow.add_step("generate_hypotheses", {"num_hypotheses": 3})
    workflow.add_step("refine_hypotheses", {})
    workflow.add_step("evolve_hypotheses", {"generations": 1})
    workflow.add_step("rank_hypotheses", {})
    
    # Run the iterative workflow
    result = co_scientist.run_custom_workflow(workflow)
    
    # Print the result
    print("\nIterative Workflow Results:")
    print(f"Number of iterations: {result.get('num_iterations', 0)}")
    print(f"Convergence achieved: {result.get('convergence_achieved', False)}")
    
    if "final_result" in result and isinstance(result["final_result"], dict):
        final_result = result["final_result"]
        if "summary" in final_result:
            print("\nFinal Summary:")
            print(final_result["summary"])
    
    # Save the result to a file
    with open("iterative_workflow_result.json", "w") as f:
        # Serialize the result, handling hypothesis objects
        def serialize_obj(obj):
            if hasattr(obj, 'to_dict'):
                return obj.to_dict()
            
            if isinstance(obj, list):
                return [serialize_obj(item) for item in obj]
            
            if isinstance(obj, dict):
                return {k: serialize_obj(v) for k, v in obj.items()}
            
            return obj
        
        serialized_result = serialize_obj(result)
        json.dump(serialized_result, f, indent=2)
    
    print(f"\nFull result saved to: iterative_workflow_result.json")

def main():
    """
    Run all examples.
    """
    print("CoScientist Library Usage Examples")
    print("=================================")
    
    # Example 1: Standard workflow
    example_standard_workflow()
    
    # Example 2: Custom workflow
    example_custom_workflow()
    
    # Example 3: Iterative workflow
    example_iterative_workflow()
    
    print("\nAll examples completed.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Main script for running the AI Co-Scientist system.
"""

import os
import argparse
import json
import logging
from typing import Dict, Any, Optional

from coscientist.core.coscientist import CoScientist
from coscientist.core.workflow import create_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("coscientist.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="AI Co-Scientist")
    
    parser.add_argument(
        "--research-goal",
        type=str,
        help="Research goal to explore",
        required=True
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
        default=None
    )
    
    parser.add_argument(
        "--workflow",
        type=str,
        choices=["standard", "custom", "iterative"],
        default="standard",
        help="Workflow type to use"
    )
    
    parser.add_argument(
        "--workflow-config",
        type=str,
        help="Path to workflow configuration file",
        default=None
    )
    
    parser.add_argument(
        "--num-hypotheses",
        type=int,
        default=5,
        help="Number of hypotheses to generate"
    )
    
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of evolution generations"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output file",
        default="research_output.json"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """
    Load configuration from a file.
    
    Args:
        config_path (Optional[str]): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    if not config_path:
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        return {}

def save_output(output: Dict[str, Any], output_path: str):
    """
    Save output to a file.
    
    Args:
        output (Dict[str, Any]): Output to save
        output_path (str): Path to output file
    """
    try:
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Output saved to {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving output: {e}")

def main():
    """
    Main function.
    """
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    workflow_config = load_config(args.workflow_config)
    
    # Create CoScientist
    logger.info(f"Initializing AI Co-Scientist with research goal: {args.research_goal}")
    co_scientist = CoScientist(
        research_goal=args.research_goal,
        config_path=args.config,
        verbose=args.verbose
    )
    
    # Create workflow
    workflow_args = {
        "num_hypotheses": args.num_hypotheses,
        "generations": args.generations,
        **workflow_config
    }
    
    if args.workflow == "standard":
        # Use the built-in full workflow
        logger.info("Running standard workflow")
        output = co_scientist.run_full_workflow(
            num_hypotheses=args.num_hypotheses,
            generations=args.generations
        )
    
    else:
        # Create and run a custom workflow
        logger.info(f"Running {args.workflow} workflow")
        workflow = create_workflow(args.workflow, **workflow_args)
        output = co_scientist.run_custom_workflow(workflow)
    
    # Save output
    save_output(output, args.output)
    
    # Print summary
    print("\n===== RESEARCH SUMMARY =====")
    print(f"Research Goal: {args.research_goal}")
    
    if "summary" in output:
        print("\nSummary:")
        print(output["summary"])
    
    if "combined_hypothesis" in output:
        print("\nCombined Hypothesis:")
        print(output["combined_hypothesis"])
    
    if "next_steps" in output:
        print("\nNext Steps:")
        for i, step in enumerate(output["next_steps"], 1):
            print(f"{i}. {step}")
    
    print("\nFull output saved to:", args.output)
    print("============================")

if __name__ == "__main__":
    main() 
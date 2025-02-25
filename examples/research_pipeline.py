#!/usr/bin/env python3
"""
Research Pipeline Script

This script demonstrates the use of specialized agents in a research pipeline:
1. Manager Agent: Understands and elaborates on the research problem
2. Literature Agent: Finds related research and creates a vector store
3. Research Agent: Generates hypotheses and finds supporting documents
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add parent directory to path to import coscientist
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coscientist.agents.specialized_agents import ManagerAgent, LiteratureAgent, ResearchAgent
from coscientist.utils.llm_utils import GeminiProvider

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("research_pipeline.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Research Pipeline with Specialized Agents")
    
    parser.add_argument(
        "--research-goal",
        type=str,
        required=True,
        help="Research goal to explore"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research_output",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (or set GEMINI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--num-hypotheses",
        type=int,
        default=5,
        help="Number of hypotheses to generate"
    )
    
    parser.add_argument(
        "--max-papers",
        type=int,
        default=20,
        help="Maximum number of papers to retrieve"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def save_json(data: Any, filepath: str):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        filepath: Path to save the file
    """
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")

def save_text(text: str, filepath: str):
    """
    Save text to a file.
    
    Args:
        text: Text to save
        filepath: Path to save the file
    """
    try:
        with open(filepath, 'w') as f:
            f.write(text)
        logger.info(f"Saved text to {filepath}")
    except Exception as e:
        logger.error(f"Error saving text to {filepath}: {e}")

def format_hypothesis_with_documents(hypothesis_data: Dict[str, Any]) -> str:
    """
    Format a hypothesis with supporting documents as text.
    
    Args:
        hypothesis_data: Hypothesis data with supporting documents
        
    Returns:
        Formatted text
    """
    hypothesis = hypothesis_data["hypothesis"]
    supporting_docs = hypothesis_data["supporting_documents"]
    
    # Format hypothesis
    text = f"# RESEARCH HYPOTHESIS\n\n"
    text += f"## DESCRIPTION\n\n{hypothesis.description}\n\n"
    
    # Add components if available
    if hypothesis.components:
        text += "## COMPONENTS\n\n"
        for i, component in enumerate(hypothesis.components, 1):
            text += f"{i}. {component}\n"
        text += "\n"
    
    # Add methodologies if available
    if "methodologies" in hypothesis.metadata and hypothesis.metadata["methodologies"]:
        text += "## METHODOLOGIES\n\n"
        methodologies = hypothesis.metadata["methodologies"]
        if isinstance(methodologies, list):
            for i, methodology in enumerate(methodologies, 1):
                text += f"{i}. {methodology}\n"
        else:
            text += f"{methodologies}\n"
        text += "\n"
    
    # Add impact if available
    if "impact" in hypothesis.metadata and hypothesis.metadata["impact"]:
        text += f"## EXPECTED IMPACT\n\n{hypothesis.metadata['impact']}\n\n"
    
    # Add supporting documents
    if supporting_docs:
        text += "## SUPPORTING DOCUMENTS\n\n"
        for i, doc in enumerate(supporting_docs, 1):
            text += f"### {i}. {doc.get('title', 'Untitled')}\n\n"
            
            # Add authors if available
            authors = doc.get('authors', [])
            if authors:
                text += f"**Authors:** {', '.join(authors)}\n\n"
            
            # Add summary if available
            summary = doc.get('summary', '')
            if summary:
                text += f"**Summary:** {summary}\n\n"
            
            # Add URL if available
            url = doc.get('url', '')
            if url:
                text += f"**URL:** {url}\n\n"
    
    # Add metadata
    text += f"## METADATA\n\n"
    text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    text += f"Hypothesis ID: {hypothesis.id}\n"
    
    return text

def main():
    """Main function to run the research pipeline."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Set up API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.error("No Gemini API key provided. Use --api-key or set GEMINI_API_KEY environment variable.")
        sys.exit(1)
    
    # Configure Gemini provider
    os.environ["GEMINI_API_KEY"] = api_key
    
    # Initialize agents
    logger.info("Initializing specialized agents")
    
    manager_agent = ManagerAgent(
        model="gemini:gemini-2.0-flash-thinking-exp-01-21"
    )
    
    literature_agent = LiteratureAgent(
        model="gemini:gemini-2.0-flash-thinking-exp-01-21"
    )
    
    research_agent = ResearchAgent(
        model="gemini:gemini-2.0-flash-thinking-exp-01-21"
    )
    
    # Step 1: Manager Agent - Understand and elaborate on the research problem
    logger.info("Step 1: Manager Agent - Elaborating on research problem")
    elaboration_result = manager_agent.run({
        "research_goal": args.research_goal
    })
    
    # Save elaboration result
    save_json(elaboration_result, os.path.join(args.output_dir, "elaboration.json"))
    
    # Step 2: Literature Agent - Find related research and create vector store
    logger.info("Step 2: Literature Agent - Finding related research")
    literature_result = literature_agent.run({
        "research_goal": args.research_goal,
        "elaboration": elaboration_result,
        "max_papers": args.max_papers
    })
    
    # Save literature result
    save_json(literature_result, os.path.join(args.output_dir, "literature.json"))
    
    # Step 3: Research Agent - Generate hypotheses and find supporting documents
    logger.info("Step 3: Research Agent - Generating hypotheses with supporting documents")
    research_result = research_agent.run({
        "research_goal": args.research_goal,
        "elaboration": elaboration_result,
        "papers": literature_result.get("papers", []),
        "literature_agent": literature_agent,
        "num_hypotheses": args.num_hypotheses
    })
    
    # Save research result
    save_json(
        {
            "research_goal": args.research_goal,
            "hypotheses": [
                {
                    "id": h["hypothesis"].id,
                    "description": h["hypothesis"].description,
                    "components": h["hypothesis"].components,
                    "score": h["hypothesis"].score,
                    "metadata": h["hypothesis"].metadata,
                    "supporting_documents": [
                        {
                            "title": doc.get("title", ""),
                            "authors": doc.get("authors", []),
                            "summary": doc.get("summary", ""),
                            "url": doc.get("url", "")
                        }
                        for doc in h["supporting_documents"]
                    ]
                }
                for h in research_result.get("hypotheses", [])
            ]
        },
        os.path.join(args.output_dir, "research_result.json")
    )
    
    # Save each hypothesis with supporting documents as a text file
    for i, hypothesis_data in enumerate(research_result.get("hypotheses", []), 1):
        hypothesis = hypothesis_data["hypothesis"]
        
        # Generate filename
        filename = f"hypothesis_{i}_{hypothesis.id[:8]}.txt"
        filepath = os.path.join(args.output_dir, filename)
        
        # Format hypothesis with supporting documents
        formatted_text = format_hypothesis_with_documents(hypothesis_data)
        
        # Save to file
        save_text(formatted_text, filepath)
    
    # Print summary
    print("\n===== RESEARCH PIPELINE COMPLETED =====")
    print(f"Research Goal: {args.research_goal}")
    print(f"Number of hypotheses generated: {len(research_result.get('hypotheses', []))}")
    print(f"Number of papers found: {len(literature_result.get('papers', []))}")
    print(f"Output directory: {args.output_dir}")
    print("=======================================")

if __name__ == "__main__":
    main() 
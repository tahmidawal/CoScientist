#!/usr/bin/env python3
"""
Script to generate the top 3 research ideas with academic references and save implementation details.

This script uses the CoScientist framework to:
1. Generate multiple research hypotheses using Gemini
2. Rank them and select the top 3
3. Search academic sources for related papers
4. Create structured implementation plans
5. Save everything to .txt files
"""

import os
import sys
import json
import logging
import argparse
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
import google.generativeai as genai

# Add parent directory to path to import coscientist
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coscientist.core.hypothesis import Hypothesis
from coscientist.tools.gemini_tools import GeminiResearchTool
from coscientist.tools.academic_search import search_academic_sources

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("top_ideas_research.log")
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate top research ideas with academic references")
    
    parser.add_argument(
        "--research-goal",
        type=str,
        default="Developing efficient and accurate multimodal deep learning models for complex data integration",
        help="Research goal to explore"
    )
    
    parser.add_argument(
        "--num-hypotheses",
        type=int,
        default=10,
        help="Number of initial hypotheses to generate"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="research_ideas",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="Gemini API key (or set GEMINI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()

def generate_hypotheses(tool: GeminiResearchTool, research_goal: str, num_hypotheses: int = 10) -> List[Hypothesis]:
    """
    Generate research hypotheses using GeminiResearchTool.
    
    Args:
        tool: GeminiResearchTool instance
        research_goal: Research goal to explore
        num_hypotheses: Number of hypotheses to generate
        
    Returns:
        List of Hypothesis objects
    """
    logger.info(f"Generating {num_hypotheses} hypotheses for research goal: {research_goal}")
    
    # Use the generate_research_hypotheses method
    try:
        hypothesis_data = tool.generate_research_hypotheses(research_goal, num_hypotheses)
        
        # Convert to Hypothesis objects
        hypotheses = []
        for i, hyp in enumerate(hypothesis_data):
            if isinstance(hyp, dict) and "description" in hyp:
                description = hyp["description"]
                components = hyp.get("components", [])
                # Create Hypothesis object without specifying ID (it will generate its own)
                h = Hypothesis(
                    description=description,
                    components=components,
                    score=0,  # Initial score
                    metadata={"index": i+1}  # Store original index
                )
                hypotheses.append(h)
            else:
                logger.warning(f"Skipping invalid hypothesis format: {hyp}")
        
        logger.info(f"Successfully generated {len(hypotheses)} hypotheses")
        return hypotheses
    except Exception as e:
        logger.error(f"Error generating hypotheses: {e}")
        return []

def rank_hypotheses(tool: GeminiResearchTool, hypotheses: List[Hypothesis], research_goal: str) -> List[Hypothesis]:
    """
    Rank hypotheses using GeminiResearchTool.
    
    Args:
        tool: GeminiResearchTool instance
        hypotheses: List of hypotheses to rank
        research_goal: Research goal to evaluate against
        
    Returns:
        List of ranked hypotheses
    """
    if not hypotheses:
        logger.warning("No hypotheses to rank")
        return []
    
    logger.info(f"Ranking {len(hypotheses)} hypotheses")
    
    # Create a mapping of index to hypothesis ID for easier reference
    # Use the index from metadata instead of enumeration
    hypothesis_index_map = {}
    hypothesis_display_list = []
    
    for h in hypotheses:
        index = h.metadata.get("index", 0)
        display_id = f"H{index}"
        hypothesis_index_map[display_id] = h.id
        hypothesis_display_list.append(f"{display_id}: {h.description}")
    
    prompt = f"""
    As an AI Co-Scientist, rank the following research hypotheses based on their potential to address this research goal:
    
    RESEARCH GOAL: {research_goal}
    
    HYPOTHESES:
    {chr(10).join(hypothesis_display_list)}
    
    For each hypothesis, provide a score from 1-10 and a brief justification. Then, provide a final ranking of the top 3 hypotheses.
    
    Format your response as a JSON array with objects containing:
    - "id": hypothesis ID (use the format H1, H2, etc.)
    - "score": score from 1-10
    - "justification": brief explanation of the score
    
    After the array, provide a "top_three" array with the IDs of the top 3 hypotheses in descending order of quality.
    """
    
    try:
        response = tool.chat_session.send_message(prompt)
        response_text = response.text
        
        # Extract JSON
        try:
            # Find JSON in the response
            json_start = response_text.find("[")
            json_end = response_text.rfind("]") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                rankings = json.loads(json_str)
                
                # Update hypothesis scores
                for rank in rankings:
                    if isinstance(rank, dict) and "id" in rank and "score" in rank:
                        display_id = rank["id"]
                        score = float(rank["score"])
                        justification = rank.get("justification", "")
                        
                        # Find matching hypothesis and update score
                        for h in hypotheses:
                            if h.id == hypothesis_index_map.get(display_id):
                                h.score = score
                                h.metadata["justification"] = justification
                                h.metadata["display_id"] = display_id
                                break
                
                # Look for top_three key
                top_three_start = response_text.find('"top_three"')
                if top_three_start >= 0:
                    top_three_json_start = response_text.find("[", top_three_start)
                    top_three_json_end = response_text.find("]", top_three_json_start) + 1
                    
                    if top_three_json_start >= 0 and top_three_json_end > top_three_json_start:
                        top_three_str = response_text[top_three_json_start:top_three_json_end]
                        try:
                            top_three_ids = json.loads(top_three_str)
                            
                            # Sort hypotheses based on top_three order
                            result = []
                            for display_id in top_three_ids:
                                real_id = hypothesis_index_map.get(display_id)
                                for h in hypotheses:
                                    if h.id == real_id:
                                        result.append(h)
                                        break
                            
                            # Add any remaining hypotheses sorted by score
                            remaining = [h for h in hypotheses if h.id not in [hyp.id for hyp in result]]
                            remaining.sort(key=lambda x: x.score, reverse=True)
                            result.extend(remaining)
                            
                            return result[:3]  # Return top 3
                        except json.JSONDecodeError:
                            # Fall back to score-based sorting
                            pass
                
                # Default: sort by score
                hypotheses.sort(key=lambda x: x.score, reverse=True)
                return hypotheses[:3]  # Return top 3
            else:
                # If JSON parsing fails, just sort by score
                hypotheses.sort(key=lambda x: x.score, reverse=True)
                return hypotheses[:3]  # Return top 3
        except Exception as e:
            logger.error(f"Error parsing ranking response: {e}")
            # Fall back to random selection if parsing fails
            random.shuffle(hypotheses)
            return hypotheses[:3]
    except Exception as e:
        logger.error(f"Error ranking hypotheses: {e}")
        # Fall back to random selection
        random.shuffle(hypotheses)
        return hypotheses[:3]

def create_implementation_plan(tool: GeminiResearchTool, hypothesis: Hypothesis) -> Dict[str, Any]:
    """
    Create a detailed implementation plan for a hypothesis.
    
    Args:
        tool: GeminiResearchTool instance
        hypothesis: The research hypothesis
        
    Returns:
        Dict containing the implementation plan
    """
    prompt = f"""
    As an AI Co-Scientist, create a detailed implementation plan for the following research hypothesis:
    
    HYPOTHESIS: {hypothesis.description}
    
    Please provide a comprehensive implementation plan including:
    
    1. Technical approach and methodology
    2. Required datasets and resources
    3. Algorithmic details and pseudocode for key components
    4. Evaluation metrics and experimental setup
    5. Implementation timeline with milestones
    6. Potential challenges and mitigation strategies
    7. Expected outcomes and impact
    
    Format your response as a JSON object with the following fields:
    - technical_approach: detailed description of the technical approach
    - datasets_resources: list of required datasets and resources with brief descriptions
    - algorithms: list of algorithms with pseudocode where relevant
    - evaluation: description of evaluation methodology, metrics, and experimental setup
    - timeline: list of major milestones and estimated timeframes
    - challenges: list of potential challenges and mitigation strategies
    - outcomes: expected outcomes and impact
    """
    
    # Generate implementation plan
    try:
        response = tool.chat_session.send_message(prompt)
        response_text = response.text
        
        # Try to extract JSON
        try:
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # Return structured but not JSON format
                return {
                    "implementation_plan": response_text,
                    "error": "Could not parse JSON"
                }
        except json.JSONDecodeError:
            return {
                "implementation_plan": response_text,
                "error": "JSON parsing error"
            }
    except Exception as e:
        logger.error(f"Error generating implementation plan: {e}")
        return {
            "error": str(e)
        }

def format_implementation_plan(plan: Dict[str, Any], hypothesis: Hypothesis, references: Dict[str, Any]) -> str:
    """
    Format the implementation plan and references into a structured text document.
    
    Args:
        plan: Implementation plan dictionary
        hypothesis: Research hypothesis
        references: Academic references
        
    Returns:
        Formatted text for saving to file
    """
    # Get display ID or use UUID
    display_id = hypothesis.metadata.get("display_id", f"H-{hypothesis.id[:8]}")
    
    # Start with hypothesis
    formatted_text = f"# RESEARCH IMPLEMENTATION PLAN\n\n"
    formatted_text += f"## HYPOTHESIS {display_id}\n\n{hypothesis.description}\n\n"
    
    # Add score if available
    if hypothesis.score > 0:
        formatted_text += f"**Score**: {hypothesis.score:.2f}/10\n\n"
    
    # Add justification if available
    if "justification" in hypothesis.metadata:
        formatted_text += f"**Justification**: {hypothesis.metadata['justification']}\n\n"
    
    # Add components if available
    if hypothesis.components:
        formatted_text += "## COMPONENTS\n\n"
        for i, component in enumerate(hypothesis.components, 1):
            formatted_text += f"{i}. {component}\n"
        formatted_text += "\n"
    
    # Add technical approach
    if "technical_approach" in plan:
        formatted_text += f"## TECHNICAL APPROACH\n\n{plan['technical_approach']}\n\n"
    
    # Add datasets and resources
    if "datasets_resources" in plan:
        formatted_text += "## DATASETS AND RESOURCES\n\n"
        if isinstance(plan["datasets_resources"], list):
            for i, resource in enumerate(plan["datasets_resources"], 1):
                formatted_text += f"{i}. {resource}\n"
        else:
            formatted_text += f"{plan['datasets_resources']}\n"
        formatted_text += "\n"
    
    # Add algorithms
    if "algorithms" in plan:
        formatted_text += "## ALGORITHMS AND METHODS\n\n"
        if isinstance(plan["algorithms"], list):
            for i, algorithm in enumerate(plan["algorithms"], 1):
                formatted_text += f"### Algorithm {i}\n\n"
                if isinstance(algorithm, dict) and "name" in algorithm:
                    formatted_text += f"**Name**: {algorithm['name']}\n\n"
                    if "description" in algorithm:
                        formatted_text += f"{algorithm['description']}\n\n"
                    if "pseudocode" in algorithm:
                        formatted_text += "```\n" + algorithm["pseudocode"] + "\n```\n\n"
                else:
                    formatted_text += f"{algorithm}\n\n"
        else:
            formatted_text += f"{plan['algorithms']}\n\n"
    
    # Add evaluation
    if "evaluation" in plan:
        formatted_text += f"## EVALUATION METHODOLOGY\n\n{plan['evaluation']}\n\n"
    
    # Add timeline
    if "timeline" in plan:
        formatted_text += "## IMPLEMENTATION TIMELINE\n\n"
        if isinstance(plan["timeline"], list):
            for i, milestone in enumerate(plan["timeline"], 1):
                formatted_text += f"{i}. {milestone}\n"
        else:
            formatted_text += f"{plan['timeline']}\n"
        formatted_text += "\n"
    
    # Add challenges
    if "challenges" in plan:
        formatted_text += "## POTENTIAL CHALLENGES AND MITIGATION\n\n"
        if isinstance(plan["challenges"], list):
            for i, challenge in enumerate(plan["challenges"], 1):
                formatted_text += f"{i}. {challenge}\n"
        else:
            formatted_text += f"{plan['challenges']}\n"
        formatted_text += "\n"
    
    # Add expected outcomes
    if "outcomes" in plan:
        formatted_text += f"## EXPECTED OUTCOMES AND IMPACT\n\n{plan['outcomes']}\n\n"
    
    # Add references
    if "papers" in references and references["papers"]:
        formatted_text += "## ACADEMIC REFERENCES\n\n"
        formatted_text += references["bibliography"]
        formatted_text += "\n\n"
    
    # Add metadata
    formatted_text += f"## METADATA\n\n"
    formatted_text += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    formatted_text += f"Hypothesis ID: {hypothesis.id}\n"
    
    return formatted_text

def generate_filename_from_hypothesis(hypothesis: Hypothesis) -> str:
    """
    Generate a filename from a hypothesis.
    
    Args:
        hypothesis: Research hypothesis
        
    Returns:
        Suitable filename
    """
    # Get display ID or use UUID
    display_id = hypothesis.metadata.get("display_id", f"H-{hypothesis.id[:8]}")
    
    # Take first 40 chars of hypothesis, replace spaces with underscores
    filename_base = hypothesis.description[:40].replace(" ", "_")
    
    # Remove special characters
    filename_base = "".join(c for c in filename_base if c.isalnum() or c == "_")
    
    # Add hypothesis ID
    return f"{display_id}_{filename_base}.txt"

def main():
    """Main function to run the script."""
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
    
    # Configure genai
    genai.configure(api_key=api_key)
    
    # Initialize GeminiResearchTool
    tool = GeminiResearchTool(api_key=api_key)
    
    logger.info(f"Initialized GeminiResearchTool with research goal: {args.research_goal}")
    
    # Generate hypotheses
    hypotheses = generate_hypotheses(tool, args.research_goal, args.num_hypotheses)
    
    if not hypotheses:
        logger.error("Failed to generate any hypotheses. Exiting.")
        sys.exit(1)
    
    # Save all generated hypotheses
    all_hypotheses_data = []
    for h in hypotheses:
        all_hypotheses_data.append({
            "id": h.id,
            "description": h.description,
            "components": h.components
        })
    
    with open(os.path.join(args.output_dir, "all_hypotheses.json"), "w") as f:
        json.dump(all_hypotheses_data, f, indent=2)
    
    # Rank hypotheses and get top 3
    top_hypotheses = rank_hypotheses(tool, hypotheses, args.research_goal)
    
    if not top_hypotheses:
        logger.error("Failed to rank hypotheses. Exiting.")
        sys.exit(1)
    
    logger.info(f"Selected top {len(top_hypotheses)} hypotheses")
    
    # Save top hypotheses
    top_hypotheses_data = []
    for h in top_hypotheses:
        top_hypotheses_data.append({
            "id": h.id,
            "display_id": h.metadata.get("display_id", f"H-{h.id[:8]}"),
            "description": h.description,
            "components": h.components,
            "score": h.score,
            "justification": h.metadata.get("justification", "")
        })
    
    with open(os.path.join(args.output_dir, "top_hypotheses.json"), "w") as f:
        json.dump(top_hypotheses_data, f, indent=2)
    
    # Process each top hypothesis
    for i, hypothesis in enumerate(top_hypotheses, 1):
        display_id = hypothesis.metadata.get("display_id", f"H-{hypothesis.id[:8]}")
        logger.info(f"Processing top hypothesis {i}/{len(top_hypotheses)}: {display_id}")
        
        # Search academic sources
        logger.info(f"Searching academic sources for hypothesis: {display_id}")
        academic_references = search_academic_sources(hypothesis.description, max_results=5)
        
        # Create implementation plan
        logger.info(f"Creating implementation plan for hypothesis: {display_id}")
        implementation_plan = create_implementation_plan(tool, hypothesis)
        
        # Save implementation plan to JSON
        plan_filename = f"{display_id}_implementation_plan.json"
        with open(os.path.join(args.output_dir, plan_filename), "w") as f:
            json.dump(implementation_plan, f, indent=2)
        
        # Format and save as text file
        logger.info(f"Formatting and saving implementation plan for hypothesis: {display_id}")
        formatted_plan = format_implementation_plan(implementation_plan, hypothesis, academic_references)
        
        # Generate filename
        text_filename = generate_filename_from_hypothesis(hypothesis)
        with open(os.path.join(args.output_dir, text_filename), "w") as f:
            f.write(formatted_plan)
        
        logger.info(f"Saved implementation plan to {text_filename}")
    
    logger.info("All top research ideas have been processed and saved.")
    print(f"\nCompleted! {len(top_hypotheses)} top research ideas with implementation plans have been saved to {args.output_dir}/")

if __name__ == "__main__":
    main() 
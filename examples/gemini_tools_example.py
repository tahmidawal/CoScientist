#!/usr/bin/env python3
"""
Example demonstrating how to use the GeminiResearchTool for research tasks.
"""

import os
import sys
import json
import logging

# Add the parent directory to the Python path to import the coscientist package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coscientist.tools.gemini_tools import GeminiResearchTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def save_json(data, filename):
    """Save data to a JSON file with nice formatting."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {filename}")

def main():
    """
    Example of using the GeminiResearchTool.
    """
    # Set Gemini API key
    api_key = "AIzaSyC7jgqPzB_krLfQG72F_IUt6afZoxubqzk"
    os.environ["GEMINI_API_KEY"] = api_key
    
    # Initialize the research tool
    tool = GeminiResearchTool(api_key=api_key)
    
    # Example research problem
    research_problem = "Developing self-supervised learning methods for multimodal foundation models that can efficiently transfer to downstream tasks with minimal labeled data."
    
    # 1. Analyze the research problem
    print("\n===== STEP 1: ANALYZE RESEARCH PROBLEM =====")
    analysis = tool.analyze_research_problem(research_problem)
    
    print("Research Problem Analysis:")
    if "challenges" in analysis:
        print("\nKey Challenges:")
        for challenge in analysis["challenges"]:
            print(f"- {challenge}")
    
    if "approaches" in analysis:
        print("\nPotential Approaches:")
        for approach in analysis["approaches"]:
            print(f"- {approach}")
    
    # Save the full analysis
    save_json(analysis, "research_problem_analysis.json")
    
    # 2. Generate research hypotheses
    print("\n===== STEP 2: GENERATE RESEARCH HYPOTHESES =====")
    hypotheses = tool.generate_research_hypotheses(
        research_goal=research_problem,
        num_hypotheses=3,
        temperature=0.8
    )
    
    print("Generated Hypotheses:")
    for i, hypothesis in enumerate(hypotheses):
        print(f"\nHypothesis {i+1}: {hypothesis.get('description', '')[:200]}...")
    
    # Save the hypotheses
    save_json(hypotheses, "generated_hypotheses.json")
    
    # 3. Evaluate a hypothesis
    print("\n===== STEP 3: EVALUATE A HYPOTHESIS =====")
    if hypotheses and "description" in hypotheses[0]:
        evaluation = tool.evaluate_hypothesis(hypotheses[0]["description"])
        
        print("Hypothesis Evaluation:")
        
        if "theoretical_soundness" in evaluation:
            print(f"\nTheoretical Soundness: {evaluation['theoretical_soundness']}/10")
            print(f"Practical Feasibility: {evaluation['practical_feasibility']}/10")
            print(f"Originality: {evaluation['originality']}/10")
            print(f"Potential Impact: {evaluation['potential_impact']}/10")
            print(f"Overall Score: {evaluation['overall_score']}/10")
        
        if "strengths" in evaluation:
            print("\nStrengths:")
            for strength in evaluation["strengths"]:
                print(f"- {strength}")
        
        if "suggestions" in evaluation:
            print("\nSuggestions for Improvement:")
            for suggestion in evaluation["suggestions"]:
                print(f"- {suggestion}")
        
        # Save the evaluation
        save_json(evaluation, "hypothesis_evaluation.json")
    
    # 4. Combine hypotheses
    print("\n===== STEP 4: COMBINE HYPOTHESES =====")
    if len(hypotheses) >= 2:
        # Extract descriptions from the first two hypotheses
        descriptions = [h.get("description", "") for h in hypotheses[:2]]
        
        combined = tool.combine_hypotheses(descriptions)
        
        print("Combined Research Approach:")
        if "combined_hypothesis" in combined:
            print(f"\n{combined['combined_hypothesis'][:300]}...")
        
        if "strengths" in combined:
            print("\nStrengths of Combined Approach:")
            for strength in combined["strengths"]:
                print(f"- {strength}")
        
        if "next_steps" in combined:
            print("\nNext Steps:")
            for step in combined["next_steps"]:
                print(f"- {step}")
        
        # Save the combined approach
        save_json(combined, "combined_approach.json")
    
    print("\nExample completed.")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Example demonstrating how to use the CoScientist library with Google Gemini.
"""

import os
import sys
import json
import logging
import getpass
from dotenv import load_dotenv

# Add the parent directory to the Python path to import the coscientist package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from coscientist.core.coscientist import CoScientist
from coscientist.utils.llm_utils import GeminiProvider, get_llm_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """
    Example using Google Gemini for hypothesis generation.
    """
    # Load environment variables from .env file if it exists
    load_dotenv()
    
    # Set Gemini API key - try to get from environment, otherwise prompt for it
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_api_key:
        print("No GEMINI_API_KEY found in environment variables.")
        # Note: In production, use a proper secrets management system
        gemini_api_key = getpass.getpass("Enter your Gemini API key (input will be hidden): ")
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    
    # Test direct use of the GeminiProvider
    print("\n===== TESTING GEMINI PROVIDER DIRECTLY =====")
    try:
        provider = GeminiProvider(api_key=gemini_api_key)
        response = provider.generate(
            prompt="What are three promising research directions in quantum machine learning?",
            temperature=0.7,
            max_tokens=500
        )
        print("Gemini response:")
        print(response)
    except Exception as e:
        print(f"Error using Gemini directly: {e}")
        
    # Test using get_llm_response with Gemini
    print("\n===== TESTING LLM RESPONSE WITH GEMINI =====")
    try:
        response = get_llm_response(
            prompt="Propose a novel research hypothesis about neural networks for time series prediction.",
            provider="gemini",
            temperature=0.8,
            max_tokens=500
        )
        print("Response using get_llm_response:")
        print(response)
    except Exception as e:
        print(f"Error using get_llm_response: {e}")
    
    # Create a CoScientist instance with custom config using Gemini
    print("\n===== COSCIENTIST WITH GEMINI =====")
    config = {
        "agents": {
            "generation": {
                "model": "gemini:gemini-2.0-flash-thinking-exp-01-21",
                "params": {
                    "temperature": 0.8
                }
            },
            "reflection": {
                "model": "gemini:gemini-2.0-flash-thinking-exp-01-21",
                "params": {
                    "temperature": 0.5
                }
            }
        }
    }
    
    with open("gemini_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    try:
        co_scientist = CoScientist(
            research_goal="Developing more efficient neural network architectures for edge devices",
            config_path="gemini_config.json",
            verbose=True
        )
        
        # Generate hypotheses
        hypotheses = co_scientist.generate_hypotheses(num_hypotheses=2)
        
        # Print hypotheses
        print("\nGenerated Hypotheses:")
        for i, hypothesis in enumerate(hypotheses, 1):
            print(f"{i}. {hypothesis.description}")
            print(f"   Components: {', '.join(hypothesis.components) if hypothesis.components else 'None'}")
            print()
            
    except Exception as e:
        print(f"Error using CoScientist with Gemini: {e}")
    
    print("Example completed.")

if __name__ == "__main__":
    main() 
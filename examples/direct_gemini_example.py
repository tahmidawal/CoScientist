#!/usr/bin/env python3
"""
Example demonstrating direct usage of the Google Gemini model.
"""

import os
import sys
import json
import logging

# Add the parent directory to the Python path to import the coscientist package if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """
    Direct example of using Google Gemini model.
    """
    # Set the API key - in a production environment, this should be stored securely
    # and not hardcoded in the script
    api_key = "AIzaSyC7jgqPzB_krLfQG72F_IUt6afZoxubqzk"
    os.environ["GEMINI_API_KEY"] = api_key
    
    try:
        import google.generativeai as genai

        # Configure the API
        genai.configure(api_key=api_key)

        # Create the model with generation configuration
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 2000,  # Reduced for example purposes
            "response_mime_type": "text/plain",
        }

        model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-thinking-exp-01-21",
            generation_config=generation_config,
        )

        # Start a chat session
        chat_session = model.start_chat(history=[])
        
        # Example research prompts to test
        research_prompts = [
            "Propose three innovative research directions for combining reinforcement learning with graph neural networks.",
            "What are the most promising approaches for developing energy-efficient deep learning models?",
            "Suggest a novel hypothesis about how attention mechanisms could be improved in transformer models."
        ]
        
        # Test each prompt
        for i, prompt in enumerate(research_prompts, 1):
            print(f"\n===== RESEARCH PROMPT {i} =====")
            print(f"Prompt: {prompt}")
            
            # Send the message and get response
            response = chat_session.send_message(prompt)
            
            print("\nResponse:")
            print(response.text)
            print("=" * 50)

    except ImportError:
        logger.error("Google Generative AI package not installed. Install with: pip install google-generativeai")
    except Exception as e:
        logger.error(f"Error using Google Gemini: {e}")

if __name__ == "__main__":
    main() 
"""
Gemini-powered research tools.

This module contains tools that leverage Google's Gemini models.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class GeminiResearchTool:
    """
    Tool for research tasks powered by Google's Gemini model.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash-thinking-exp-01-21"):
        """
        Initialize the Gemini research tool.
        
        Args:
            api_key (str, optional): Google Gemini API key. Defaults to None.
            model (str, optional): Model to use. Defaults to "gemini-2.0-flash-thinking-exp-01-21".
        """
        # Use provided API key or try to get from environment
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
        if not self.api_key:
            logger.warning("No Gemini API key provided. Set GEMINI_API_KEY environment variable.")
        
        self.model = model
        
        # Import Google Generative AI if available
        try:
            import google.generativeai as genai
            self.genai = genai
            self.genai_available = True
            # Configure the API
            self.genai.configure(api_key=self.api_key)
            
            # Default generation config
            self.generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 64,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            }
            
            # Initialize model
            self.model_instance = self.genai.GenerativeModel(
                model_name=self.model,
                generation_config=self.generation_config,
            )
            
            # Start a chat session
            self.chat_session = self.model_instance.start_chat(history=[])
            
        except ImportError:
            logger.warning("Google Generative AI package not installed. Install with: pip install google-generativeai")
            self.genai_available = False
    
    def analyze_research_problem(self, problem_statement: str) -> Dict[str, Any]:
        """
        Analyze a research problem and provide insights.
        
        Args:
            problem_statement (str): The research problem to analyze
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        if not self.genai_available:
            logger.error("Google Generative AI package not installed.")
            return {"error": "Google Generative AI package not installed"}
        
        prompt = f"""
        As an AI Co-Scientist, please analyze the following research problem:
        
        RESEARCH PROBLEM: {problem_statement}
        
        Provide a comprehensive analysis including:
        1. Key challenges and obstacles
        2. Related research areas and subfields
        3. Potential methodological approaches
        4. Datasets or resources that might be relevant
        5. Evaluation metrics that could be used
        
        Format your response as a structured JSON object with the following fields:
        - challenges: list of key challenges
        - related_areas: list of related research areas
        - approaches: list of potential methodological approaches
        - resources: list of relevant datasets or resources
        - metrics: list of possible evaluation metrics
        - overall_assessment: brief overall assessment of the research problem
        """
        
        try:
            response = self.chat_session.send_message(prompt)
            response_text = response.text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON content
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # If no JSON found, return the full text
                    return {"analysis": response_text}
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, return the full text
                return {"analysis": response_text}
                
        except Exception as e:
            logger.error(f"Error analyzing research problem: {e}")
            return {"error": str(e)}
    
    def generate_research_hypotheses(self, 
                                   research_goal: str, 
                                   num_hypotheses: int = 5,
                                   temperature: float = 0.8) -> List[Dict[str, Any]]:
        """
        Generate research hypotheses for a research goal.
        
        Args:
            research_goal (str): The research goal
            num_hypotheses (int, optional): Number of hypotheses to generate. Defaults to 5.
            temperature (float, optional): Temperature parameter. Defaults to 0.8.
            
        Returns:
            List[Dict[str, Any]]: Generated hypotheses
        """
        if not self.genai_available:
            logger.error("Google Generative AI package not installed.")
            return [{"error": "Google Generative AI package not installed"}]
        
        # Store original temperature
        original_temp = self.generation_config.get("temperature", 0.7)
        
        # Create a new generation config with the desired temperature
        temp_generation_config = dict(self.generation_config)
        temp_generation_config["temperature"] = temperature
        
        # Create a new model instance with the temporary config
        temp_model = self.genai.GenerativeModel(
            model_name=self.model,
            generation_config=temp_generation_config,
        )
        
        # Create a new chat session with the temporary model
        temp_chat_session = temp_model.start_chat(history=[])
        
        prompt = f"""
        As an AI Co-Scientist, generate {num_hypotheses} innovative and scientifically sound research hypotheses 
        for the following research goal:
        
        RESEARCH GOAL: {research_goal}
        
        For each hypothesis, include:
        1. A clear and detailed description
        2. Key components or elements involved
        3. Potential methodologies or approaches to test it
        4. Expected impact if validated
        
        Format your response as a JSON array of objects, where each object represents a hypothesis with the following fields:
        - description: detailed description of the hypothesis
        - components: array of key components
        - methodologies: array of potential methodologies to test the hypothesis
        - impact: expected impact if validated
        """
        
        try:
            response = temp_chat_session.send_message(prompt)
            response_text = response.text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON array
                json_start = response_text.find("[")
                json_end = response_text.rfind("]") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # If no JSON array found, try to parse manually
                    hypotheses = []
                    
                    # Simple parsing based on numbered hypotheses
                    import re
                    hypothesis_sections = re.split(r'Hypothesis \d+:|Hypothesis #\d+:', response_text)
                    
                    if len(hypothesis_sections) > 1:
                        # Skip the first element which is usually empty or contains preamble
                        for i, section in enumerate(hypothesis_sections[1:]):
                            hypotheses.append({
                                "description": section.strip(),
                                "components": [],
                                "methodologies": [],
                                "impact": ""
                            })
                    else:
                        # If no clear separation, just return the whole text
                        hypotheses.append({
                            "description": response_text,
                            "components": [],
                            "methodologies": [],
                            "impact": ""
                        })
                    
                    return hypotheses
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, return text in a structured format
                return [{
                    "description": response_text,
                    "components": [],
                    "methodologies": [],
                    "impact": ""
                }]
                
        except Exception as e:
            logger.error(f"Error generating research hypotheses: {e}")
            return [{"error": str(e)}]
    
    def evaluate_hypothesis(self, hypothesis: str) -> Dict[str, Any]:
        """
        Evaluate a research hypothesis.
        
        Args:
            hypothesis (str): The hypothesis to evaluate
            
        Returns:
            Dict[str, Any]: Evaluation results
        """
        if not self.genai_available:
            logger.error("Google Generative AI package not installed.")
            return {"error": "Google Generative AI package not installed"}
        
        prompt = f"""
        As an AI Co-Scientist, evaluate the following research hypothesis:
        
        HYPOTHESIS: {hypothesis}
        
        Provide a comprehensive evaluation including:
        1. Theoretical soundness (0-10 scale)
        2. Practical feasibility (0-10 scale)
        3. Originality (0-10 scale)
        4. Potential impact (0-10 scale)
        5. Strengths of the hypothesis
        6. Weaknesses or limitations
        7. Suggestions for improvement
        
        Format your response as a JSON object with the following fields:
        - theoretical_soundness: numeric score
        - practical_feasibility: numeric score
        - originality: numeric score
        - potential_impact: numeric score
        - overall_score: weighted average of the above scores
        - strengths: array of strengths
        - weaknesses: array of weaknesses
        - suggestions: array of suggestions for improvement
        - detailed_assessment: textual detailed assessment
        """
        
        try:
            response = self.chat_session.send_message(prompt)
            response_text = response.text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON content
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # If no JSON found, return the full text
                    return {"evaluation": response_text}
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, return the full text
                return {"evaluation": response_text}
                
        except Exception as e:
            logger.error(f"Error evaluating hypothesis: {e}")
            return {"error": str(e)}

    def combine_hypotheses(self, hypotheses: List[str]) -> Dict[str, Any]:
        """
        Combine multiple hypotheses into a unified research approach.
        
        Args:
            hypotheses (List[str]): List of hypotheses to combine
            
        Returns:
            Dict[str, Any]: Combined research approach
        """
        if not self.genai_available:
            logger.error("Google Generative AI package not installed.")
            return {"error": "Google Generative AI package not installed"}
        
        hypotheses_text = "\n".join([f"HYPOTHESIS {i+1}: {h}" for i, h in enumerate(hypotheses)])
        
        prompt = f"""
        As an AI Co-Scientist, analyze the following research hypotheses and combine them into a unified research approach:
        
        {hypotheses_text}
        
        Create a synthesis that:
        1. Identifies complementary aspects across hypotheses
        2. Resolves any contradictions
        3. Creates a more comprehensive and robust research approach
        4. Leverages the strengths of each individual hypothesis
        
        Format your response as a JSON object with the following fields:
        - combined_hypothesis: detailed description of the combined hypothesis
        - components: array of key components
        - methodologies: array of methodologies to test the hypothesis
        - strengths: array of strengths of this combined approach
        - limitations: array of limitations
        - next_steps: array of suggested next steps
        """
        
        try:
            response = self.chat_session.send_message(prompt)
            response_text = response.text
            
            # Try to extract JSON from the response
            try:
                # Look for JSON content
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    return json.loads(json_str)
                else:
                    # If no JSON found, return the full text
                    return {"combined_approach": response_text}
                    
            except json.JSONDecodeError:
                # If JSON parsing fails, return the full text
                return {"combined_approach": response_text}
                
        except Exception as e:
            logger.error(f"Error combining hypotheses: {e}")
            return {"error": str(e)} 
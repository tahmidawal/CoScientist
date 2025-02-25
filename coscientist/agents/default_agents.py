from typing import Dict, List, Any, Optional, Union
import logging
import random
import json

from coscientist.agents.base_agent import BaseAgent
from coscientist.core.hypothesis import Hypothesis
from coscientist.utils.llm_utils import get_llm_response

logger = logging.getLogger(__name__)

class DefaultAgent(BaseAgent):
    """
    Default agent implementation.
    """
    
    def _execute(self, params: Dict[str, Any]) -> Any:
        """
        Execute the default agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
            
        Returns:
            Any: Result of the execution
        """
        logger.warning(f"Using default agent implementation for {self.name}")
        return {"message": "Default agent implementation. Override this method."}


class GenerationAgent(BaseAgent):
    """
    Agent responsible for generating hypotheses.
    """
    
    def __init__(self, 
                 name: str = "generation",
                 description: str = "Generates novel research hypotheses",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a generation agent.
        
        Args:
            name (str, optional): Name of the agent. Defaults to "generation".
            description (str, optional): Description of the agent. 
                Defaults to "Generates novel research hypotheses".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        if tools is None:
            tools = ["arxiv_search", "literature_review"]
        
        super().__init__(name, description, model, tools, config)
    
    def _execute(self, params: Dict[str, Any]) -> List[Hypothesis]:
        """
        Execute the generation agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
                - research_goal (str): The research goal
                - num_hypotheses (int, optional): Number of hypotheses to generate. Defaults to 5.
                - components (List[str], optional): List of components to include. Defaults to None.
            
        Returns:
            List[Hypothesis]: Generated hypotheses
        """
        research_goal = params.get("research_goal", "")
        num_hypotheses = params.get("num_hypotheses", 5)
        components = params.get("components", [])
        
        if not research_goal:
            logger.error("No research goal provided to generation agent")
            return []
        
        logger.info(f"Generating {num_hypotheses} hypotheses for research goal: {research_goal}")
        
        # For a real implementation, this would call an LLM or other generative model
        # This is a dummy implementation for demonstration purposes
        
        prompt = self._create_generation_prompt(research_goal, num_hypotheses, components)
        
        try:
            # Call LLM to generate hypotheses
            response = get_llm_response(
                prompt=prompt,
                model=self.model,
                temperature=0.8,  # Higher temperature for more creativity
                max_tokens=1000
            )
            
            # Parse the response
            hypothesis_list = self._parse_generation_response(response)
            
            # Convert to Hypothesis objects
            hypotheses = []
            for h_data in hypothesis_list:
                hypothesis = Hypothesis(
                    description=h_data["description"],
                    components=h_data.get("components", []),
                    score=0.0,
                    metadata={"source": "generation_agent"}
                )
                hypotheses.append(hypothesis)
            
            logger.info(f"Successfully generated {len(hypotheses)} hypotheses")
            return hypotheses
        
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            
            # Fallback to dummy hypotheses for demonstration
            return self._generate_dummy_hypotheses(research_goal, num_hypotheses, components)
    
    def _create_generation_prompt(self, 
                                 research_goal: str, 
                                 num_hypotheses: int, 
                                 components: List[str]) -> str:
        """
        Create a prompt for hypothesis generation.
        
        Args:
            research_goal (str): The research goal
            num_hypotheses (int): Number of hypotheses to generate
            components (List[str]): List of components to include
            
        Returns:
            str: Generated prompt
        """
        prompt = f"""
        As an AI Co-Scientist, your task is to generate {num_hypotheses} novel and high-impact research hypotheses 
        related to the following research goal:
        
        RESEARCH GOAL: {research_goal}
        
        For each hypothesis:
        1. Provide a clear and concise description
        2. List key components or elements
        3. Consider theoretical soundness and practical feasibility
        
        These hypotheses should be innovative and push the boundaries of current research.
        """
        
        if components:
            prompt += f"\n\nPlease incorporate the following components where applicable: {', '.join(components)}"
        
        prompt += """
        
        Output the hypotheses in the following JSON format:
        [
            {
                "description": "Detailed description of the hypothesis",
                "components": ["component1", "component2", ...]
            },
            ...
        ]
        """
        
        return prompt
    
    def _parse_generation_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the response from the LLM.
        
        Args:
            response (str): Response from the LLM
            
        Returns:
            List[Dict[str, Any]]: Parsed hypotheses
        """
        try:
            # Extract JSON from the response
            json_start = response.find("[")
            json_end = response.rfind("]") + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in response, attempting to parse entire response")
                return json.loads(response)
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from response: {e}")
            
            # Attempt to parse in a more lenient way
            import re
            hypothesis_matches = re.findall(r'Hypothesis \d+: (.*?)(?=Hypothesis \d+:|$)', response, re.DOTALL)
            
            result = []
            for match in hypothesis_matches:
                result.append({"description": match.strip()})
            
            if not result:
                logger.error("Failed to parse response, returning empty list")
                return []
            
            return result
    
    def _generate_dummy_hypotheses(self, 
                                  research_goal: str, 
                                  num_hypotheses: int,
                                  components: List[str]) -> List[Hypothesis]:
        """
        Generate dummy hypotheses for demonstration purposes.
        
        Args:
            research_goal (str): The research goal
            num_hypotheses (int): Number of hypotheses to generate
            components (List[str]): List of components to include
            
        Returns:
            List[Hypothesis]: Generated dummy hypotheses
        """
        hypotheses = []
        
        for i in range(num_hypotheses):
            description = f"Hypothesis {i+1} for {research_goal}"
            
            # Add some components if available
            h_components = []
            if components:
                # Randomly select some components
                num_components = random.randint(1, min(3, len(components)))
                h_components = random.sample(components, num_components)
            
            hypothesis = Hypothesis(
                description=description,
                components=h_components,
                score=0.0,
                metadata={"source": "dummy_generation"}
            )
            
            hypotheses.append(hypothesis)
        
        return hypotheses


class ReflectionAgent(BaseAgent):
    """
    Agent responsible for refining and reflecting on hypotheses.
    """
    
    def __init__(self, 
                 name: str = "reflection",
                 description: str = "Refines and reflects on research hypotheses",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a reflection agent.
        
        Args:
            name (str, optional): Name of the agent. Defaults to "reflection".
            description (str, optional): Description of the agent. 
                Defaults to "Refines and reflects on research hypotheses".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        if tools is None:
            tools = ["symbolic_math", "simulation_review"]
        
        super().__init__(name, description, model, tools, config)
    
    def _execute(self, params: Dict[str, Any]) -> List[Hypothesis]:
        """
        Execute the reflection agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
                - hypotheses (List[Hypothesis]): Hypotheses to refine
            
        Returns:
            List[Hypothesis]: Refined hypotheses
        """
        hypotheses = params.get("hypotheses", [])
        
        if not hypotheses:
            logger.error("No hypotheses provided to reflection agent")
            return []
        
        logger.info(f"Refining {len(hypotheses)} hypotheses")
        
        # For a real implementation, this would call an LLM or other model to refine the hypotheses
        # This is a dummy implementation for demonstration purposes
        
        refined_hypotheses = []
        
        for hypothesis in hypotheses:
            prompt = self._create_reflection_prompt(hypothesis)
            
            try:
                # Call LLM to refine the hypothesis
                response = get_llm_response(
                    prompt=prompt,
                    model=self.model,
                    temperature=0.5,  # Lower temperature for more focused refinement
                    max_tokens=500
                )
                
                # Parse the response
                refined_data = self._parse_reflection_response(response)
                
                # Create a new refined hypothesis
                refined_hypothesis = Hypothesis(
                    description=refined_data.get("description", hypothesis.description),
                    components=refined_data.get("components", hypothesis.components),
                    score=refined_data.get("score", hypothesis.score),
                    metadata=hypothesis.metadata.copy()
                )
                
                # Update metadata
                refined_hypothesis.metadata["parent_id"] = hypothesis.id
                refined_hypothesis.metadata["refinement_source"] = "reflection_agent"
                
                if "feedback" in refined_data:
                    refined_hypothesis.metadata["refinement_feedback"] = refined_data["feedback"]
                
                refined_hypotheses.append(refined_hypothesis)
            
            except Exception as e:
                logger.error(f"Error refining hypothesis: {e}")
                
                # Fallback to simple refinement for demonstration
                refined_hypothesis = self._refine_dummy_hypothesis(hypothesis)
                refined_hypotheses.append(refined_hypothesis)
        
        logger.info(f"Successfully refined {len(refined_hypotheses)} hypotheses")
        return refined_hypotheses
    
    def _create_reflection_prompt(self, hypothesis: Hypothesis) -> str:
        """
        Create a prompt for hypothesis reflection.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to refine
            
        Returns:
            str: Generated prompt
        """
        prompt = f"""
        As an AI Co-Scientist, your task is to refine and reflect on the following research hypothesis:
        
        HYPOTHESIS: {hypothesis.description}
        
        COMPONENTS: {', '.join(hypothesis.components) if hypothesis.components else 'None'}
        
        Please evaluate this hypothesis based on:
        1. Theoretical soundness
        2. Practical feasibility
        3. Originality and novelty
        4. Potential impact
        
        Then, provide a refined version of the hypothesis that addresses any weaknesses or limitations.
        
        Output the refined hypothesis in the following JSON format:
        {{
            "description": "Refined description of the hypothesis",
            "components": ["component1", "component2", ...],
            "score": float_value_between_0_and_1,
            "feedback": "Your feedback and reasoning for the refinements"
        }}
        """
        
        return prompt
    
    def _parse_reflection_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the LLM.
        
        Args:
            response (str): Response from the LLM
            
        Returns:
            Dict[str, Any]: Parsed refined hypothesis
        """
        try:
            # Extract JSON from the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in response, attempting to parse entire response")
                return json.loads(response)
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from response: {e}")
            
            # Attempt to extract information in a more lenient way
            import re
            
            result = {}
            
            # Try to find description
            description_match = re.search(r'description["\s:]+([^"]+)', response)
            if description_match:
                result["description"] = description_match.group(1).strip()
            
            # Try to find score
            score_match = re.search(r'score["\s:]+([0-9.]+)', response)
            if score_match:
                try:
                    result["score"] = float(score_match.group(1))
                except ValueError:
                    pass
            
            # Try to find feedback
            feedback_match = re.search(r'feedback["\s:]+([^"]+)', response)
            if feedback_match:
                result["feedback"] = feedback_match.group(1).strip()
            
            return result
    
    def _refine_dummy_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        """
        Generate a dummy refined hypothesis for demonstration purposes.
        
        Args:
            hypothesis (Hypothesis): The original hypothesis
            
        Returns:
            Hypothesis: Refined hypothesis
        """
        # Simple refinement: Add "Refined: " to the description and a new score
        refined_description = f"Refined: {hypothesis.description}"
        refined_score = min(hypothesis.score + 0.1, 1.0)
        
        refined_hypothesis = Hypothesis(
            description=refined_description,
            components=hypothesis.components.copy(),
            score=refined_score,
            metadata=hypothesis.metadata.copy()
        )
        
        refined_hypothesis.metadata["parent_id"] = hypothesis.id
        refined_hypothesis.metadata["refinement_source"] = "dummy_reflection"
        
        return refined_hypothesis


class EvolutionAgent(BaseAgent):
    """
    Agent responsible for evolving hypotheses.
    """
    
    def __init__(self, 
                 name: str = "evolution",
                 description: str = "Evolves research hypotheses using evolutionary algorithms",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize an evolution agent.
        
        Args:
            name (str, optional): Name of the agent. Defaults to "evolution".
            description (str, optional): Description of the agent. 
                Defaults to "Evolves research hypotheses using evolutionary algorithms".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        if tools is None:
            tools = ["mutation_engine", "crossover_optimizer"]
        
        super().__init__(name, description, model, tools, config)
    
    def _execute(self, params: Dict[str, Any]) -> List[Hypothesis]:
        """
        Execute the evolution agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
                - hypotheses (List[Hypothesis]): Hypotheses to evolve
                - generation (int, optional): Current generation number. Defaults to 0.
                - mutation_rate (float, optional): Mutation rate. Defaults to 0.3.
                - crossover_rate (float, optional): Crossover rate. Defaults to 0.7.
            
        Returns:
            List[Hypothesis]: Evolved hypotheses
        """
        hypotheses = params.get("hypotheses", [])
        generation = params.get("generation", 0)
        mutation_rate = params.get("mutation_rate", 0.3)
        crossover_rate = params.get("crossover_rate", 0.7)
        
        if not hypotheses:
            logger.error("No hypotheses provided to evolution agent")
            return []
        
        logger.info(f"Evolving {len(hypotheses)} hypotheses (generation {generation})")
        
        # For a real implementation, this would use evolutionary algorithms or call an LLM
        # This is a demonstration implementation
        
        evolved_population = hypotheses.copy()
        
        # Mutation
        mutated_hypotheses = self._perform_mutations(evolved_population, mutation_rate)
        evolved_population.extend(mutated_hypotheses)
        
        # Crossover
        if len(hypotheses) >= 2:
            crossed_hypotheses = self._perform_crossovers(evolved_population, crossover_rate)
            evolved_population.extend(crossed_hypotheses)
        
        # Selection (keep the best hypotheses)
        evolved_population = self._perform_selection(evolved_population, len(hypotheses))
        
        logger.info(f"Evolved population size: {len(evolved_population)}")
        return evolved_population
    
    def _perform_mutations(self, 
                          hypotheses: List[Hypothesis],
                          mutation_rate: float) -> List[Hypothesis]:
        """
        Perform mutations on hypotheses.
        
        Args:
            hypotheses (List[Hypothesis]): Hypotheses to mutate
            mutation_rate (float): Mutation rate
            
        Returns:
            List[Hypothesis]: Mutated hypotheses
        """
        mutated_hypotheses = []
        
        for hypothesis in hypotheses:
            if random.random() < mutation_rate:
                # Choose a mutation type
                mutation_types = ["add_component", "remove_component", "modify_description"]
                mutation_type = random.choice(mutation_types)
                
                mutation_data = {}
                
                if mutation_type == "add_component":
                    # Dummy component for demonstration
                    mutation_data = {"component": f"component_{random.randint(1, 100)}"}
                
                elif mutation_type == "remove_component" and hypothesis.components:
                    component_to_remove = random.choice(hypothesis.components)
                    mutation_data = {"component": component_to_remove}
                
                elif mutation_type == "modify_description":
                    # Dummy modification for demonstration
                    modification_options = [
                        "with enhanced performance",
                        "using advanced techniques",
                        "optimized for scalability",
                        "with improved efficiency"
                    ]
                    mutation_data = {"modification": random.choice(modification_options)}
                
                # Create the mutated hypothesis
                mutated_hypothesis = hypothesis.mutate(mutation_type, mutation_data)
                mutated_hypotheses.append(mutated_hypothesis)
        
        return mutated_hypotheses
    
    def _perform_crossovers(self, 
                           hypotheses: List[Hypothesis],
                           crossover_rate: float) -> List[Hypothesis]:
        """
        Perform crossovers on hypotheses.
        
        Args:
            hypotheses (List[Hypothesis]): Hypotheses for crossover
            crossover_rate (float): Crossover rate
            
        Returns:
            List[Hypothesis]: Crossed hypotheses
        """
        crossed_hypotheses = []
        
        if len(hypotheses) < 2:
            return crossed_hypotheses
        
        num_crossovers = int(len(hypotheses) * crossover_rate / 2)
        
        for _ in range(num_crossovers):
            # Select two random hypotheses
            parents = random.sample(hypotheses, 2)
            
            # Create a merged hypothesis
            merged_hypothesis = parents[0].merge(parents[1])
            crossed_hypotheses.append(merged_hypothesis)
        
        return crossed_hypotheses
    
    def _perform_selection(self, 
                          hypotheses: List[Hypothesis],
                          target_size: int) -> List[Hypothesis]:
        """
        Perform selection on hypotheses.
        
        Args:
            hypotheses (List[Hypothesis]): Hypotheses to select from
            target_size (int): Target population size
            
        Returns:
            List[Hypothesis]: Selected hypotheses
        """
        # Sort by score (descending)
        sorted_hypotheses = sorted(hypotheses, key=lambda h: h.score, reverse=True)
        
        # Keep the best hypotheses (elitism)
        selected = sorted_hypotheses[:target_size]
        
        return selected


class RankingAgent(BaseAgent):
    """
    Agent responsible for ranking hypotheses.
    """
    
    def __init__(self, 
                 name: str = "ranking",
                 description: str = "Ranks research hypotheses",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a ranking agent.
        
        Args:
            name (str, optional): Name of the agent. Defaults to "ranking".
            description (str, optional): Description of the agent. 
                Defaults to "Ranks research hypotheses".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        if tools is None:
            tools = ["tournament_review", "deep_verification"]
        
        super().__init__(name, description, model, tools, config)
    
    def _execute(self, params: Dict[str, Any]) -> List[Hypothesis]:
        """
        Execute the ranking agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
                - hypotheses (List[Hypothesis]): Hypotheses to rank
                - criteria (Dict[str, float], optional): Ranking criteria and weights. Defaults to None.
            
        Returns:
            List[Hypothesis]: Ranked hypotheses
        """
        hypotheses = params.get("hypotheses", [])
        criteria = params.get("criteria", {
            "novelty": 0.3,
            "feasibility": 0.2,
            "impact": 0.3,
            "theoretical_soundness": 0.2
        })
        
        if not hypotheses:
            logger.error("No hypotheses provided to ranking agent")
            return []
        
        logger.info(f"Ranking {len(hypotheses)} hypotheses")
        
        # For a real implementation, this would call an LLM or other model to rank the hypotheses
        # This is a dummy implementation for demonstration purposes
        
        ranked_hypotheses = []
        
        for hypothesis in hypotheses:
            prompt = self._create_ranking_prompt(hypothesis, criteria)
            
            try:
                # Call LLM to rank the hypothesis
                response = get_llm_response(
                    prompt=prompt,
                    model=self.model,
                    temperature=0.3,  # Lower temperature for more focused evaluation
                    max_tokens=300
                )
                
                # Parse the response
                ranking_data = self._parse_ranking_response(response)
                
                # Update the hypothesis with the new score and evaluation
                hypothesis.update_score(ranking_data.get("score", hypothesis.score))
                
                for criterion, score in ranking_data.get("criteria_scores", {}).items():
                    hypothesis.update_metadata(f"criterion_{criterion}", score)
                
                if "justification" in ranking_data:
                    hypothesis.update_metadata("ranking_justification", ranking_data["justification"])
                
                ranked_hypotheses.append(hypothesis)
            
            except Exception as e:
                logger.error(f"Error ranking hypothesis: {e}")
                
                # Fallback to simple ranking for demonstration
                self._rank_dummy_hypothesis(hypothesis, criteria)
                ranked_hypotheses.append(hypothesis)
        
        # Sort by score (descending)
        ranked_hypotheses = sorted(ranked_hypotheses, key=lambda h: h.score, reverse=True)
        
        logger.info(f"Successfully ranked {len(ranked_hypotheses)} hypotheses")
        return ranked_hypotheses
    
    def _create_ranking_prompt(self, 
                              hypothesis: Hypothesis,
                              criteria: Dict[str, float]) -> str:
        """
        Create a prompt for hypothesis ranking.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to rank
            criteria (Dict[str, float]): Ranking criteria and weights
            
        Returns:
            str: Generated prompt
        """
        criteria_str = "\n".join([f"{criterion} (weight: {weight})" for criterion, weight in criteria.items()])
        
        prompt = f"""
        As an AI Co-Scientist, your task is to evaluate and rank the following research hypothesis:
        
        HYPOTHESIS: {hypothesis.description}
        
        COMPONENTS: {', '.join(hypothesis.components) if hypothesis.components else 'None'}
        
        Please evaluate this hypothesis based on the following criteria:
        {criteria_str}
        
        For each criterion, provide a score between 0 and 1, where 1 is the highest score.
        Then, compute a weighted average as the overall score.
        
        Output the evaluation in the following JSON format:
        {{
            "score": overall_score,
            "criteria_scores": {{
                "criterion1": score1,
                "criterion2": score2,
                ...
            }},
            "justification": "Your justification for the scores"
        }}
        """
        
        return prompt
    
    def _parse_ranking_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the LLM.
        
        Args:
            response (str): Response from the LLM
            
        Returns:
            Dict[str, Any]: Parsed ranking data
        """
        try:
            # Extract JSON from the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in response, attempting to parse entire response")
                return json.loads(response)
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from response: {e}")
            
            # Attempt to extract information in a more lenient way
            import re
            
            result = {"criteria_scores": {}}
            
            # Try to find overall score
            score_match = re.search(r'score["\s:]+([0-9.]+)', response)
            if score_match:
                try:
                    result["score"] = float(score_match.group(1))
                except ValueError:
                    pass
            
            # Try to find justification
            justification_match = re.search(r'justification["\s:]+([^"]+)', response)
            if justification_match:
                result["justification"] = justification_match.group(1).strip()
            
            return result
    
    def _rank_dummy_hypothesis(self, 
                              hypothesis: Hypothesis,
                              criteria: Dict[str, float]):
        """
        Rank a hypothesis with dummy scores for demonstration purposes.
        
        Args:
            hypothesis (Hypothesis): The hypothesis to rank
            criteria (Dict[str, float]): Ranking criteria and weights
        """
        # Generate random scores for each criterion
        criteria_scores = {}
        for criterion in criteria:
            criteria_scores[criterion] = random.uniform(0.5, 1.0)
        
        # Compute weighted average
        score = sum(criteria_scores[c] * criteria[c] for c in criteria)
        
        # Update the hypothesis
        hypothesis.update_score(score, "Dummy ranking")
        
        for criterion, score in criteria_scores.items():
            hypothesis.update_metadata(f"criterion_{criterion}", score)


class MetaReviewAgent(BaseAgent):
    """
    Agent responsible for generating a meta-review of ranked hypotheses.
    """
    
    def __init__(self, 
                 name: str = "meta_review",
                 description: str = "Generates a meta-review of ranked hypotheses",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a meta-review agent.
        
        Args:
            name (str, optional): Name of the agent. Defaults to "meta_review".
            description (str, optional): Description of the agent. 
                Defaults to "Generates a meta-review of ranked hypotheses".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        if tools is None:
            tools = ["research_overview_formulation"]
        
        super().__init__(name, description, model, tools, config)
    
    def _execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the meta-review agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
                - hypotheses (List[Hypothesis]): Ranked hypotheses
                - research_goal (str): The original research goal
                - num_top (int, optional): Number of top hypotheses to include. Defaults to 3.
            
        Returns:
            Dict[str, Any]: Meta-review of the hypotheses
        """
        hypotheses = params.get("hypotheses", [])
        research_goal = params.get("research_goal", "")
        num_top = params.get("num_top", 3)
        
        if not hypotheses:
            logger.error("No hypotheses provided to meta-review agent")
            return {"error": "No hypotheses provided"}
        
        if not research_goal:
            logger.warning("No research goal provided to meta-review agent")
        
        # Sort by score (descending)
        ranked_hypotheses = sorted(hypotheses, key=lambda h: h.score, reverse=True)
        
        # Get top hypotheses
        top_hypotheses = ranked_hypotheses[:num_top]
        
        logger.info(f"Generating meta-review of top {len(top_hypotheses)} hypotheses")
        
        # For a real implementation, this would call an LLM or other model
        # This is a dummy implementation for demonstration purposes
        
        prompt = self._create_meta_review_prompt(research_goal, top_hypotheses)
        
        try:
            # Call LLM to generate the meta-review
            response = get_llm_response(
                prompt=prompt,
                model=self.model,
                temperature=0.5,
                max_tokens=1000
            )
            
            # Parse the response
            meta_review = self._parse_meta_review_response(response)
            
            logger.info("Successfully generated meta-review")
            return meta_review
        
        except Exception as e:
            logger.error(f"Error generating meta-review: {e}")
            
            # Fallback to simple meta-review for demonstration
            return self._generate_dummy_meta_review(research_goal, top_hypotheses)
    
    def _create_meta_review_prompt(self, 
                                  research_goal: str,
                                  top_hypotheses: List[Hypothesis]) -> str:
        """
        Create a prompt for meta-review generation.
        
        Args:
            research_goal (str): The research goal
            top_hypotheses (List[Hypothesis]): Top ranked hypotheses
            
        Returns:
            str: Generated prompt
        """
        hypotheses_str = "\n\n".join([
            f"HYPOTHESIS {i+1} (Score: {h.score:.2f}):\n{h.description}\n"
            f"Components: {', '.join(h.components) if h.components else 'None'}"
            for i, h in enumerate(top_hypotheses)
        ])
        
        prompt = f"""
        As an AI Co-Scientist, your task is to generate a comprehensive meta-review of the top ranked hypotheses
        for the following research goal:
        
        RESEARCH GOAL: {research_goal}
        
        TOP HYPOTHESES:
        {hypotheses_str}
        
        Please provide:
        1. A summary of the top hypotheses and their key strengths
        2. Common themes or patterns across the hypotheses
        3. Potential next steps for investigation
        4. Suggestions for how to combine or extend these hypotheses
        
        Output the meta-review in the following JSON format:
        {{
            "summary": "Summary of the top hypotheses",
            "common_themes": ["theme1", "theme2", ...],
            "next_steps": ["step1", "step2", ...],
            "suggestions": ["suggestion1", "suggestion2", ...],
            "combined_hypothesis": "A proposed combined hypothesis that leverages the strengths of the top hypotheses"
        }}
        """
        
        return prompt
    
    def _parse_meta_review_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the LLM.
        
        Args:
            response (str): Response from the LLM
            
        Returns:
            Dict[str, Any]: Parsed meta-review
        """
        try:
            # Extract JSON from the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in response, attempting to parse entire response")
                return {"summary": response}
            
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from response: {e}")
            
            # Just return the raw response as the summary
            return {"summary": response}
    
    def _generate_dummy_meta_review(self, 
                                   research_goal: str,
                                   top_hypotheses: List[Hypothesis]) -> Dict[str, Any]:
        """
        Generate a dummy meta-review for demonstration purposes.
        
        Args:
            research_goal (str): The research goal
            top_hypotheses (List[Hypothesis]): Top ranked hypotheses
            
        Returns:
            Dict[str, Any]: Generated dummy meta-review
        """
        return {
            "summary": f"Meta-review of {len(top_hypotheses)} hypotheses for {research_goal}",
            "common_themes": ["Theme 1", "Theme 2"],
            "next_steps": ["Step 1", "Step 2"],
            "suggestions": ["Suggestion 1", "Suggestion 2"],
            "combined_hypothesis": "Combined hypothesis based on the top-ranked ideas."
        } 
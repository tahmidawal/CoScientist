"""
Specialized agents for the CoScientist system.

This module contains specialized agents for different aspects of the research process:
1. Manager Agent: Understands and elaborates on research problems
2. Literature Agent: Finds related research and creates vector stores
3. Research Agent: Generates hypotheses and finds supporting documents
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
import numpy as np

from coscientist.agents.base_agent import BaseAgent
from coscientist.core.hypothesis import Hypothesis
from coscientist.utils.llm_utils import get_llm_response
from coscientist.tools.academic_search import search_academic_sources

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed. Vector storage functionality will be limited. Install with: pip install faiss-cpu or faiss-gpu")

class ManagerAgent(BaseAgent):
    """
    Agent responsible for understanding and elaborating on research problems.
    """
    
    def __init__(self, 
                 name: str = "manager",
                 description: str = "Understands and elaborates on research problems",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a manager agent.
        
        Args:
            name (str, optional): Name of the agent. Defaults to "manager".
            description (str, optional): Description of the agent. 
                Defaults to "Understands and elaborates on research problems".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        if tools is None:
            tools = ["problem_analysis", "domain_knowledge"]
        
        super().__init__(name, description, model, tools, config)
    
    def _execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the manager agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
                - research_goal (str): The research goal to elaborate on
                
        Returns:
            Dict[str, Any]: Elaborated research problem
        """
        research_goal = params.get("research_goal", "")
        
        if not research_goal:
            logger.error("No research goal provided to manager agent")
            return {"error": "No research goal provided"}
        
        logger.info(f"Elaborating on research goal: {research_goal}")
        
        prompt = self._create_elaboration_prompt(research_goal)
        
        try:
            # Call LLM to elaborate on the research goal
            response = get_llm_response(
                prompt=prompt,
                model=self.model,
                temperature=0.5,
                max_tokens=1500
            )
            
            # Parse the response
            elaboration = self._parse_elaboration_response(response)
            
            logger.info("Successfully elaborated on research goal")
            return elaboration
        
        except Exception as e:
            logger.error(f"Error elaborating on research goal: {e}")
            return {"error": str(e)}
    
    def _create_elaboration_prompt(self, research_goal: str) -> str:
        """
        Create a prompt for research goal elaboration.
        
        Args:
            research_goal (str): The research goal
            
        Returns:
            str: Generated prompt
        """
        prompt = f"""
        As an AI Co-Scientist, your task is to elaborate on the following research goal:
        
        RESEARCH GOAL: {research_goal}
        
        Please provide a comprehensive elaboration including:
        1. Problem statement and background
        2. Key challenges and obstacles
        3. Current state of the art
        4. Potential research directions
        5. Relevant domains and subfields
        6. Potential applications and impact
        
        Format your response as a JSON object with the following fields:
        - problem_statement: detailed problem statement
        - background: background information
        - challenges: list of key challenges
        - state_of_art: current state of the art
        - research_directions: list of potential research directions
        - domains: list of relevant domains and subfields
        - applications: list of potential applications
        - impact: potential impact of successful research
        """
        
        return prompt
    
    def _parse_elaboration_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the response from the LLM.
        
        Args:
            response (str): Response from the LLM
            
        Returns:
            Dict[str, Any]: Parsed elaboration
        """
        try:
            # Extract JSON from the response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                # If no JSON found, return the full text
                return {"elaboration": response}
                
        except json.JSONDecodeError:
            # If JSON parsing fails, return the full text
            return {"elaboration": response}


class LiteratureAgent(BaseAgent):
    """
    Agent responsible for finding related research and creating vector stores.
    """
    
    def __init__(self, 
                 name: str = "literature",
                 description: str = "Finds related research and creates vector stores",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a literature agent.
        
        Args:
            name (str, optional): Name of the agent. Defaults to "literature".
            description (str, optional): Description of the agent. 
                Defaults to "Finds related research and creates vector stores".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        if tools is None:
            tools = ["arxiv_search", "semantic_scholar", "vector_store"]
        
        super().__init__(name, description, model, tools, config)
        
        # Initialize vector store
        self.vector_store = None
        self.paper_index = {}
        self.initialize_vector_store()
    
    def initialize_vector_store(self):
        """
        Initialize the vector store for paper embeddings.
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available. Vector store will not be created.")
            return
        
        try:
            # Initialize a FAISS index for paper embeddings
            # Using a simple L2 distance index for 768-dimensional embeddings (typical for many models)
            dimension = 768
            self.vector_store = faiss.IndexFlatL2(dimension)
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
    
    def _execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the literature agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
                - research_goal (str): The research goal
                - elaboration (Dict[str, Any], optional): Elaborated research problem
                - max_papers (int, optional): Maximum number of papers to retrieve. Defaults to 20.
                
        Returns:
            Dict[str, Any]: Retrieved papers and vector store information
        """
        research_goal = params.get("research_goal", "")
        elaboration = params.get("elaboration", {})
        max_papers = params.get("max_papers", 20)
        
        if not research_goal:
            logger.error("No research goal provided to literature agent")
            return {"error": "No research goal provided"}
        
        logger.info(f"Finding related research for: {research_goal}")
        
        # Generate search queries based on research goal and elaboration
        search_queries = self._generate_search_queries(research_goal, elaboration)
        
        # Search for papers
        all_papers = []
        for query in search_queries:
            logger.info(f"Searching for papers with query: {query}")
            search_results = search_academic_sources(query, max_results=max_papers // len(search_queries))
            if "papers" in search_results:
                all_papers.extend(search_results["papers"])
        
        # Deduplicate papers
        unique_papers = self._deduplicate_papers(all_papers)
        
        # Create embeddings and add to vector store
        if FAISS_AVAILABLE and self.vector_store is not None:
            self._add_papers_to_vector_store(unique_papers)
        
        logger.info(f"Found {len(unique_papers)} unique papers")
        
        return {
            "papers": unique_papers,
            "vector_store_available": FAISS_AVAILABLE and self.vector_store is not None,
            "num_papers_indexed": len(self.paper_index) if self.paper_index else 0
        }
    
    def _generate_search_queries(self, research_goal: str, elaboration: Dict[str, Any]) -> List[str]:
        """
        Generate search queries based on research goal and elaboration.
        
        Args:
            research_goal (str): The research goal
            elaboration (Dict[str, Any]): Elaborated research problem
            
        Returns:
            List[str]: List of search queries
        """
        queries = [research_goal]
        
        # Add queries based on elaboration
        if "research_directions" in elaboration and isinstance(elaboration["research_directions"], list):
            for direction in elaboration["research_directions"][:3]:  # Limit to top 3 directions
                queries.append(f"{research_goal} {direction}")
        
        if "challenges" in elaboration and isinstance(elaboration["challenges"], list):
            for challenge in elaboration["challenges"][:2]:  # Limit to top 2 challenges
                queries.append(f"{research_goal} {challenge}")
        
        return queries
    
    def _deduplicate_papers(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate papers based on title.
        
        Args:
            papers (List[Dict[str, Any]]): List of papers
            
        Returns:
            List[Dict[str, Any]]: Deduplicated list of papers
        """
        unique_papers = []
        seen_titles = set()
        
        for paper in papers:
            title = paper.get("title", "").lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_papers.append(paper)
        
        return unique_papers
    
    def _add_papers_to_vector_store(self, papers: List[Dict[str, Any]]):
        """
        Add papers to the vector store.
        
        Args:
            papers (List[Dict[str, Any]]): List of papers to add
        """
        if not FAISS_AVAILABLE or self.vector_store is None:
            logger.warning("Vector store not available. Papers will not be indexed.")
            return
        
        try:
            # For each paper, create an embedding and add to the vector store
            for i, paper in enumerate(papers):
                # Create a text representation of the paper
                paper_text = f"{paper.get('title', '')} {paper.get('summary', '')}"
                
                # In a real implementation, we would use a proper embedding model here
                # For demonstration, we'll use random embeddings
                embedding = np.random.randn(768).astype('float32')
                embedding = embedding / np.linalg.norm(embedding)  # Normalize
                
                # Add to FAISS index
                self.vector_store.add(np.array([embedding]))
                
                # Store paper information
                self.paper_index[i] = paper
            
            logger.info(f"Added {len(papers)} papers to vector store")
        
        except Exception as e:
            logger.error(f"Error adding papers to vector store: {e}")
    
    def search_similar_papers(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for papers similar to the query text.
        
        Args:
            query_text (str): Query text
            top_k (int, optional): Number of top results to return. Defaults to 5.
            
        Returns:
            List[Dict[str, Any]]: List of similar papers
        """
        if not FAISS_AVAILABLE or self.vector_store is None or not self.paper_index:
            logger.warning("Vector store not available or empty. Cannot search for similar papers.")
            return []
        
        try:
            # In a real implementation, we would use a proper embedding model here
            # For demonstration, we'll use a random embedding
            query_embedding = np.random.randn(768).astype('float32')
            query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalize
            
            # Search in FAISS index
            distances, indices = self.vector_store.search(np.array([query_embedding]), top_k)
            
            # Get papers
            similar_papers = []
            for idx in indices[0]:
                if idx in self.paper_index:
                    similar_papers.append(self.paper_index[idx])
            
            return similar_papers
        
        except Exception as e:
            logger.error(f"Error searching for similar papers: {e}")
            return []


class ResearchAgent(BaseAgent):
    """
    Agent responsible for generating hypotheses and finding supporting documents.
    """
    
    def __init__(self, 
                 name: str = "research",
                 description: str = "Generates hypotheses and finds supporting documents",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a research agent.
        
        Args:
            name (str, optional): Name of the agent. Defaults to "research".
            description (str, optional): Description of the agent. 
                Defaults to "Generates hypotheses and finds supporting documents".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        if tools is None:
            tools = ["hypothesis_generation", "document_search"]
        
        super().__init__(name, description, model, tools, config)
    
    def _execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the research agent logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
                - research_goal (str): The research goal
                - elaboration (Dict[str, Any], optional): Elaborated research problem
                - papers (List[Dict[str, Any]], optional): Retrieved papers
                - literature_agent (LiteratureAgent, optional): Literature agent for vector search
                - num_hypotheses (int, optional): Number of hypotheses to generate. Defaults to 5.
                
        Returns:
            Dict[str, Any]: Generated hypotheses with supporting documents
        """
        research_goal = params.get("research_goal", "")
        elaboration = params.get("elaboration", {})
        papers = params.get("papers", [])
        literature_agent = params.get("literature_agent")
        num_hypotheses = params.get("num_hypotheses", 5)
        
        if not research_goal:
            logger.error("No research goal provided to research agent")
            return {"error": "No research goal provided"}
        
        logger.info(f"Generating hypotheses for: {research_goal}")
        
        # Generate hypotheses
        hypotheses = self._generate_hypotheses(research_goal, elaboration, papers, num_hypotheses)
        
        # Find supporting documents for each hypothesis
        hypotheses_with_documents = []
        for hypothesis in hypotheses:
            supporting_docs = self._find_supporting_documents(hypothesis, papers, literature_agent)
            hypotheses_with_documents.append({
                "hypothesis": hypothesis,
                "supporting_documents": supporting_docs
            })
        
        logger.info(f"Generated {len(hypotheses)} hypotheses with supporting documents")
        
        return {
            "hypotheses": hypotheses_with_documents
        }
    
    def _generate_hypotheses(self, 
                           research_goal: str, 
                           elaboration: Dict[str, Any],
                           papers: List[Dict[str, Any]],
                           num_hypotheses: int) -> List[Hypothesis]:
        """
        Generate research hypotheses.
        
        Args:
            research_goal (str): The research goal
            elaboration (Dict[str, Any]): Elaborated research problem
            papers (List[Dict[str, Any]]): Retrieved papers
            num_hypotheses (int): Number of hypotheses to generate
            
        Returns:
            List[Hypothesis]: Generated hypotheses
        """
        # Create a prompt that includes information from papers
        paper_summaries = ""
        if papers:
            # Include summaries of top papers
            top_papers = papers[:min(5, len(papers))]
            paper_summaries = "\n\n".join([
                f"Paper: {paper.get('title', '')}\nAuthors: {', '.join(paper.get('authors', []))}\nSummary: {paper.get('summary', '')}"
                for paper in top_papers
            ])
        
        prompt = f"""
        As an AI Co-Scientist, your task is to generate {num_hypotheses} innovative and scientifically sound research hypotheses 
        for the following research goal:
        
        RESEARCH GOAL: {research_goal}
        
        ELABORATION:
        {json.dumps(elaboration, indent=2) if elaboration else "No elaboration provided."}
        
        RELEVANT LITERATURE:
        {paper_summaries if paper_summaries else "No literature provided."}
        
        For each hypothesis:
        1. Provide a clear and detailed description
        2. List key components or elements involved
        3. Explain the potential methodologies to test it
        4. Describe the expected impact if validated
        
        These hypotheses should be innovative, pushing the boundaries of current research, while being grounded in the existing literature.
        
        Format your response as a JSON array of objects, where each object represents a hypothesis with the following fields:
        - description: detailed description of the hypothesis
        - components: array of key components
        - methodologies: array of potential methodologies
        - impact: expected impact if validated
        """
        
        try:
            # Call LLM to generate hypotheses
            response = get_llm_response(
                prompt=prompt,
                model=self.model,
                temperature=0.8,  # Higher temperature for more creativity
                max_tokens=2000
            )
            
            # Parse the response
            hypothesis_list = self._parse_hypotheses_response(response)
            
            # Convert to Hypothesis objects
            hypotheses = []
            for i, h_data in enumerate(hypothesis_list):
                hypothesis = Hypothesis(
                    description=h_data.get("description", ""),
                    components=h_data.get("components", []),
                    score=0.0,
                    metadata={
                        "methodologies": h_data.get("methodologies", []),
                        "impact": h_data.get("impact", ""),
                        "source": "research_agent"
                    }
                )
                hypotheses.append(hypothesis)
            
            logger.info(f"Successfully generated {len(hypotheses)} hypotheses")
            return hypotheses
        
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
            return []
    
    def _parse_hypotheses_response(self, response: str) -> List[Dict[str, Any]]:
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
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                logger.warning("No JSON array found in response, attempting to parse entire response")
                # Try to parse the entire response as JSON
                return json.loads(response)
        
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
    
    def _find_supporting_documents(self, 
                                 hypothesis: Hypothesis, 
                                 papers: List[Dict[str, Any]],
                                 literature_agent: Optional[LiteratureAgent]) -> List[Dict[str, Any]]:
        """
        Find supporting documents for a hypothesis.
        
        Args:
            hypothesis (Hypothesis): The hypothesis
            papers (List[Dict[str, Any]]): Retrieved papers
            literature_agent (LiteratureAgent, optional): Literature agent for vector search
            
        Returns:
            List[Dict[str, Any]]: Supporting documents
        """
        # If literature agent is available, use vector search
        if literature_agent is not None:
            supporting_docs = literature_agent.search_similar_papers(hypothesis.description, top_k=5)
            if supporting_docs:
                return supporting_docs
        
        # Otherwise, use keyword matching
        supporting_docs = []
        
        # Extract keywords from hypothesis
        keywords = hypothesis.description.lower().split()
        keywords = [word for word in keywords if len(word) > 3]  # Filter out short words
        
        # Score papers based on keyword matches
        paper_scores = []
        for paper in papers:
            title = paper.get("title", "").lower()
            summary = paper.get("summary", "").lower()
            
            score = 0
            for keyword in keywords:
                if keyword in title:
                    score += 2  # Higher weight for title matches
                if keyword in summary:
                    score += 1  # Lower weight for summary matches
            
            paper_scores.append((score, paper))
        
        # Sort by score (descending) and take top 5
        paper_scores.sort(reverse=True, key=lambda x: x[0])
        supporting_docs = [paper for score, paper in paper_scores[:5] if score > 0]
        
        return supporting_docs 
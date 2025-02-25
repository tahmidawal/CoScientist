from typing import Dict, List, Any, Optional
import logging

from coscientist.agents.agent_manager import AgentManager
from coscientist.core.workflow import ResearchWorkflow
from coscientist.core.hypothesis import Hypothesis

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CoScientist:
    """
    Main class for the AI Co-Scientist system.
    This class orchestrates the entire research process from idea generation to final output.
    """

    def __init__(self, 
                 research_goal: str,
                 config_path: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize the AI Co-Scientist system.
        
        Args:
            research_goal (str): The description of the research problem to be explored
            config_path (str, optional): Path to configuration file. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to True.
        """
        self.research_goal = research_goal
        self.verbose = verbose
        self.config = self._load_config(config_path) if config_path else {}
        
        # Initialize agent manager
        self.agent_manager = AgentManager(self.config.get("agents", {}))
        
        # Initialize internal state
        self.hypotheses = []
        self.refined_hypotheses = []
        self.ranked_hypotheses = []
        self.research_summary = None
        
        if self.verbose:
            logger.info(f"AI Co-Scientist initialized with research goal: {research_goal}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        import json
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return {}
    
    def generate_hypotheses(self, num_hypotheses: int = 5) -> List[Hypothesis]:
        """
        Generate initial hypotheses based on the research goal.
        
        Args:
            num_hypotheses (int, optional): Number of hypotheses to generate. Defaults to 5.
            
        Returns:
            List[Hypothesis]: Generated hypotheses
        """
        if self.verbose:
            logger.info(f"Generating {num_hypotheses} hypotheses...")
        
        self.hypotheses = self.agent_manager.run_agent(
            "generation", 
            {"research_goal": self.research_goal, "num_hypotheses": num_hypotheses}
        )
        
        return self.hypotheses
    
    def refine_hypotheses(self) -> List[Hypothesis]:
        """
        Refine the generated hypotheses.
        
        Returns:
            List[Hypothesis]: Refined hypotheses
        """
        if not self.hypotheses:
            logger.warning("No hypotheses to refine. Generate hypotheses first.")
            return []
        
        if self.verbose:
            logger.info(f"Refining {len(self.hypotheses)} hypotheses...")
        
        self.refined_hypotheses = self.agent_manager.run_agent(
            "reflection", 
            {"hypotheses": self.hypotheses}
        )
        
        return self.refined_hypotheses
    
    def evolve_hypotheses(self, generations: int = 3) -> List[Hypothesis]:
        """
        Evolve the refined hypotheses using evolutionary algorithms.
        
        Args:
            generations (int, optional): Number of evolution generations. Defaults to 3.
            
        Returns:
            List[Hypothesis]: Evolved hypotheses
        """
        hypotheses_to_evolve = self.refined_hypotheses or self.hypotheses
        
        if not hypotheses_to_evolve:
            logger.warning("No hypotheses to evolve. Generate or refine hypotheses first.")
            return []
        
        if self.verbose:
            logger.info(f"Evolving hypotheses over {generations} generations...")
        
        evolved_hypotheses = hypotheses_to_evolve
        for i in range(generations):
            if self.verbose:
                logger.info(f"Evolution generation {i+1}/{generations}...")
            
            evolved_hypotheses = self.agent_manager.run_agent(
                "evolution", 
                {"hypotheses": evolved_hypotheses, "generation": i}
            )
        
        self.evolved_hypotheses = evolved_hypotheses
        return evolved_hypotheses
    
    def rank_hypotheses(self) -> List[Hypothesis]:
        """
        Rank the evolved hypotheses.
        
        Returns:
            List[Hypothesis]: Ranked hypotheses
        """
        hypotheses_to_rank = (self.evolved_hypotheses or 
                             self.refined_hypotheses or 
                             self.hypotheses)
        
        if not hypotheses_to_rank:
            logger.warning("No hypotheses to rank. Generate, refine, or evolve hypotheses first.")
            return []
        
        if self.verbose:
            logger.info(f"Ranking {len(hypotheses_to_rank)} hypotheses...")
        
        self.ranked_hypotheses = self.agent_manager.run_agent(
            "ranking", 
            {"hypotheses": hypotheses_to_rank}
        )
        
        return self.ranked_hypotheses
    
    def generate_research_summary(self) -> Dict[str, Any]:
        """
        Generate a research summary from the ranked hypotheses.
        
        Returns:
            Dict[str, Any]: Research summary
        """
        if not self.ranked_hypotheses:
            logger.warning("No ranked hypotheses. Rank hypotheses first.")
            return {}
        
        if self.verbose:
            logger.info("Generating research summary...")
        
        self.research_summary = self.agent_manager.run_agent(
            "meta_review", 
            {"hypotheses": self.ranked_hypotheses, "research_goal": self.research_goal}
        )
        
        return self.research_summary
    
    def run_full_workflow(self, num_hypotheses: int = 5, generations: int = 3) -> Dict[str, Any]:
        """
        Run the full research workflow.
        
        Args:
            num_hypotheses (int, optional): Number of hypotheses to generate. Defaults to 5.
            generations (int, optional): Number of evolution generations. Defaults to 3.
            
        Returns:
            Dict[str, Any]: Research summary
        """
        if self.verbose:
            logger.info("Starting full AI Co-Scientist workflow...")
        
        # Step 1: Generate hypotheses
        self.generate_hypotheses(num_hypotheses)
        
        # Step 2: Refine hypotheses
        self.refine_hypotheses()
        
        # Step 3: Evolve hypotheses
        self.evolve_hypotheses(generations)
        
        # Step 4: Rank hypotheses
        self.rank_hypotheses()
        
        # Step 5: Generate research summary
        final_output = self.generate_research_summary()
        
        if self.verbose:
            logger.info("AI Co-Scientist workflow completed.")
        
        return final_output
    
    def run_custom_workflow(self, workflow: ResearchWorkflow) -> Dict[str, Any]:
        """
        Run a custom research workflow.
        
        Args:
            workflow (ResearchWorkflow): Custom workflow to run
            
        Returns:
            Dict[str, Any]: Research output
        """
        if self.verbose:
            logger.info(f"Running custom workflow: {workflow.name}")
        
        return workflow.execute(self) 
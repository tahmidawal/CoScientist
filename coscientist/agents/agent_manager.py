from typing import Dict, List, Any, Optional, Union, Callable
import logging
import importlib
import json

from coscientist.agents.base_agent import BaseAgent
from coscientist.core.hypothesis import Hypothesis

logger = logging.getLogger(__name__)

class AgentManager:
    """
    Class for managing and coordinating multiple agents.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the agent manager.
        
        Args:
            config (Dict[str, Any], optional): Configuration for agents. Defaults to None.
        """
        self.config = config or {}
        self.agents = {}
        self._initialize_agents()
    
    def _initialize_agents(self):
        """
        Initialize agents from configuration.
        """
        # Initialize default agents if not in config
        if not self.config:
            self._initialize_default_agents()
            return
        
        # Initialize agents from config
        for agent_name, agent_config in self.config.items():
            try:
                agent_class_path = agent_config.get("class", "coscientist.agents.default_agents.DefaultAgent")
                agent_class = self._load_class(agent_class_path)
                
                agent_instance = agent_class(
                    name=agent_name,
                    **agent_config.get("params", {})
                )
                
                self.agents[agent_name] = agent_instance
                logger.info(f"Initialized agent: {agent_name}")
            
            except Exception as e:
                logger.error(f"Error initializing agent {agent_name}: {e}")
    
    def _initialize_default_agents(self):
        """
        Initialize default agents if no configuration is provided.
        """
        from coscientist.agents.default_agents import (
            GenerationAgent,
            ReflectionAgent,
            EvolutionAgent,
            RankingAgent,
            MetaReviewAgent
        )
        
        self.agents = {
            "generation": GenerationAgent(name="generation"),
            "reflection": ReflectionAgent(name="reflection"),
            "evolution": EvolutionAgent(name="evolution"),
            "ranking": RankingAgent(name="ranking"),
            "meta_review": MetaReviewAgent(name="meta_review")
        }
        
        logger.info("Initialized default agents")
    
    def _load_class(self, class_path: str) -> type:
        """
        Load a class from a module path.
        
        Args:
            class_path (str): Path to the class
            
        Returns:
            type: Class type
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """
        Get an agent by name.
        
        Args:
            agent_name (str): Name of the agent
            
        Returns:
            Optional[BaseAgent]: The agent, or None if not found
        """
        return self.agents.get(agent_name)
    
    def run_agent(self, 
                 agent_name: str, 
                 params: Dict[str, Any]) -> Any:
        """
        Run an agent.
        
        Args:
            agent_name (str): Name of the agent to run
            params (Dict[str, Any]): Parameters for the agent
            
        Returns:
            Any: Result of the agent run
        """
        agent = self.get_agent(agent_name)
        
        if not agent:
            logger.error(f"Agent {agent_name} not found")
            return None
        
        logger.info(f"Running agent: {agent_name}")
        return agent.run(params)
    
    def run_pipeline(self, 
                    pipeline: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run a pipeline of agents.
        
        Args:
            pipeline (List[Dict[str, Any]]): List of pipeline steps
            
        Returns:
            Dict[str, Any]: Results of the pipeline
        """
        results = {}
        
        for i, step in enumerate(pipeline):
            agent_name = step["agent"]
            params = step.get("params", {})
            
            # Gather data from previous steps if specified
            if "gather_from" in step:
                for gather_key, result_key in step["gather_from"].items():
                    if gather_key in results:
                        params[result_key] = results[gather_key]
            
            logger.info(f"Pipeline step {i+1}/{len(pipeline)}: {agent_name}")
            
            result = self.run_agent(agent_name, params)
            results[agent_name] = result
            
            # Store with alias if specified
            if "alias" in step:
                results[step["alias"]] = result
        
        return results
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the agent manager.
        
        Returns:
            Dict[str, Any]: Serialized agent manager
        """
        return {
            "config": self.config,
            "agents": {name: agent.serialize() for name, agent in self.agents.items()}
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'AgentManager':
        """
        Deserialize an agent manager.
        
        Args:
            data (Dict[str, Any]): Serialized agent manager
            
        Returns:
            AgentManager: Deserialized agent manager
        """
        manager = cls(config=data.get("config", {}))
        
        # Restore agents
        for name, agent_data in data.get("agents", {}).items():
            agent_class_path = agent_data.get("class", "coscientist.agents.default_agents.DefaultAgent")
            agent_class = manager._load_class(agent_class_path)
            
            agent_instance = agent_class.deserialize(agent_data)
            manager.agents[name] = agent_instance
        
        return manager 
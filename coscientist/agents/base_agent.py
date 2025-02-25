from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    Base class for all agents in the system.
    """
    
    def __init__(self, 
                 name: str,
                 description: str = "",
                 model: str = "default",
                 tools: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize a base agent.
        
        Args:
            name (str): Name of the agent
            description (str, optional): Description of the agent. Defaults to "".
            model (str, optional): Model to use for the agent. Defaults to "default".
            tools (List[str], optional): List of tools available to the agent. Defaults to None.
            config (Dict[str, Any], optional): Additional configuration. Defaults to None.
        """
        self.name = name
        self.description = description
        self.model = model
        self.tools = tools or []
        self.config = config or {}
        self.history = []
    
    def _record_run(self, 
                   params: Dict[str, Any], 
                   result: Any, 
                   error: Optional[Exception] = None,
                   duration: float = 0.0):
        """
        Record a run in the agent's history.
        
        Args:
            params (Dict[str, Any]): Parameters for the run
            result (Any): Result of the run
            error (Optional[Exception], optional): Error that occurred. Defaults to None.
            duration (float, optional): Duration of the run in seconds. Defaults to 0.0.
        """
        run_record = {
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "duration": duration,
            "success": error is None
        }
        
        if error:
            run_record["error"] = str(error)
        else:
            # Only store result if no error occurred
            # We might need to handle serialization better here
            try:
                # Try to serialize the result to ensure it's storable
                json.dumps(result)
                run_record["result"] = result
            except (TypeError, OverflowError):
                run_record["result"] = str(result)
        
        self.history.append(run_record)
    
    @abstractmethod
    def _execute(self, params: Dict[str, Any]) -> Any:
        """
        Execute the agent's logic.
        
        Args:
            params (Dict[str, Any]): Parameters for the execution
            
        Returns:
            Any: Result of the execution
        """
        pass
    
    def run(self, params: Dict[str, Any]) -> Any:
        """
        Run the agent.
        
        Args:
            params (Dict[str, Any]): Parameters for the run
            
        Returns:
            Any: Result of the run
        """
        start_time = time.time()
        result = None
        error = None
        
        try:
            logger.info(f"Running agent {self.name}")
            result = self._execute(params)
            return result
        
        except Exception as e:
            logger.error(f"Error running agent {self.name}: {e}")
            error = e
            raise
        
        finally:
            duration = time.time() - start_time
            logger.info(f"Agent {self.name} run completed in {duration:.2f}s")
            self._record_run(params, result, error, duration)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get the agent's run history.
        
        Returns:
            List[Dict[str, Any]]: The agent's history
        """
        return self.history
    
    def clear_history(self):
        """
        Clear the agent's run history.
        """
        self.history = []
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the agent.
        
        Returns:
            Dict[str, Any]: Serialized agent
        """
        return {
            "class": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "tools": self.tools,
            "config": self.config,
            "history": self.history
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'BaseAgent':
        """
        Deserialize an agent.
        
        Args:
            data (Dict[str, Any]): Serialized agent
            
        Returns:
            BaseAgent: Deserialized agent
        """
        agent = cls(
            name=data["name"],
            description=data.get("description", ""),
            model=data.get("model", "default"),
            tools=data.get("tools", []),
            config=data.get("config", {})
        )
        
        agent.history = data.get("history", [])
        
        return agent 
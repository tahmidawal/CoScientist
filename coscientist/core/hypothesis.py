from typing import Dict, List, Any, Optional
from uuid import uuid4
from datetime import datetime

class Hypothesis:
    """
    Class representing a research hypothesis.
    """
    
    def __init__(self, 
                 description: str,
                 components: Optional[List[str]] = None,
                 score: float = 0.0,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a research hypothesis.
        
        Args:
            description (str): Main description of the hypothesis
            components (List[str], optional): List of components/parts of the hypothesis. Defaults to None.
            score (float, optional): Score of the hypothesis. Defaults to 0.0.
            metadata (Dict[str, Any], optional): Additional metadata. Defaults to None.
        """
        self.id = str(uuid4())
        self.description = description
        self.components = components or []
        self.score = score
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.history = []
        
        # Record the initial state
        self._record_history("created")
    
    def _record_history(self, action: str, details: Optional[Dict[str, Any]] = None):
        """
        Record an action in the history.
        
        Args:
            action (str): The action performed
            details (Dict[str, Any], optional): Additional details. Defaults to None.
        """
        history_entry = {
            "action": action,
            "timestamp": datetime.now(),
            "details": details or {}
        }
        self.history.append(history_entry)
        self.updated_at = history_entry["timestamp"]
    
    def update_description(self, new_description: str):
        """
        Update the hypothesis description.
        
        Args:
            new_description (str): New description for the hypothesis
        """
        old_description = self.description
        self.description = new_description
        self._record_history("description_updated", {
            "old_description": old_description,
            "new_description": new_description
        })
    
    def add_component(self, component: str):
        """
        Add a component to the hypothesis.
        
        Args:
            component (str): Component to add
        """
        self.components.append(component)
        self._record_history("component_added", {"component": component})
    
    def remove_component(self, component: str):
        """
        Remove a component from the hypothesis.
        
        Args:
            component (str): Component to remove
        """
        if component in self.components:
            self.components.remove(component)
            self._record_history("component_removed", {"component": component})
    
    def update_score(self, score: float, reason: Optional[str] = None):
        """
        Update the hypothesis score.
        
        Args:
            score (float): New score
            reason (str, optional): Reason for the score update. Defaults to None.
        """
        old_score = self.score
        self.score = score
        self._record_history("score_updated", {
            "old_score": old_score,
            "new_score": score,
            "reason": reason
        })
    
    def update_metadata(self, key: str, value: Any):
        """
        Update a metadata value.
        
        Args:
            key (str): Metadata key
            value (Any): Metadata value
        """
        old_value = self.metadata.get(key)
        self.metadata[key] = value
        self._record_history("metadata_updated", {
            "key": key,
            "old_value": old_value,
            "new_value": value
        })
    
    def merge(self, other_hypothesis: 'Hypothesis'):
        """
        Merge with another hypothesis.
        
        Args:
            other_hypothesis (Hypothesis): Hypothesis to merge with
            
        Returns:
            Hypothesis: New merged hypothesis
        """
        merged_description = f"Merged: {self.description} + {other_hypothesis.description}"
        merged_components = list(set(self.components + other_hypothesis.components))
        
        merged_metadata = self.metadata.copy()
        for key, value in other_hypothesis.metadata.items():
            if key in merged_metadata:
                if isinstance(value, list) and isinstance(merged_metadata[key], list):
                    merged_metadata[key] = list(set(merged_metadata[key] + value))
                elif isinstance(value, dict) and isinstance(merged_metadata[key], dict):
                    merged_metadata[key].update(value)
                else:
                    merged_metadata[key] = value
            else:
                merged_metadata[key] = value
        
        merged_hypothesis = Hypothesis(
            description=merged_description,
            components=merged_components,
            score=(self.score + other_hypothesis.score) / 2,
            metadata=merged_metadata
        )
        
        merged_hypothesis.metadata["parent_ids"] = [self.id, other_hypothesis.id]
        
        self._record_history("merged", {"with_hypothesis_id": other_hypothesis.id})
        other_hypothesis._record_history("merged", {"with_hypothesis_id": self.id})
        
        return merged_hypothesis
    
    def mutate(self, mutation_type: str, mutation_data: Dict[str, Any]) -> 'Hypothesis':
        """
        Create a mutated version of this hypothesis.
        
        Args:
            mutation_type (str): Type of mutation
            mutation_data (Dict[str, Any]): Data for the mutation
            
        Returns:
            Hypothesis: New mutated hypothesis
        """
        mutated_description = self.description
        mutated_components = self.components.copy()
        mutated_metadata = self.metadata.copy()
        
        if mutation_type == "add_component":
            new_component = mutation_data.get("component", "")
            if new_component and new_component not in mutated_components:
                mutated_components.append(new_component)
                mutated_description = f"{mutated_description} with {new_component}"
        
        elif mutation_type == "remove_component":
            component_to_remove = mutation_data.get("component", "")
            if component_to_remove and component_to_remove in mutated_components:
                mutated_components.remove(component_to_remove)
                mutated_description = mutated_description.replace(f" with {component_to_remove}", "")
        
        elif mutation_type == "modify_description":
            modification = mutation_data.get("modification", "")
            if modification:
                mutated_description = f"{mutated_description} {modification}"
        
        mutated_hypothesis = Hypothesis(
            description=mutated_description,
            components=mutated_components,
            score=self.score,
            metadata=mutated_metadata
        )
        
        mutated_hypothesis.metadata["parent_id"] = self.id
        mutated_hypothesis.metadata["mutation_type"] = mutation_type
        mutated_hypothesis.metadata["mutation_data"] = mutation_data
        
        self._record_history("mutated", {
            "mutation_type": mutation_type,
            "mutation_data": mutation_data,
            "child_hypothesis_id": mutated_hypothesis.id
        })
        
        return mutated_hypothesis
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the hypothesis to a dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the hypothesis
        """
        return {
            "id": self.id,
            "description": self.description,
            "components": self.components,
            "score": self.score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Hypothesis':
        """
        Create a hypothesis from a dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary representation of a hypothesis
            
        Returns:
            Hypothesis: Created hypothesis
        """
        hypothesis = cls(
            description=data["description"],
            components=data.get("components", []),
            score=data.get("score", 0.0),
            metadata=data.get("metadata", {})
        )
        
        hypothesis.id = data["id"]
        hypothesis.created_at = datetime.fromisoformat(data["created_at"])
        hypothesis.updated_at = datetime.fromisoformat(data["updated_at"])
        
        return hypothesis
    
    def __str__(self) -> str:
        """
        String representation of the hypothesis.
        
        Returns:
            str: String representation
        """
        return f"Hypothesis({self.id[:8]}): {self.description} (Score: {self.score:.2f})"
    
    def __repr__(self) -> str:
        """
        Representation of the hypothesis.
        
        Returns:
            str: Representation
        """
        return self.__str__() 
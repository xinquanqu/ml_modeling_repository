from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from app.models import AgentState

class GatewayBase(ABC):
    """Abstract base class for LLM Gateways."""
    
    @abstractmethod
    def process_message(self, message: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Process incoming user message and determine response/tool calls."""
        pass

class AgentBase(ABC):
    """Abstract base class for Agents."""
    
    @abstractmethod
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the agent with the given state."""
        pass
        
    @abstractmethod
    def get_graph_structure(self, subgraph_id: str = None) -> Dict[str, Any]:
        """Return the structure of the agent's graph."""
        pass

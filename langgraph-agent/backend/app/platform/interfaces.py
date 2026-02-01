from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional
from app.models import AgentState

class GatewayBase(ABC):
    """Abstract base class for LLM Gateways."""
    
    @abstractmethod
    def process_message(self, message: str, callbacks: List[Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """Process incoming user message and determine response/tool calls."""
        pass

class AgentBase(ABC):
    """Abstract base class for Agents."""
    
    @abstractmethod
    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke the agent with the given state."""
        pass
        
    @abstractmethod
    def get_graph_structure(self, subgraph_id: str = None) -> Dict[str, Any]:
        """Return the structure of the agent's graph."""
        pass

class LLMClient(ABC):
    """Abstract base class for LLM Adapters."""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, callbacks: List[Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate a response from the LLM based on messages and optional tools.
        Returns: (content, tool_calls)
        """
        pass

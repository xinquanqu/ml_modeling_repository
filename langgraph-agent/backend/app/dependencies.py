from app.platform.interfaces import AgentBase, GatewayBase
from app.domain.chat.agent import ChatAgent
from app.domain.chat.gateway import ChatGateway
from app.domain.chat import nodes
from functools import lru_cache

# Simple Dependency Injection Container

from app.infrastructure.llm.factory import get_llm_client

@lru_cache()
def get_gateway() -> GatewayBase:
    llm_client = get_llm_client()
    return ChatGateway(llm_client)

@lru_cache()
def get_agent() -> AgentBase:
    # Wire gateway into nodes if needed
    # The nodes are currently using a global placeholder logic in simple refactor
    # Let's set it here
    gateway = get_gateway()
    nodes.set_gateway(gateway)
    
    return ChatAgent()

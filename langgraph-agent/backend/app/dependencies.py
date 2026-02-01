from app.platform.interfaces import AgentBase, GatewayBase
from app.domain.chat.agent import ChatAgent
from app.domain.chat.gateway import ChatGateway
from app.domain.chat import nodes
from functools import lru_cache

# Simple Dependency Injection Container

@lru_cache()
def get_gateway() -> GatewayBase:
    return ChatGateway()

@lru_cache()
def get_agent() -> AgentBase:
    # Wire gateway into nodes if needed
    # The nodes are currently using a global placeholder logic in simple refactor
    # Let's set it here
    gateway = get_gateway()
    nodes.set_gateway(gateway)
    
    return ChatAgent()

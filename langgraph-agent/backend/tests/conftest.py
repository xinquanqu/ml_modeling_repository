import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from app.main import app
from app.models import AgentState
from app.platform.interfaces import GatewayBase, AgentBase
from langchain_core.messages import AIMessage
from app.domain.chat.gateway import ChatGateway
from app.dependencies import get_gateway, get_agent

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def mock_llm_gateway():
    """
    Fixture that mocks the LLMGateway.
    We patch the dependency injection to return our mock.
    """
    mock_instance = MagicMock(spec=ChatGateway)
    
    # Clear lru_cache to ensure we get a fresh instance
    get_gateway.cache_clear()
    get_agent.cache_clear()
    
    # Patch dependencies.get_gateway to return our mock
    with patch("app.dependencies.get_gateway", return_value=mock_instance):
        yield mock_instance

@pytest.fixture
def mock_agent():
    """
    Fixture that mocks the Agent.
    """
    mock_instance = MagicMock(spec=AgentBase)
    # Make invoke async
    async def async_invoke(*args, **kwargs):
         return {"messages": [AIMessage(content="Mocked Response")], "current_node": "end"}
    
    mock_instance.invoke.side_effect = async_invoke
    
    get_agent.cache_clear()
    with patch("app.dependencies.get_agent", return_value=mock_instance):
        yield mock_instance

@pytest.fixture
def demo_state():
    """Returns a fresh AgentState"""
    return AgentState(messages=[])

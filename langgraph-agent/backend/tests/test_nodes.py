from app.domain.chat.nodes import chatbot_node, tool_executor_node, should_use_tools, set_gateway
from app.models import AgentState
from unittest.mock import MagicMock
import pytest

# We need to ensure the node has the gateway set for testing
@pytest.fixture(autouse=True)
def setup_node_gateway(mock_llm_gateway):
    set_gateway(mock_llm_gateway)

def test_chatbot_node_run(mock_llm_gateway):
    """Test chatbot node calls gateway and updates state"""
    # Setup
    input_message = {"role": "user", "content": "Hello world"}
    state = AgentState(messages=[input_message])
    
    # Mock Gateway response
    mock_llm_gateway.process_message.return_value = ("Hello back", [])
    
    # Execution
    new_state = chatbot_node(state)
    
    # Verification
    # Check that gateway was called with content
    mock_llm_gateway.process_message.assert_called_once_with("Hello world")
    
    # Check state update
    assert new_state["current_node"] == "chatbot"
    assert len(new_state["messages"]) == 1
    assert new_state["messages"][0]["content"] == "Hello back"
    assert new_state["tool_calls"] == []

def test_chatbot_node_with_tools(mock_llm_gateway):
    """Test chatbot node handling tool calls"""
    # Setup
    state = AgentState(messages=[{"role": "user", "content": "weather in Tokyo"}])
    
    # Mock
    tool_calls_mock = [{"tool": "weather", "args": {"query": "Tokyo"}}]
    mock_llm_gateway.process_message.return_value = ("Checking weather...", tool_calls_mock)
    
    # Exec
    new_state = chatbot_node(state)
    
    # Verify
    assert new_state["tool_calls"] == tool_calls_mock
    assert new_state["messages"][0]["content"] == "Checking weather..."

def test_tool_executor_node_weather():
    """Test tool execution logic"""
    # Setup
    tool_calls = [{"tool": "weather", "args": {"query": "SF"}}]
    state = AgentState(
        messages=[], 
        tool_calls=tool_calls
    )
    
    # Exec
    new_state = tool_executor_node(state)
    
    # Verify
    assert "☀️" in new_state["messages"][0]["content"]
    assert new_state["current_node"] == "tool_executor"
    assert new_state["tool_calls"] == [] # Should be cleared

def test_should_use_tools():
    """Test conditional edge logic"""
    # Case 1: Has tools
    state_with_tools = AgentState(messages=[], tool_calls=[{"tool": "foo"}])
    assert should_use_tools(state_with_tools) == "tool_executor"
    
    # Case 2: No tools
    state_no_tools = AgentState(messages=[], tool_calls=[])
    assert should_use_tools(state_no_tools) == "end"

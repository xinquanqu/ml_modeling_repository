from unittest.mock import MagicMock
from app.domain.chat.gateway import ChatGateway
from app.platform.interfaces import LLMClient

def test_gateway_delegates_to_llm_client():
    """Test that ChatGateway delegates generation to the LLM Client."""
    # Setup Mock Client
    mock_client = MagicMock(spec=LLMClient)
    expected_response = "Mocked Response"
    expected_tools = [{"tool": "test", "args": {}}]
    mock_client.generate_response.return_value = (expected_response, expected_tools)
    
    # Instantiate Gateway with Mock
    gateway = ChatGateway(mock_client)
    
    # Execute
    response, tool_calls = gateway.process_message("hello")
    
    # Verify
    assert response == expected_response
    assert tool_calls == expected_tools
    
    # Verify call arguments
    mock_client.generate_response.assert_called_once()
    args, kwargs = mock_client.generate_response.call_args
    messages = args[0]
    tools = kwargs.get("tools")
    
    assert len(messages) == 2
    assert messages[1]["content"] == "hello"
    assert len(tools) == 2
    assert tools[0]["name"] == "weather"

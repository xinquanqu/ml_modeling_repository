from app.domain.chat.gateway import ChatGateway

def test_gateway_process_message_weather():
    """Test weather query routing"""
    gateway = ChatGateway()
    response, tool_calls = gateway.process_message("What is the weather?")
    
    assert "weather" in response.lower() or "check" in response.lower()
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "weather"

def test_gateway_process_message_search():
    """Test search query routing"""
    gateway = ChatGateway()
    response, tool_calls = gateway.process_message("search for python")
    
    assert "search" in response.lower()
    assert len(tool_calls) == 1
    assert tool_calls[0]["tool"] == "search"

def test_gateway_process_message_general():
    """Test general chat routing"""
    gateway = ChatGateway()
    response, tool_calls = gateway.process_message("hello")
    
    assert tool_calls == []
    assert "received your message" in response.lower()

from typing import List, Dict, Tuple, Any
from app.platform.interfaces import LLMClient
import time

class MockLLMAdapter(LLMClient):
    """Mock Adapter for testing/default."""
    
    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, callbacks: List[Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        # Extract last user message
        user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "").lower()
        
        # Simple mock logic
        if "weather" in user_message:
            return "I'll check the weather for you.", [{"tool": "weather", "args": {"query": user_message}, "id": "call_mock_weather"}]
        elif "search" in user_message:
            return "Searching the web...", [{"tool": "search", "args": {"query": user_message}, "id": "call_mock_search"}]
        elif "help" in user_message:
            return "I can help you with weather, search, and general inquiries.", []
        else:
            return f"I received your message: '{user_message}'. How can I help?", []

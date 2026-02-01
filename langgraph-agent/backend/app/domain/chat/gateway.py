from typing import Any, Dict, List, Tuple
from app.platform.interfaces import GatewayBase, LLMClient

class ChatGateway(GatewayBase):
    """Gateway implementation for the Chat domain."""
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.tools = [
            {
                "name": "weather",
                "description": "Get current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"}
                    },
                    "required": ["query"]
                }
            }
        ]

    def process_message(self, message: str, callbacks: List[Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process incoming user message and determine response/tool calls.
        Returns a tuple: (response_text, tool_calls_list)
        """
        # Construct simplified message history for single turn
        # In a real app, this might accept the full history or `state`
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message}
        ]
        
        return self.llm_client.generate_response(messages, tools=self.tools, callbacks=callbacks)

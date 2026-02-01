from typing import List, Dict, Tuple, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.platform.interfaces import LLMClient
import os

class GeminiAdapter(LLMClient):
    """Adapter for Google Gemini models."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.client = ChatGoogleGenerativeAI(model=model_name, temperature=0)

    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, callbacks: List[Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        lc_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))

        config = {"callbacks": callbacks} if callbacks else {}

        if tools:
            runnable = self.client.bind_tools(tools)
        else:
            runnable = self.client
            
        result = runnable.invoke(lc_messages, config=config)
        
        content = result.content
        tool_calls = []
        
        if result.tool_calls:
            for tc in result.tool_calls:
                tool_calls.append({
                    "tool": tc["name"],
                    "args": tc["args"],
                    "id": tc["id"]
                })
                
        return content, tool_calls

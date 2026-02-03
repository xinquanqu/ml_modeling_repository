from typing import List, Dict, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from app.platform.interfaces import LLMClient
import os

class OpenAIAdapter(LLMClient):
    """Adapter for OpenAI models."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            # Fallback or error? For now, let it fail during instantiation if key missing is standard
            # But maybe we want safe initialization
            pass
            
        self.client = ChatOpenAI(model=model_name, 
                                 api_key=api_key,
                                 temperature=0)

    def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict[str, Any]] = None, callbacks: List[Any] = None) -> Tuple[str, List[Dict[str, Any]]]:
        # Convert dict messages to LangChain messages
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
            # Handle tool messages if needed
                
        # Bind tools if provided
        config = {"callbacks": callbacks} if callbacks else {}
        
        if tools:
            # LangChain bind_tools usually takes functions or pydantic models or dicts
            # Assuming tools is a list of structured tool definitions
            runnable = self.client.bind_tools(tools)
        else:
            runnable = self.client
            
        result = runnable.invoke(lc_messages, config=config)
        
        # Parse result
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

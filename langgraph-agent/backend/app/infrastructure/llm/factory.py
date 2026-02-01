import os
from app.platform.interfaces import LLMClient
from app.infrastructure.llm.mock import MockLLMAdapter
from app.infrastructure.llm.openai import OpenAIAdapter
from app.infrastructure.llm.gemini import GeminiAdapter
from app.infrastructure.llm.claude import ClaudeAdapter

def get_llm_client(provider: str = None) -> LLMClient:
    """Factory to get the appropriate LLM Client."""
    if not provider:
        provider = os.getenv("LLM_PROVIDER", "mock").lower()
        
    if provider == "openai":
        return OpenAIAdapter()
    elif provider == "gemini":
        return GeminiAdapter()
    elif provider == "claude" or provider == "anthropic":
        return ClaudeAdapter()
    else:
        return MockLLMAdapter()

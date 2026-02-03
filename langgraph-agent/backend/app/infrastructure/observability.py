import os
from typing import Optional, Any

def get_langfuse_handler(session_id: str = None, user_id: str = None) -> Optional[Any]:
    """
    Initialize and return a Langfuse CallbackHandler.
    Returns None if proper credentials aren't set or integration is disabled.
    
    NOTE: Due to compatibility issues between Langfuse v2 and LangChain 1.0+,
    this integration may not work properly. The errors are caught and logged
    to prevent application crashes, but traces may not be sent to Langfuse.
    
    For a fully working solution, consider:
    1. Upgrading to Langfuse v3 (requires ClickHouse)
    2. Using Langfuse SDK directly instead of callback handlers
    3. Temporarily disabling Langfuse integration
    """
    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST", "http://langfuse-server:3000") 
    
    # Simple check if keys are essentially set
    if not (public_key and secret_key):
        print("Langfuse credentials not found in environment variables.")
        return None
    
    # Lazy import to ensure compatibility shim in main.py runs first
    try:
        from langfuse.callback import CallbackHandler
        
        handler = CallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            session_id=session_id,
            user_id=user_id,
            version="1.0.0"
        )
        return handler
    except Exception as e:
        import traceback
        print(f"Failed to initialize Langfuse handler: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        print("Langfuse integration disabled - application will continue without tracing")
        return None

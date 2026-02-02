# from langfuse.callback import CallbackHandler
import os
from typing import Optional, Any

def get_langfuse_handler(session_id: str = None, user_id: str = None) -> Optional[Any]:
    """
    Initialize and return a Langfuse CallbackHandler.
    Returns None if proper credentials aren't set or integration is disabled.
    """
    # Langfuse integration temporarily disabled
    return None

    # Original implementation commented out for future reference:
    # public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    # secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    # host = os.getenv("LANGFUSE_HOST")
    
    # if not (public_key and secret_key and host):
    #     return None
        
    # return CallbackHandler(
    #     public_key=public_key,
    #     secret_key=secret_key,
    #     host=host,
    #     session_id=session_id,
    #     user_id=user_id,
    #     version="1.0.0"
    # )

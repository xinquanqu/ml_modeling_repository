from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    state: Dict[str, Any]

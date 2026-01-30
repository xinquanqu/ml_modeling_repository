from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """State maintained throughout the agent's execution."""
    messages: Annotated[list, add_messages]
    current_node: str
    tool_calls: list
    iteration: int

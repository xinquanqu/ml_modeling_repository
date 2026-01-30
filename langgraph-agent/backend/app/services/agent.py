from langgraph.graph import StateGraph, START, END
from app.models import AgentState
from app.services.nodes import chatbot_node, tool_executor_node, should_use_tools

def build_agent_graph() -> StateGraph:
    """Construct the LangGraph agent."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tool_executor", tool_executor_node)
    
    # Add edges
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", should_use_tools, {
        "tool_executor": "tool_executor",
        "end": END,
    })
    graph.add_edge("tool_executor", END)
    
    return graph.compile()

# Singleton instance
agent = build_agent_graph()

def get_graph_structure() -> dict:
    """Export graph structure for frontend visualization."""
    return {
        "nodes": [
            {"id": "start", "label": "START", "type": "start"},
            {"id": "chatbot", "label": "Chatbot", "type": "node"},
            {"id": "tool_executor", "label": "Tool Executor", "type": "node"},
            {"id": "end", "label": "END", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "chatbot", "label": ""},
            {"from": "chatbot", "to": "tool_executor", "label": "has_tools", "conditional": True},
            {"from": "chatbot", "to": "end", "label": "no_tools", "conditional": True},
            {"from": "tool_executor", "to": "end", "label": ""},
        ],
    }

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

from typing import Optional, Dict, Any, List
from langgraph.graph.graph import Graph

def get_graph_structure(subgraph_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Dynamically build graph structure from the LangGraph agent.
    If subgraph_id is provided, attempts to find that node and return its internal graph.
    """
    # 1. Resolve the target graph (agent top-level or specific subgraph)
    target_graph = agent.get_graph()
    
    if subgraph_id:
        # Traverse to find the subgraph node
        found = False
        queue = [target_graph]
        visited = set()
        
        while queue:
            curr_graph = queue.pop(0)
            if id(curr_graph) in visited:
                continue
            visited.add(id(curr_graph))
            
            # Check nodes for the target ID
            # curr_graph.nodes is a dict-like object where keys are node IDs or it iterates over nodes
            # Based on explore_graph.py, iterating yields the node object themselves OR keys
            # Let's handle both safely
            
            target_node_data = None
            if isinstance(curr_graph.nodes, dict):
                 if subgraph_id in curr_graph.nodes:
                     target_node_data = curr_graph.nodes[subgraph_id]
            else:
                 # Assume iterable of nodes
                 for n in curr_graph.nodes:
                     if hasattr(n, "id") and n.id == subgraph_id:
                         target_node_data = n
                         break
            
            if target_node_data:
                 runnable = target_node_data.data
                 if hasattr(runnable, "get_graph"):
                     target_graph = runnable.get_graph()
                     found = True
                     break
            
            if found:
                break
                
            # Add subgraphs to queue to search deeper (if desired)
            # For now, simplistic search
            
        if not found:
            # If not found, revert to top level or error? Returning top level for safety
            pass

    # 2. Build normalized dictionary structure
    nodes = []
    edges = []
    
    # Process Nodes
    # Process Nodes
    # Handle if nodes is dict or list
    nodes_iter = target_graph.nodes.values() if isinstance(target_graph.nodes, dict) else target_graph.nodes
    
    for node in nodes_iter:
        # node is a Node object
        node_id = node.id
        node_label = node_id.replace("_", " ").title()
        node_type = "node"
        
        if node_id == "__start__":
            node_id = "start"
            node_label = "START"
            node_type = "start"
        elif node_id == "__end__":
            node_id = "end"
            node_label = "END"
            node_type = "end"
            
        # Check for subgraph capability
        is_subgraph = hasattr(node.data, "get_graph")
        
        nodes.append({
            "id": node_id,
            "label": node_label,
            "type": node_type,
            "is_subgraph": is_subgraph
        })

    # Process Edges
    for edge in target_graph.edges:
        # edge is an Edge object
        source = edge.source
        target = edge.target
        
        if source == "__start__": source = "start"
        if target == "__start__": target = "start"
        if source == "__end__": source = "end"
        if target == "__end__": target = "end"
        
        label = edge.data if isinstance(edge.data, str) else ""
        conditional = edge.conditional
        
        edges.append({
            "from": source,
            "to": target,
            "label": label,
            "conditional": conditional
        })
        
    return {
        "nodes": nodes,
        "edges": edges
    }

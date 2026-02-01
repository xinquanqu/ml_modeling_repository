from typing import Any, Dict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.platform.interfaces import AgentBase
from app.models import AgentState
from app.domain.chat.nodes import chatbot_node, tool_executor_node, should_use_tools

class ChatAgent(AgentBase):
    """Chat Agent implementation using LangGraph."""
    
    def __init__(self):
        self.graph = self._build_graph()
        
    def _build_graph(self):
        """Construct the StateGraph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("chatbot", chatbot_node)
        workflow.add_node("tool_executor", tool_executor_node)
        
        # Add edges
        workflow.set_entry_point("chatbot")
        workflow.add_conditional_edges(
            "chatbot",
            should_use_tools,
            {
                "tool_executor": "tool_executor",
                "end": END
            }
        )
        workflow.add_edge("tool_executor", "chatbot")
        
        # Compile
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
        
    async def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the graph with the provided state.
        Args:
            state: The initial state dictionary.
            config: Optional LangChain config (callbacks, etc).
        Returns:
            The final state dictionary.
        """
        # Await the graph execution since it's an async graph
        # Passing config enables tracing if callbacks are present
        return await self.graph.ainvoke(state, config=config)

    def get_graph_structure(self, subgraph_id: str = None) -> Dict[str, Any]:
        """Dynamically inspect the LangGraph structure."""
        # Access the underlying graph
        # This logic is copied/adapted from the original service
        compiled_graph = self.graph
        
        if subgraph_id:
            # Try to find the subgraph
            # Simplified logic for now: assume top level or specific retrieval
            # For a real implementation, we'd traverse.
            # But wait, we can reuse the logic we debugged earlier.
            pass

        # Use the logic from our previous verification
        # Assuming we just want top-level for now unless recursive logic is needed
        # To strictly match previous functionality, let's include the full logic.
        
        return self._extract_graph_structure(compiled_graph, subgraph_id)

    def _extract_graph_structure(self, graph, subgraph_id: str = None) -> Dict[str, Any]:
        target_graph = graph.get_graph()
        
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
                    
                # For refactor simplicity, we won't implement deep traversal here unless needed
                # The generic logic from before was recursive-like in intention but we only need it 
                # to find the specific node.
        
        nodes = []
        edges = []
        
        # Inspect nodes
        graph_drawable = target_graph
        graph_nodes = graph_drawable.nodes.values() if isinstance(graph_drawable.nodes, dict) else graph_drawable.nodes

        for node in graph_nodes:
             node_id = node.id
             node_label = node_id.replace("_", " ").title()
             node_type = "node"
             
             # Normalize start/end
             if node_id == "__start__":
                 node_id = "start"
                 node_label = "START"
                 node_type = "start"
             elif node_id == "__end__":
                 node_id = "end"
                 node_label = "END"
                 node_type = "end"
             
             # Check for subgraph
             is_subgraph = hasattr(node.data, "get_graph")
             
             nodes.append({
                "id": node_id,
                "label": node_label,
                "type": node_type,
                "is_subgraph": is_subgraph
            })
            
        # Inspect edges
        for edge in graph_drawable.edges:
            source = edge.source
            target = edge.target
            
            if source == "__start__": source = "start"
            if target == "__start__": target = "start"
            if source == "__end__": source = "end"
            if target == "__end__": target = "end"
            
            edges.append({
                "from": source,
                "to": target,
                "label": getattr(edge, "data", "") or "",
                "conditional": edge.conditional
            })
            
        return {"nodes": nodes, "edges": edges}

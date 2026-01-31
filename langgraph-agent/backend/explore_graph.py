from app.services.agent import agent
import json

def explore():
    graph = agent.get_graph()
    print(f"Graph type: {type(graph)}")
    
    # Print generic graph structure
    drawable = graph.draw_mermaid_png() # Just to trigger internal structure access if needed
    
    print("\nNodes:")
    for node in graph.nodes:
        print(f" - {node}")
        
    print("\nEdges:")
    for edge in graph.edges:
        print(f" - {edge}")
        
    # Try to access internal dict representation if possible standard langchain method
    print("\nChecking for subgraphs:")
    for node in graph.nodes:
        # Check if corresponding runnable is a graph
        # Note: 'graph.nodes' are just IDs/Schemas in this view, we need the runnable
        # Accessing runnable from the graph nodes dictionary if possible
        try:
           node_obj = graph.nodes[node.id] if hasattr(node, 'id') else graph.nodes.get(node)
           print(f"Node: {node}, Type: {type(node_obj)}")
           print(f"Node Data Type: {type(node_obj.data)}")
           print(f"Node Data: {node_obj.data}")
           # Check if it has get_graph method
           if hasattr(node_obj.data, "get_graph"):
               print(f" *** Node {node} is a subgraph! ***")
        except Exception as e:
           print(f"Could not inspect {node}: {e}")


if __name__ == "__main__":
    explore()

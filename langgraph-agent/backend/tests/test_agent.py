from app.domain.chat.agent import ChatAgent

def test_get_graph_structure_top_level():
    """Test retrieving top level graph structure"""
    agent = ChatAgent()
    structure = agent.get_graph_structure()
    
    assert "nodes" in structure
    assert "edges" in structure
    
    # Check for core nodes
    node_ids = [n["id"] for n in structure["nodes"]]
    assert "start" in node_ids
    assert "chatbot" in node_ids
    assert "tool_executor" in node_ids
    assert "end" in node_ids
    
    # Check chatbot is marked as subgraph
    chatbot_node = next(n for n in structure["nodes"] if n["id"] == "chatbot")
    assert chatbot_node.get("is_subgraph") is True

def test_get_graph_structure_subgraph():
    """Test retrieving a specific subgraph"""
    agent = ChatAgent()
    # Fetch chatbot subgraph
    structure = agent.get_graph_structure(subgraph_id="chatbot")
    
    assert "nodes" in structure
    assert "edges" in structure
    
    assert len(structure["nodes"]) > 0

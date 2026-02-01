def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_graph_endpoint(client):
    response = client.get("/graph")
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data

def test_graph_endpoint_subgraph(client):
    """Test fetching a subgraph"""
    response = client.get("/graph?node_id=chatbot")
    if response.status_code == 200:
        data = response.json()
        assert "nodes" in data
    else:
        # If accessing the subgraph fails (e.g. invalid ID handling or finding), fail safely
        # But we expect 200 based on our manual verification
        assert response.status_code == 200

def test_chat_endpoint(client, mock_agent):
    """Test full chat flow"""
    # mock_agent fixture already sets up side_effect for async invoke
    # returning "Mocked Response"
    
    payload = {"message": "Hi there"}
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "response" in data
    assert "state" in data
    
    # Check content from mock_agent
    assert data["response"] == "Mocked Response"
    
    # Verify mock was called
    mock_agent.invoke.assert_called()

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

def test_chat_endpoint(client, mock_llm_gateway):
    """Test full chat flow"""
    # Setup mock
    mock_llm_gateway.process_message.return_value = ("Hello from mock", [])
    
    payload = {"message": "Hi there"}
    # The router now uses get_agent() -> get_gateway()
    # verify_gateway mock in conftest patches get_gateway() so the agent instantiated
    # within the router (or cached) will use our mock gateway via set_gateway
    
    response = client.post("/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check response structure
    assert "response" in data
    assert "state" in data
    
    # Check content
    # The API might be returning the last message content or the full state
    # Let's check based on typical response "response": "Hello from mock"
    assert data["response"] == "Hello from mock"
    
    # Verify mock was called
    mock_llm_gateway.process_message.assert_called()

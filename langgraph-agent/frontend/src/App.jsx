import React, { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import ChatPanel from './components/ChatPanel';
import GraphSidebar from './components/GraphSidebar';
import StateSidebar from './components/StateSidebar';
import './styles/App.css';

// Direct connection config to avoid proxy issues for now
const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

function App() {
  const [messages, setMessages] = useState([]);
  const [graphStructure, setGraphStructure] = useState(null);
  const [currentState, setCurrentState] = useState({});
  const [activeNode, setActiveNode] = useState(null);
  const [stateHistory, setStateHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  // WebSocket Message Handler
  const handleWebSocketMessage = useCallback((data) => {
    console.log("WS Data:", data);

    if (data.type === 'state_update') {
      setActiveNode(data.node);
      setCurrentState(data.state || {});

      // Add to history
      setStateHistory(prev => [...prev, {
        timestamp: new Date().toISOString(),
        node: data.node,
        state: data.state
      }]);

      // If node is end, loading is done? Not necessarily if waiting for response
      if (data.node !== 'end') setIsLoading(true);

    } else if (data.type === 'response') {
      setMessages(prev => [...prev, { role: 'assistant', content: data.content }]);
      if (data.final_state) setCurrentState(data.final_state);
      setActiveNode('end');
      setIsLoading(false);

    } else if (data.type === 'error') {
      setMessages(prev => [...prev, { role: 'system', content: `Error: ${data.error}` }]);
      setIsLoading(false);
    }
  }, []);

  const { isConnected, sendMessage } = useWebSocket(WS_URL, handleWebSocketMessage);

  // Fetch Graph Structure on Mount
  useEffect(() => {
    fetch(`${API_URL}/graph`)
      .then(res => {
        if (!res.ok) throw new Error("Failed to fetch graph");
        return res.json();
      })
      .then(data => setGraphStructure(data))
      .catch(err => console.error("Graph fetch error:", err));
  }, []);

  const handleSendMessage = (text) => {
    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setIsLoading(true);
    setActiveNode('start');
    sendMessage({ message: text });
  };

  const handleClear = () => {
    setMessages([]);
    setCurrentState({});
    setActiveNode(null);
    setStateHistory([]);
  };

  return (
    <div className="app">
      <GraphSidebar
        graphStructure={graphStructure}
        activeNode={activeNode}
      />

      <ChatPanel
        messages={messages}
        onSendMessage={handleSendMessage}
        isLoading={isLoading}
        connectionStatus={isConnected}
        onClear={handleClear}
      />

      <StateSidebar
        currentState={currentState}
        stateHistory={stateHistory}
        activeNode={activeNode}
      />
    </div>
  );
}

export default App;

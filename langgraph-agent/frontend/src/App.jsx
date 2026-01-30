import React, { useState, useEffect, useCallback } from 'react';
import ChatPanel from './components/ChatPanel';
import GraphSidebar from './components/GraphSidebar';
import StateSidebar from './components/StateSidebar';
import { useWebSocket } from './hooks/useWebSocket';
import './styles/App.css';

const API_URL = 'http://localhost:8000';
const WS_URL = 'ws://localhost:8000/ws';

function App() {
  const [messages, setMessages] = useState([]);
  const [graphStructure, setGraphStructure] = useState(null);
  const [currentState, setCurrentState] = useState({});
  const [activeNode, setActiveNode] = useState(null);
  const [stateHistory, setStateHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  const { sendMessage, lastMessage, connectionStatus } = useWebSocket(WS_URL);

  // Fetch graph structure on mount
  useEffect(() => {
    fetch(`${API_URL}/graph`)
      .then(res => res.json())
      .then(data => setGraphStructure(data))
      .catch(err => console.error('Failed to fetch graph:', err));
  }, []);

  // Handle WebSocket messages
  useEffect(() => {
    if (!lastMessage) return;

    if (lastMessage.type === 'state_update') {
      setActiveNode(lastMessage.node);
      setCurrentState(lastMessage.state);
      setStateHistory(prev => [...prev, {
        timestamp: new Date().toISOString(),
        node: lastMessage.node,
        state: lastMessage.state,
      }]);
    } else if (lastMessage.type === 'response') {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: lastMessage.content,
      }]);
      setCurrentState(lastMessage.final_state);
      setActiveNode('end');
      setIsLoading(false);
    }
  }, [lastMessage]);

  const handleSendMessage = useCallback((text) => {
    if (!text.trim()) return;

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    setIsLoading(true);
    setActiveNode('start');
    sendMessage({ message: text });
  }, [sendMessage]);

  const clearChat = useCallback(() => {
    setMessages([]);
    setStateHistory([]);
    setCurrentState({});
    setActiveNode(null);
  }, []);

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
        connectionStatus={connectionStatus}
        onClear={clearChat}
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

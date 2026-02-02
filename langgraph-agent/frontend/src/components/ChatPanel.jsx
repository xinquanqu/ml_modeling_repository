import React, { useState, useRef, useEffect } from 'react';

function ChatPanel({ messages, onSendMessage, isLoading, connectionStatus, onClear }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading && connectionStatus) {
      onSendMessage(input);
      setInput('');
    }
  };

  const isConnected = connectionStatus; // Passed as boolean in new App

  return (
    <main className="chat-panel">
      <header className="chat-header">
        <h1>LangGraph Agent</h1>
        <div className="header-actions">
          <span className="connection-status" style={{ color: isConnected ? '#00ff88' : '#ff4757' }}>
            {isConnected ? '● Connected' : '● Disconnected'}
          </span>
          <button className="clear-btn" onClick={onClear}>Clear</button>
        </div>
      </header>

      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <p className="empty-icon">💬</p>
            <p>Start a conversation</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-avatar">
                {msg.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className="message-content">
                <span className="message-role">{msg.role}</span>
                <p>{msg.content}</p>
              </div>
            </div>
          ))
        )}

        {isLoading && (
          <div className="message assistant loading">
            <div className="message-content">
              <p>Processing...</p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <form className="input-form" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your message..."
          disabled={!isConnected || isLoading}
        />
        <button type="submit" disabled={!isConnected || isLoading || !input.trim()}>
          Send
        </button>
      </form>
    </main>
  );
}

export default ChatPanel;

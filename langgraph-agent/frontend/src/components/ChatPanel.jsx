import React, { useState, useRef, useEffect } from 'react';

function ChatPanel({ messages, onSendMessage, isLoading, connectionStatus, onClear }) {
  const [input, setInput] = useState('');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const statusColor = {
    connected: '#10b981',
    connecting: '#f59e0b',
    disconnected: '#ef4444',
  };

  return (
    <main className="chat-panel">
      <header className="chat-header">
        <h1>LangGraph Agent</h1>
        <div className="header-actions">
          <span
            className="connection-status"
            style={{ color: statusColor[connectionStatus] || '#888' }}
          >
            ● {connectionStatus}
          </span>
          <button className="clear-btn" onClick={onClear}>
            Clear
          </button>
        </div>
      </header>

      <div className="messages-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <div className="empty-icon">💬</div>
            <p>Start a conversation with the agent</p>
            <p className="hint">Try asking about weather, searching for info, or just say hi!</p>
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
            <div className="message-avatar">🤖</div>
            <div className="message-content">
              <span className="message-role">assistant</span>
              <div className="typing-indicator">
                <span></span><span></span><span></span>
              </div>
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
          disabled={isLoading || connectionStatus !== 'connected'}
        />
        <button
          type="submit"
          disabled={isLoading || !input.trim() || connectionStatus !== 'connected'}
        >
          Send
        </button>
      </form>
    </main>
  );
}

export default ChatPanel;

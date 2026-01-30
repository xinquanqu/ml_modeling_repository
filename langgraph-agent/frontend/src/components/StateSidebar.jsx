import React, { useState } from 'react';

function StateSidebar({ currentState, stateHistory, activeNode }) {
  const [showHistory, setShowHistory] = useState(false);

  const formatValue = (value) => {
    if (typeof value === 'object') {
      return JSON.stringify(value, null, 2);
    }
    return String(value);
  };

  return (
    <aside className="sidebar state-sidebar">
      <h2>Agent State</h2>

      <div className="state-tabs">
        <button 
          className={`tab ${!showHistory ? 'active' : ''}`}
          onClick={() => setShowHistory(false)}
        >
          Current
        </button>
        <button 
          className={`tab ${showHistory ? 'active' : ''}`}
          onClick={() => setShowHistory(true)}
        >
          History ({stateHistory.length})
        </button>
      </div>

      {!showHistory ? (
        <div className="current-state">
          <div className="state-section">
            <h3>Active Node</h3>
            <div className={`active-node-display ${activeNode || 'none'}`}>
              {activeNode || 'None'}
            </div>
          </div>

          <div className="state-section">
            <h3>State Values</h3>
            {Object.keys(currentState).length === 0 ? (
              <p className="empty-state-text">No state yet. Send a message to see state updates.</p>
            ) : (
              <div className="state-values">
                {Object.entries(currentState).map(([key, value]) => (
                  <div key={key} className="state-item">
                    <span className="state-key">{key}</span>
                    <pre className="state-value">{formatValue(value)}</pre>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="state-history">
          {stateHistory.length === 0 ? (
            <p className="empty-state-text">No history yet.</p>
          ) : (
            <div className="history-list">
              {[...stateHistory].reverse().map((entry, idx) => (
                <div key={idx} className="history-entry">
                  <div className="history-header">
                    <span className={`history-node ${entry.node}`}>
                      {entry.node}
                    </span>
                    <span className="history-time">
                      {new Date(entry.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="history-state">
                    {Object.entries(entry.state || {}).map(([key, value]) => (
                      <div key={key} className="history-item">
                        <span className="history-key">{key}:</span>
                        <span className="history-value">
                          {typeof value === 'object' ? JSON.stringify(value) : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </aside>
  );
}

export default StateSidebar;

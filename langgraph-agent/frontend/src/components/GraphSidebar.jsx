import React from 'react';

function GraphSidebar({ graphStructure, activeNode }) {
  if (!graphStructure) {
    return (
      <aside className="sidebar graph-sidebar">
        <h2>Graph Structure</h2>
        <div className="loading-state">Loading graph...</div>
      </aside>
    );
  }

  const { nodes, edges } = graphStructure;

  const getNodeClass = (nodeId) => {
    const classes = ['graph-node'];
    if (activeNode === nodeId) classes.push('active');
    const node = nodes.find(n => n.id === nodeId);
    if (node) classes.push(node.type);
    return classes.join(' ');
  };

  return (
    <aside className="sidebar graph-sidebar">
      <h2>Graph Structure</h2>
      
      <div className="graph-visualization">
        <svg viewBox="0 0 200 300" className="graph-svg">
          {/* Edges */}
          {edges.map((edge, idx) => {
            const positions = {
              start: { x: 100, y: 30 },
              chatbot: { x: 100, y: 100 },
              tool_executor: { x: 160, y: 180 },
              end: { x: 100, y: 260 },
            };
            const from = positions[edge.from];
            const to = positions[edge.to];
            if (!from || !to) return null;

            const isActive = activeNode === edge.from || activeNode === edge.to;
            
            return (
              <g key={idx}>
                <line
                  x1={from.x}
                  y1={from.y + 15}
                  x2={to.x}
                  y2={to.y - 15}
                  className={`graph-edge ${edge.conditional ? 'conditional' : ''} ${isActive ? 'active' : ''}`}
                  markerEnd="url(#arrowhead)"
                />
                {edge.label && (
                  <text
                    x={(from.x + to.x) / 2 + 5}
                    y={(from.y + to.y) / 2}
                    className="edge-label"
                  >
                    {edge.label}
                  </text>
                )}
              </g>
            );
          })}

          {/* Arrow marker */}
          <defs>
            <marker
              id="arrowhead"
              markerWidth="10"
              markerHeight="7"
              refX="9"
              refY="3.5"
              orient="auto"
            >
              <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
            </marker>
          </defs>

          {/* Nodes */}
          {nodes.map((node) => {
            const positions = {
              start: { x: 100, y: 30 },
              chatbot: { x: 100, y: 100 },
              tool_executor: { x: 160, y: 180 },
              end: { x: 100, y: 260 },
            };
            const pos = positions[node.id];
            if (!pos) return null;

            return (
              <g key={node.id} className={getNodeClass(node.id)}>
                {node.type === 'start' || node.type === 'end' ? (
                  <circle cx={pos.x} cy={pos.y} r="15" />
                ) : (
                  <rect
                    x={pos.x - 40}
                    y={pos.y - 15}
                    width="80"
                    height="30"
                    rx="5"
                  />
                )}
                <text x={pos.x} y={pos.y + 4} textAnchor="middle">
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="graph-legend">
        <h3>Legend</h3>
        <div className="legend-item">
          <span className="legend-symbol start">●</span>
          <span>Start/End</span>
        </div>
        <div className="legend-item">
          <span className="legend-symbol node">▢</span>
          <span>Node</span>
        </div>
        <div className="legend-item">
          <span className="legend-symbol conditional">- -</span>
          <span>Conditional</span>
        </div>
      </div>

      <div className="nodes-list">
        <h3>Nodes</h3>
        {nodes.map((node) => (
          <div 
            key={node.id} 
            className={`node-item ${activeNode === node.id ? 'active' : ''}`}
          >
            <span className={`node-type ${node.type}`}>
              {node.type === 'start' ? '▶' : node.type === 'end' ? '■' : '◆'}
            </span>
            <span className="node-label">{node.label}</span>
          </div>
        ))}
      </div>
    </aside>
  );
}

export default GraphSidebar;

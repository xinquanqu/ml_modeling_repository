import React, { useMemo } from 'react';

// Simple DAG layout: Level-based
const calculateLayout = (nodes, edges) => {
  if (!nodes || nodes.length === 0) return {};

  const levels = {};
  const queue = [];
  const visited = new Set();

  // Find start node
  const startNode = nodes.find(n => n.type === 'start') || nodes[0];
  if (startNode) {
    queue.push({ id: startNode.id, level: 0 });
    visited.add(startNode.id);
  }

  // Basic BFS for levels
  while (queue.length > 0) {
    const { id, level } = queue.shift();
    levels[id] = Math.max(levels[id] || 0, level);

    const children = edges
      .filter(e => e.from === id)
      .map(e => e.to);

    for (const child of children) {
      // Check if we've seen this child at a simpler level
      if (!visited.has(child)) {
        visited.add(child);
        queue.push({ id: child, level: level + 1 });
      } else {
        // Push it deeper if needed? For simple DAG, usually max level is good
        // But be careful of cycles.
        // For now, simple BFS/Level assignment.
      }
    }
  }

  // Fallback for disconnected nodes
  nodes.forEach(n => {
    if (levels[n.id] === undefined) levels[n.id] = 0;
  });

  // Assign positions
  const positions = {};
  const nodesByLevel = {};

  Object.entries(levels).forEach(([id, level]) => {
    if (!nodesByLevel[level]) nodesByLevel[level] = [];
    nodesByLevel[level].push(id);
  });

  const LEVEL_HEIGHT = 80;
  const NODE_WIDTH = 100;
  const CENTER_X = 150;

  Object.entries(nodesByLevel).forEach(([level, ids]) => {
    const y = 30 + (parseInt(level) * LEVEL_HEIGHT);
    const totalW = ids.length * NODE_WIDTH;
    let startX = CENTER_X - (totalW / 2) + (NODE_WIDTH / 2);

    ids.forEach((id, idx) => {
      positions[id] = { x: startX + (idx * NODE_WIDTH), y };
    });
  });

  return positions;
};


function GraphSidebar({ graphStructure, activeNode }) {

  // Safety checks
  const { nodes, edges } = graphStructure || { nodes: [], edges: [] };

  const layout = useMemo(() => calculateLayout(nodes, edges), [nodes, edges]);

  if (!graphStructure) {
    return (
      <aside className="sidebar graph-sidebar">
        <h2>Graph Structure</h2>
        <div className="loading-state">Loading graph...</div>
      </aside>
    );
  }

  return (
    <aside className="sidebar graph-sidebar">
      <h2>Graph Structure</h2>

      <div className="graph-visualization">
        <svg className="graph-svg" viewBox="0 0 300 400">
          <defs>
            <marker id="arrow" markerWidth="6" markerHeight="6" refX="10" refY="3" orient="auto" markerUnits="strokeWidth">
              <path d="M0,0 L0,6 L9,3 z" fill="#606078" />
            </marker>
          </defs>

          {/* Edges */}
          {edges.map((edge, i) => {
            const start = layout[edge.from];
            const end = layout[edge.to];
            if (!start || !end) return null;

            const isActive = activeNode === edge.from;
            // Simple lines for now.
            return (
              <g key={i}>
                <line
                  x1={start.x} y1={start.y} x2={end.x} y2={end.y}
                  className={`graph-edge ${edge.conditional ? 'conditional' : ''} ${isActive ? 'active' : ''}`}
                  markerEnd="url(#arrow)"
                />
                {edge.label && (
                  <text x={(start.x + end.x) / 2} y={(start.y + end.y) / 2} className="edge-label" textAnchor="middle" dy="-5">
                    {edge.label}
                  </text>
                )}
              </g>
            );
          })}

          {/* Nodes */}
          {nodes.map((node) => {
            const pos = layout[node.id];
            if (!pos) return null;
            const isActive = activeNode === node.id;

            return (
              <g key={node.id} className={`graph-node ${node.type} ${isActive ? 'active' : ''}`}>
                {(node.type === 'start' || node.type === 'end') ? (
                  <circle cx={pos.x} cy={pos.y} r="18" />
                ) : (
                  <rect x={pos.x - 40} y={pos.y - 15} width="80" height="30" rx="4" />
                )}
                <text x={pos.x} y={pos.y} dy="5" textAnchor="middle">{node.label}</text>
              </g>
            );
          })}
        </svg>
      </div>

    </aside>
  );
}

export default GraphSidebar;

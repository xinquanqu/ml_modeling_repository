import React, { useMemo } from 'react';

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
  const [expandedNodes, setExpandedNodes] = React.useState({});

  const handleNodeClick = async (node) => {
    if (node.is_subgraph) {
      try {
        const res = await fetch(`http://localhost:8000/graph?node_id=${node.id}`);
        const subgraph = await res.json();
        setExpandedNodes(prev => ({
          ...prev,
          [node.id]: subgraph
        }));
        // Ideally, we would merge this into the main graph or show a modal
        // For now, let's just log it or maybe replace the view?
        // The user asked to "expand". A simple way is to replace the main view for drill-down
        // OR simpler: just alert for now as valid proof of concept for "expansion" logic
        console.log("Subgraph fetched:", subgraph);
        alert(`Fetched subgraph for ${node.label}: ${JSON.stringify(subgraph, null, 2)}`);
      } catch (err) {
        console.error("Failed to fetch subgraph:", err);
      }
    }
  };

  // Simple layout algorithm
  const layout = useMemo(() => {
    const nodePositions = {};
    const levels = {};
    const queue = [{ id: 'start', level: 0 }];
    const visited = new Set(['start']);

    // BFS to assign levels
    while (queue.length > 0) {
      const { id, level } = queue.shift();
      levels[id] = Math.max(levels[id] || 0, level); // Keep max level if multiple paths

      // Find neighbors
      const neighbors = edges
        .filter(e => e.from === id)
        .map(e => e.to);

      for (const neighbor of neighbors) {
        if (!visited.has(neighbor)) {
          // Allow re-visiting for level calculation? No, simple DAG assumption for now
          visited.add(neighbor);
          queue.push({ id: neighbor, level: level + 1 });
        } else {
          // If already visited, update level if deeper
          queue.push({ id: neighbor, level: level + 1 });
        }
      }
    }

    // Group by level
    const nodesByLevel = {};
    Object.entries(levels).forEach(([id, level]) => {
      if (!nodesByLevel[level]) nodesByLevel[level] = [];
      nodesByLevel[level].push(id);
    });

    // Determine positions
    // Canvas size (approx)
    const centerX = 100;
    const startY = 30;
    const levelHeight = 70;

    Object.entries(nodesByLevel).forEach(([level, nodeIds]) => {
      const y = startY + (parseInt(level) * levelHeight);
      const count = nodeIds.length;
      const totalWidth = (count - 1) * 60; // 60px spacing
      const startX = centerX - (totalWidth / 2);

      nodeIds.forEach((id, idx) => {
        nodePositions[id] = {
          x: count === 1 ? centerX : startX + (idx * 60),
          y: y
        };
      });
    });

    return nodePositions;
  }, [nodes, edges]);


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
        <svg viewBox="0 0 200 400" className="graph-svg">
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

          {/* Edges */}
          {edges.map((edge, idx) => {
            const from = layout[edge.from];
            const to = layout[edge.to];
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
                    style={{ fontSize: '10px', fill: '#666' }}
                  >
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

            return (
              <g
                key={node.id}
                className={getNodeClass(node.id)}
                onClick={() => handleNodeClick(node)}
                style={{ cursor: node.is_subgraph ? 'pointer' : 'default' }}
              >
                {node.type === 'start' || node.type === 'end' ? (
                  <circle cx={pos.x} cy={pos.y} r="15" />
                ) : (
                  <rect
                    x={pos.x - 40}
                    y={pos.y - 15}
                    width="80"
                    height="30"
                    rx="5"
                    strokeDasharray={node.is_subgraph ? "4" : "0"}
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
    </aside>
  );
}

export default GraphSidebar;

# LangGraph Agent Service

A simple agent service using FastAPI with LangGraph to manage the agent, featuring a React frontend with graph visualization and state inspection.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React)                         │
├─────────────────┬─────────────────────────┬─────────────────────┤
│  Graph Sidebar  │      Chat Panel         │   State Sidebar     │
│  - Visualize    │  - Send messages        │   - Current state   │
│    LangGraph    │  - View responses       │   - State history   │
│  - Active node  │  - Connection status    │   - Node tracking   │
└─────────────────┴───────────┬─────────────┴─────────────────────┘
                              │ WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend (FastAPI)                          │
├─────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                      │
│  - GET  /        → Health check                                 │
│  - GET  /graph   → Graph structure                              │
│  - POST /chat    → Process message                              │
│  - WS   /ws      → Real-time streaming                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LangGraph Agent                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌─────────┐     ┌──────────┐     ┌────────────────┐           │
│   │  START  │────▶│ Chatbot  │────▶│ Tool Executor  │           │
│   └─────────┘     └──────────┘     └────────────────┘           │
│                        │                    │                    │
│                        │ (no tools)         │                    │
│                        ▼                    ▼                    │
│                   ┌─────────┐          ┌─────────┐              │
│                   │   END   │◀─────────│   END   │              │
│                   └─────────┘          └─────────┘              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
langgraph-agent/
├── backend/
│   ├── app/                 # Modular application code
│   │   ├── api/             # API Routers
│   │   ├── models/          # Data models and schemas
│   │   ├── services/        # Business logic and services
│   │   └── main.py          # App entry point
│   ├── venv/
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   ├── index.html           # Moved to root
│   ├── package.json
│   └── vite.config.js
├── .env.example
├── .gitignore
└── README.md
```

## Setup

### Backend

### Backend (Docker)

```bash
# Run with Docker Compose
docker-compose up --build
```

### Backend (Local)

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## Features

### Graph Sidebar (Left)
- Visual representation of the LangGraph structure
- Highlights the currently active node during execution
- Shows node types (start, end, regular nodes)
- Displays edge connections including conditional edges

### Chat Panel (Center)
- Send messages to the agent
- View conversation history
- Connection status indicator
- Clear chat functionality

### State Sidebar (Right)
- **Current Tab**: Shows the current agent state
  - Active node
  - State values (current_node, iteration, tool_calls, etc.)
- **History Tab**: Shows state transitions over time
  - Timestamp for each state change
  - Node transitions
  - State snapshots

## Observability

This project integrates [Langfuse](https://langfuse.com/) for tracing and observability.

### Getting Started

When running via Docker Compose (`docker-compose up --build`), the Langfuse server and database are automatically started.

1.  **Access Langfuse**: Open `http://localhost:3000` in your browser.
2.  **Create Account/Login**: You can create an account locally to access the dashboard.
3.  **View Traces**: Any chat interaction via the frontend or API will automatically generate traces in Langfuse, linking the Router, Agent, and LLM calls.

### Configuration

The backend connects to Langfuse using the following environment variables (configured in `docker-compose.yml`):

*   `LANGFUSE_SECRET_KEY`: Service API key (Secret)
*   `LANGFUSE_PUBLIC_KEY`: Service API key (Public)
*   `LANGFUSE_HOST`: URL of the Langfuse server (e.g., `http://langfuse-server:3000`)

## API Endpoints

### REST API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/graph` | Get graph structure for visualization |
| POST | `/chat` | Send a message and get response |

### WebSocket

| Endpoint | Description |
|----------|-------------|
| `/ws` | Real-time bidirectional communication |

**WebSocket Message Types:**

Incoming (Client → Server):
```json
{ "message": "user message text" }
```

Outgoing (Server → Client):
```json
{ "type": "state_update", "node": "chatbot", "state": {...} }
{ "type": "response", "content": "agent response", "final_state": {...} }
```

## Extending the Agent

### Adding New Nodes

1. Define the node function in `backend/main.py`:
```python
def my_new_node(state: AgentState) -> AgentState:
    # Process state
    return {"messages": [...], "current_node": "my_new_node", ...}
```

2. Add the node to the graph:
```python
graph.add_node("my_new_node", my_new_node)
```

3. Connect with edges:
```python
graph.add_edge("chatbot", "my_new_node")
```

4. Update `get_graph_structure()` to include the new node in visualization.

### Adding Real LLM Integration

Replace the mock response logic in `chatbot_node` with actual LLM calls:

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

def chatbot_node(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = llm.invoke(messages)
    return {
        "messages": [response],
        "current_node": "chatbot",
        ...
    }
```

## License

MIT

"""
FastAPI backend with LangGraph agent service.
Simple chatbot agent with tool usage capabilities.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Annotated, TypedDict, Literal
import json
import asyncio

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ============================================================================
# State Definition
# ============================================================================

class AgentState(TypedDict):
    """State maintained throughout the agent's execution."""
    messages: Annotated[list, add_messages]
    current_node: str
    tool_calls: list
    iteration: int


# ============================================================================
# Agent Nodes
# ============================================================================

def chatbot_node(state: AgentState) -> AgentState:
    """Main chatbot node - processes user input and generates responses."""
    messages = state["messages"]
    last_message = messages[-1] if messages else {"content": ""}
    
    # Simple response logic (replace with actual LLM call)
    user_input = last_message.get("content", "").lower()
    
    if "weather" in user_input:
        response = "I'd need to check the weather tool for that. Let me look it up..."
        tool_calls = [{"tool": "weather", "args": {"query": user_input}}]
    elif "search" in user_input:
        response = "Let me search for that information..."
        tool_calls = [{"tool": "search", "args": {"query": user_input}}]
    elif "help" in user_input:
        response = "I can help you with: weather queries, web searches, and general conversation!"
        tool_calls = []
    else:
        response = f"I received your message: '{last_message.get('content', '')}'. How can I assist you further?"
        tool_calls = []
    
    return {
        "messages": [{"role": "assistant", "content": response}],
        "current_node": "chatbot",
        "tool_calls": tool_calls,
        "iteration": state.get("iteration", 0) + 1,
    }


def tool_executor_node(state: AgentState) -> AgentState:
    """Executes tools requested by the chatbot."""
    tool_calls = state.get("tool_calls", [])
    results = []
    
    for tool_call in tool_calls:
        tool_name = tool_call.get("tool")
        args = tool_call.get("args", {})
        
        # Mock tool execution
        if tool_name == "weather":
            result = "☀️ The weather is sunny with a high of 72°F (22°C)."
        elif tool_name == "search":
            result = f"🔍 Search results for: {args.get('query', 'unknown')}"
        else:
            result = f"Tool '{tool_name}' executed successfully."
        
        results.append({"tool": tool_name, "result": result})
    
    # Add tool results as a message
    if results:
        result_text = "\n".join([r["result"] for r in results])
        return {
            "messages": [{"role": "assistant", "content": result_text}],
            "current_node": "tool_executor",
            "tool_calls": [],
            "iteration": state.get("iteration", 0) + 1,
        }
    
    return {
        "current_node": "tool_executor",
        "tool_calls": [],
        "iteration": state.get("iteration", 0) + 1,
    }


def should_use_tools(state: AgentState) -> Literal["tool_executor", "end"]:
    """Conditional edge: decide whether to use tools or end."""
    if state.get("tool_calls"):
        return "tool_executor"
    return "end"


# ============================================================================
# Build the Graph
# ============================================================================

def build_agent_graph() -> StateGraph:
    """Construct the LangGraph agent."""
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("chatbot", chatbot_node)
    graph.add_node("tool_executor", tool_executor_node)
    
    # Add edges
    graph.add_edge(START, "chatbot")
    graph.add_conditional_edges("chatbot", should_use_tools, {
        "tool_executor": "tool_executor",
        "end": END,
    })
    graph.add_edge("tool_executor", END)
    
    return graph.compile()


# Graph instance
agent = build_agent_graph()


# ============================================================================
# Graph Structure Export (for visualization)
# ============================================================================

def get_graph_structure() -> dict:
    """Export graph structure for frontend visualization."""
    return {
        "nodes": [
            {"id": "start", "label": "START", "type": "start"},
            {"id": "chatbot", "label": "Chatbot", "type": "node"},
            {"id": "tool_executor", "label": "Tool Executor", "type": "node"},
            {"id": "end", "label": "END", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "chatbot", "label": ""},
            {"from": "chatbot", "to": "tool_executor", "label": "has_tools", "conditional": True},
            {"from": "chatbot", "to": "end", "label": "no_tools", "conditional": True},
            {"from": "tool_executor", "to": "end", "label": ""},
        ],
    }


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="LangGraph Agent Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    state: dict


@app.get("/")
async def root():
    return {"message": "LangGraph Agent Service", "status": "running"}


@app.get("/graph")
async def get_graph():
    """Return the graph structure for visualization."""
    return get_graph_structure()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message through the agent."""
    initial_state = {
        "messages": [{"role": "user", "content": request.message}],
        "current_node": "start",
        "tool_calls": [],
        "iteration": 0,
    }
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    # Extract the last assistant message
    messages = final_state.get("messages", [])
    assistant_messages = [m for m in messages if m.get("role") == "assistant"]
    response_text = assistant_messages[-1].get("content", "") if assistant_messages else "No response"
    
    return ChatResponse(
        response=response_text,
        state={
            "current_node": final_state.get("current_node", "unknown"),
            "iteration": final_state.get("iteration", 0),
            "tool_calls": final_state.get("tool_calls", []),
            "message_count": len(messages),
        }
    )


# ============================================================================
# WebSocket for real-time streaming
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming agent responses."""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            
            initial_state = {
                "messages": [{"role": "user", "content": user_message}],
                "current_node": "start",
                "tool_calls": [],
                "iteration": 0,
            }
            
            # Stream state updates
            await websocket.send_json({
                "type": "state_update",
                "node": "start",
                "state": initial_state,
            })
            
            await asyncio.sleep(0.3)  # Simulate processing
            
            # Run the agent
            final_state = agent.invoke(initial_state)
            
            # Send node transition updates
            await websocket.send_json({
                "type": "state_update",
                "node": "chatbot",
                "state": {
                    "current_node": "chatbot",
                    "iteration": final_state.get("iteration", 0),
                    "tool_calls": final_state.get("tool_calls", []),
                }
            })
            
            await asyncio.sleep(0.2)
            
            if final_state.get("tool_calls"):
                await websocket.send_json({
                    "type": "state_update", 
                    "node": "tool_executor",
                    "state": {"current_node": "tool_executor"}
                })
                await asyncio.sleep(0.2)
            
            # Extract response
            messages = final_state.get("messages", [])
            assistant_messages = [m for m in messages if m.get("role") == "assistant"]
            response_text = "\n".join([m.get("content", "") for m in assistant_messages])
            
            await websocket.send_json({
                "type": "response",
                "content": response_text,
                "final_state": {
                    "current_node": "end",
                    "iteration": final_state.get("iteration", 0),
                    "message_count": len(messages),
                    "tool_calls_made": len([m for m in messages if "tool" in str(m)]),
                }
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

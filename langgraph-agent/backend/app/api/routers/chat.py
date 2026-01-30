import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.models import ChatRequest, ChatResponse
from app.services import agent

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
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
    assistant_messages = [m for m in messages if m.role == "assistant"] # FIX: Use object access
    # Use object access for content as well, with fallback if needed
    if assistant_messages:
        last_msg = assistant_messages[-1]
        response_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    else:
        response_text = "No response"
        
    return ChatResponse(
        response=response_text,
        state={
            "current_node": final_state.get("current_node", "unknown"),
            "iteration": final_state.get("iteration", 0),
            "tool_calls": final_state.get("tool_calls", []),
            "message_count": len(messages),
        }
    )

@router.websocket("/ws")
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
            assistant_messages = [m for m in messages if hasattr(m, "role") and m.role == "assistant"]
            response_text = "\n".join([m.content for m in assistant_messages])
            
            await websocket.send_json({
                "type": "response",
                "content": response_text,
                "final_state": {
                    "current_node": "end",
                    "iteration": final_state.get("iteration", 0),
                    "message_count": len(messages),
                    # Simple check for tool calls in message representation
                    "tool_calls_made": len([m for m in messages if "tool" in str(m).lower()]),
                }
            })
            
    except WebSocketDisconnect:
        print("Client disconnected")

import json
import asyncio
import uuid
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.encoders import jsonable_encoder
from app.models import ChatRequest, ChatResponse, AgentState
from app.dependencies import get_agent, AgentBase
from langchain_core.messages import HumanMessage
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

from app.infrastructure.observability import get_langfuse_handler

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, agent: AgentBase = Depends(get_agent)):
    """
    Chat endpoint that processes user messages using the injected agent.
    """
    try:
        # Initialize Langfuse Handler
        # Use session_id if provided, else thread_id
        session_id = request.thread_id or "default_session"
        langfuse_handler = get_langfuse_handler(session_id=session_id)
        
        callbacks = [langfuse_handler] if langfuse_handler else []
        config = {
            "configurable": {"thread_id": request.thread_id or "1"},
            "callbacks": callbacks
        }
        
        # Prepare state
        initial_state = AgentState(
            messages=[HumanMessage(content=request.message)],
            current_node="start",
            tool_calls=[],
            iteration=0
        )
        
        # Invoke agent
        final_state = await agent.invoke(initial_state, config=config)
        
        if langfuse_handler:
            try:
                langfuse_handler.flush()
            except Exception as e:
                logger.warning(f"Langfuse flush failed: {e}")
        
        # Extract response
        messages = final_state.get("messages", [])
        last_message = messages[-1] if messages else None
        response_text = last_message.content if last_message else "No response generated."
        
        # If last message is AIMessage, check for tool calls? 
        # But we return text.
        
        return ChatResponse(
            response=response_text,
            thread_id=request.thread_id or "1"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming agent responses."""
    await websocket.accept()
    thread_id = str(uuid.uuid4())
    
    try:
        # Initialize Langfuse for this WS session
        langfuse_handler = get_langfuse_handler(session_id=thread_id)
        callbacks = [langfuse_handler] if langfuse_handler else []

        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            user_message = message_data.get("message", "")
            #logger.info(f"User message: {user_message}")
            #logger.debug(f"[DEBUG] Handler: {langfuse_handler}, Public Key: {langfuse_handler.public_key if langfuse_handler else 'None'}")
            #print(user_message)
            initial_state = {
                "messages": [{"role": "user", "content": user_message}],
                "current_node": "start",
                "tool_calls": [],
                "iteration": 0,
            }
            
            # Stream state updates
            await websocket.send_json(jsonable_encoder({
                "type": "state_update",
                "node": "start",
                "state": initial_state,
            }))
            
            await asyncio.sleep(0.3)  # Simulate processing
            
            # Run the agent
            agent = get_agent()
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": callbacks
            }
            final_state = await agent.invoke(initial_state, config=config)
            
            if langfuse_handler:
                try:
                    langfuse_handler.flush()
                except Exception as e:
                    logger.warning(f"Langfuse flush failed: {e}")
            
            # Send node transition updates
            await websocket.send_json(jsonable_encoder({
                "type": "state_update",
                "node": "chatbot",
                "state": {
                    "current_node": "chatbot",
                    "iteration": final_state.get("iteration", 0),
                    "tool_calls": final_state.get("tool_calls", []),
                }
            }))
            
            await asyncio.sleep(0.2)
            
            if final_state.get("tool_calls"):
                await websocket.send_json({
                    "type": "state_update", 
                    "node": "tool_executor",
                    "state": {"current_node": "tool_executor"}
                })
                await asyncio.sleep(0.2)
            
            # Extract response safely
            messages = final_state.get("messages", [])
            response_text = ""
            if messages:
                last_msg = messages[-1]
                if isinstance(last_msg, dict):
                    response_text = last_msg.get("content", "")
                elif hasattr(last_msg, "content"):
                    response_text = last_msg.content
                else:
                    response_text = str(last_msg)
            
            await websocket.send_json(jsonable_encoder({
                "type": "response",
                "content": response_text,
                "final_state": {
                    "current_node": "end",
                    "iteration": final_state.get("iteration", 0),
                    "message_count": len(messages),
                    # Simple check for tool calls in message representation
                    "tool_calls_made": len([m for m in messages if "tool" in str(m).lower()]),
                }
            }))
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in websocket loop: {e}")
        try:
            await websocket.close(code=1011)
        except:
            pass

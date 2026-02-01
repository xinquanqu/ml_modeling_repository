from typing import Dict, Any, List, Literal
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from app.models import AgentState
from app.domain.chat.gateway import ChatGateway

# We will let the container wire this up, but for now we might need a way to pass it in.
# To keep signatures compatible with LangGraph (state -> state), we'll assume a global or injected instance.
# For strictly functional purity, we'd use partials, but let's stick to the simplest refactor first.

# Placeholder for the gateway instance, to be set by the container/agent
gateway_instance: ChatGateway = None 

def set_gateway(gateway: ChatGateway):
    global gateway_instance
    gateway_instance = gateway

def chatbot_node(state: AgentState, config: RunnableConfig = None) -> Dict[str, Any]:
    """Node that handles chatbot interaction."""
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    last_message = messages[-1]
    
    # FIX: Check if it's a dict or object and access content accordingly
    if isinstance(last_message, dict):
        user_input_content = last_message.get("content", "")
    elif hasattr(last_message, "content"):
        user_input_content = last_message.content
    else:
        user_input_content = ""
        
    # Extract callbacks
    callbacks = config.get("callbacks") if config else None
    
    # Use Gateway to process message
    if gateway_instance:
        response_text, tool_calls = gateway_instance.process_message(user_input_content, callbacks=callbacks)
    else:
         # Fallback if not wired (shouldn't happen in prod)
         response_text = "System Error: Gateway not initialized."
         tool_calls = []

    new_messages = [AIMessage(content=response_text)]
    
    if tool_calls:
        # If tools are called, we should ideally represent them.
        # For this simplified node, we just append the AI message.
        # In a real tool use case, we'd add tool_calls to the message.
        pass
        
    return {
        "messages": new_messages,
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

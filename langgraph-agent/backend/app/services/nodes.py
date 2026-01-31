from app.services.gateway import llm_gateway
from app.models import AgentState
from typing import Literal

def chatbot_node(state: AgentState) -> AgentState:
    """Main chatbot node - processes user input and generates responses."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    print("last_message", last_message)
    # FIX: Check if it's a dict or object and access content accordingly
    if isinstance(last_message, dict):
        user_input_content = last_message.get("content", "")
    elif hasattr(last_message, "content"):
        user_input_content = last_message.content
    else:
        user_input_content = ""
    
    # Use Gateway to process message
    response, tool_calls = llm_gateway.process_message(user_input_content)
    
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

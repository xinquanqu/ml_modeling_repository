from typing import Literal
from langchain_core.messages import HumanMessage
from app.models import AgentState

def chatbot_node(state: AgentState) -> AgentState:
    """Main chatbot node - processes user input and generates responses."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # FIX: Check if it's a dict or object and access content accordingly
    if isinstance(last_message, dict):
        user_input = last_message.get("content", "").lower()
    elif hasattr(last_message, "content"):
        user_input = last_message.content.lower()
    else:
        user_input = ""
    
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
        # Construct response with safe content access
        msg_content = ""
        if isinstance(last_message, dict):
             msg_content = last_message.get("content", "")
        elif hasattr(last_message, "content"):
             msg_content = last_message.content
             
        response = f"I received your message: '{msg_content}'. How can I assist you further?"
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

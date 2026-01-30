
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot_node(state: AgentState):
    messages = state["messages"]
    print(f"Messages type: {type(messages)}")
    print(f"First message type: {type(messages[0])}")
    
    # This should fail if messages[0] is an object
    try:
        content = messages[0].get("content")
        print(f"Content: {content}")
    except AttributeError as e:
        print(f"Caught expected error: {e}")

graph = StateGraph(AgentState)
graph.add_node("chatbot", chatbot_node)
graph.add_edge(START, "chatbot")
graph.add_edge("chatbot", END)
app = graph.compile()

# Invoke with a dict, which add_messages should convert to HumanMessage
app.invoke({"messages": [{"role": "user", "content": "hello"}]})

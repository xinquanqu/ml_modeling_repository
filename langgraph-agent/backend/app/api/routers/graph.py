from fastapi import APIRouter
from app.dependencies import get_agent

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "LangGraph Agent Service", "status": "running"}

@router.get("/graph")
async def get_graph(node_id: str = None):
    """Return the graph structure (nodes and edges)."""
    agent = get_agent()
    return agent.get_graph_structure(subgraph_id=node_id)

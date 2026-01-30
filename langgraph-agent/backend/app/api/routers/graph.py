from fastapi import APIRouter
from app.services import get_graph_structure

router = APIRouter()

@router.get("/")
async def root():
    return {"message": "LangGraph Agent Service", "status": "running"}

@router.get("/graph")
async def get_graph():
    """Return the graph structure for visualization."""
    return get_graph_structure()

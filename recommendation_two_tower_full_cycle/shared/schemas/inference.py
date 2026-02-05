"""
Inference-related Pydantic schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request for model prediction."""
    user_id: str
    item_ids: Optional[List[str]] = None  # None for recommendations
    num_recommendations: int = 10
    include_scores: bool = True
    model_version: Optional[str] = None  # None for latest


class PredictionResponse(BaseModel):
    """Response for model prediction."""
    user_id: str
    predictions: List[Dict[str, Any]]  # [{item_id, score}]
    model_version: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelInfo(BaseModel):
    """Model metadata."""
    model_id: str
    model_type: str
    version: str
    mlflow_run_id: Optional[str] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    created_at: datetime
    is_active: bool = False

"""
Feature-related Pydantic schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Feature(BaseModel):
    """Single feature definition."""
    name: str
    dtype: str = "float32"
    value: Any
    version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureSet(BaseModel):
    """Collection of features for an entity."""
    entity_id: str
    entity_type: str  # "user" or "item"
    features: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FeatureRequest(BaseModel):
    """Request for feature retrieval."""
    entity_ids: List[str]
    entity_type: str
    feature_names: Optional[List[str]] = None  # None means all features
    include_metadata: bool = False

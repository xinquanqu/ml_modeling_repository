"""
Data-related Pydantic schemas.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DataRecord(BaseModel):
    """Single data record."""
    id: Optional[str] = None
    user_id: str
    item_id: str
    features: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class DataBatch(BaseModel):
    """Batch of data records."""
    records: List[DataRecord]
    source: str = "api"
    batch_id: Optional[str] = None


class UploadResponse(BaseModel):
    """Response for data upload operations."""
    success: bool
    records_processed: int
    batch_id: str
    errors: List[str] = Field(default_factory=list)
    message: str = ""

"""
Training-related Pydantic schemas.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TrainingStatus(str, Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingConfig(BaseModel):
    """Configuration for a training job."""
    model_type: str = "two_tower"
    
    # Model architecture
    user_embedding_dim: int = 64
    item_embedding_dim: int = 64
    hidden_dims: List[int] = Field(default_factory=lambda: [128, 64])
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 10
    
    # Data
    train_split: float = 0.8
    validation_split: float = 0.1
    
    # MLflow
    experiment_name: str = "recommendation_model"
    run_name: Optional[str] = None


class TrainingJob(BaseModel):
    """Training job representation."""
    job_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    
    # Progress
    current_epoch: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    
    # MLflow
    mlflow_run_id: Optional[str] = None
    model_uri: Optional[str] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Errors
    error_message: Optional[str] = None

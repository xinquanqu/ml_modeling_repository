"""
Shared Pydantic schemas for ML Platform.
"""

from .data import DataRecord, DataBatch, UploadResponse
from .features import Feature, FeatureSet, FeatureRequest
from .training import TrainingJob, TrainingConfig, TrainingStatus
from .inference import PredictionRequest, PredictionResponse

__all__ = [
    "DataRecord",
    "DataBatch",
    "UploadResponse",
    "Feature",
    "FeatureSet",
    "FeatureRequest",
    "TrainingJob",
    "TrainingConfig",
    "TrainingStatus",
    "PredictionRequest",
    "PredictionResponse",
]

"""
API routes for model serving.
"""

import time
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from prometheus_client import Counter

router = APIRouter()

# Metrics
PREDICTION_COUNTER = Counter(
    'predictions_total',
    'Total predictions made',
    ['model_version', 'status']
)


class PredictionRequest(BaseModel):
    """Request for model prediction."""
    user_id: int
    item_ids: Optional[List[int]] = None
    num_recommendations: int = 10
    exclude_items: Optional[List[int]] = None
    model_version: Optional[str] = None
    include_embeddings: bool = False


class PredictionResponse(BaseModel):
    """Response for model prediction."""
    request_id: str
    user_id: int
    predictions: List[Dict[str, Any]]
    model_version: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    user_ids: List[int]
    item_ids: List[int]
    model_version: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    request_id: str
    scores: List[float]
    model_version: str
    latency_ms: float


class KServeRequest(BaseModel):
    """KServe-compatible inference request."""
    instances: List[Dict[str, Any]]


class KServeResponse(BaseModel):
    """KServe-compatible inference response."""
    predictions: List[Any]
    model_name: str = "recommendation-model"
    model_version: str


# =============================================================================
# Standard Inference Endpoints
# =============================================================================

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, http_request: Request):
    """
    Get predictions/recommendations for a user.
    
    If item_ids is provided, returns scores for those items.
    Otherwise, returns top-K recommendations.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    model_manager = http_request.app.state.model_manager
    predictor = model_manager.get_predictor(request.model_version)
    
    if not predictor:
        raise HTTPException(503, "No model available for serving")
    
    try:
        if request.item_ids:
            # Score specific items
            results = predictor.predict(request.user_id, request.item_ids)
        else:
            # Get recommendations
            results = predictor.recommend(
                request.user_id,
                request.num_recommendations,
                exclude_items=request.exclude_items
            )
        
        predictions = []
        for item_id, score in results:
            pred = {"item_id": item_id, "score": round(score, 4)}
            
            if request.include_embeddings:
                pred["item_embedding"] = predictor.get_item_embedding(item_id).tolist()
            
            predictions.append(pred)
        
        latency_ms = (time.time() - start_time) * 1000
        
        PREDICTION_COUNTER.labels(model_version=predictor.version, status="success").inc()
        
        return PredictionResponse(
            request_id=request_id,
            user_id=request.user_id,
            predictions=predictions,
            model_version=predictor.version,
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        PREDICTION_COUNTER.labels(model_version=predictor.version, status="error").inc()
        raise HTTPException(500, f"Prediction failed: {str(e)}")


@router.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest, http_request: Request):
    """
    Batch prediction for user-item pairs.
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    if len(request.user_ids) != len(request.item_ids):
        raise HTTPException(400, "user_ids and item_ids must have same length")
    
    model_manager = http_request.app.state.model_manager
    predictor = model_manager.get_predictor(request.model_version)
    
    if not predictor:
        raise HTTPException(503, "No model available for serving")
    
    try:
        scores = predictor.batch_predict(request.user_ids, request.item_ids)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            request_id=request_id,
            scores=[round(float(s), 4) for s in scores],
            model_version=predictor.version,
            latency_ms=round(latency_ms, 2)
        )
        
    except Exception as e:
        raise HTTPException(500, f"Batch prediction failed: {str(e)}")


# =============================================================================
# KServe-Compatible Endpoints
# =============================================================================

@router.post("/v1/models/recommendation:predict", response_model=KServeResponse)
@router.post("/v2/models/recommendation/infer", response_model=KServeResponse)
async def kserve_predict(request: KServeRequest, http_request: Request):
    """
    KServe-compatible inference endpoint.
    
    Expected instance format:
    {"user_id": int, "item_ids": list[int]} or
    {"user_id": int, "num_recommendations": int}
    """
    model_manager = http_request.app.state.model_manager
    predictor = model_manager.get_predictor()
    
    if not predictor:
        raise HTTPException(503, "No model available")
    
    predictions = []
    
    for instance in request.instances:
        user_id = instance.get("user_id")
        item_ids = instance.get("item_ids")
        num_recs = instance.get("num_recommendations", 10)
        
        if user_id is None:
            predictions.append({"error": "user_id required"})
            continue
        
        try:
            if item_ids:
                results = predictor.predict(user_id, item_ids)
            else:
                results = predictor.recommend(user_id, num_recs)
            
            predictions.append({
                "user_id": user_id,
                "items": [{"id": i, "score": s} for i, s in results]
            })
        except Exception as e:
            predictions.append({"error": str(e)})
    
    return KServeResponse(
        predictions=predictions,
        model_version=predictor.version
    )


# =============================================================================
# Model Management Endpoints
# =============================================================================

@router.get("/models")
async def list_models(http_request: Request):
    """List all loaded models."""
    model_manager = http_request.app.state.model_manager
    return {"models": model_manager.list_models()}


@router.get("/models/{version}")
async def get_model_info(version: str, http_request: Request):
    """Get information about a specific model."""
    model_manager = http_request.app.state.model_manager
    predictor = model_manager.get_predictor(version)
    
    if not predictor:
        raise HTTPException(404, f"Model version {version} not found")
    
    return predictor.get_model_info()


@router.post("/models/{version}/activate")
async def activate_model(version: str, http_request: Request):
    """Set a model as the active version."""
    model_manager = http_request.app.state.model_manager
    
    if model_manager.set_active_model(version):
        return {"success": True, "active_model": version}
    
    raise HTTPException(404, f"Model version {version} not found")


@router.post("/models/load")
async def load_model(
    model_uri: str,
    version: str,
    http_request: Request
):
    """Load a model from MLflow."""
    model_manager = http_request.app.state.model_manager
    
    success = await model_manager.load_model(model_uri, version)
    
    if success:
        return {"success": True, "version": version}
    
    raise HTTPException(500, "Failed to load model")


@router.delete("/models/{version}")
async def unload_model(version: str, http_request: Request):
    """Unload a model from memory."""
    model_manager = http_request.app.state.model_manager
    
    if await model_manager.unload_model(version):
        return {"success": True, "unloaded": version}
    
    raise HTTPException(404, f"Model version {version} not found")


# =============================================================================
# Embeddings Endpoints
# =============================================================================

@router.get("/embeddings/user/{user_id}")
async def get_user_embedding(user_id: int, http_request: Request):
    """Get the embedding for a user."""
    model_manager = http_request.app.state.model_manager
    predictor = model_manager.get_predictor()
    
    if not predictor:
        raise HTTPException(503, "No model available")
    
    try:
        embedding = predictor.get_user_embedding(user_id)
        return {
            "user_id": user_id,
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get embedding: {str(e)}")


@router.get("/embeddings/item/{item_id}")
async def get_item_embedding(item_id: int, http_request: Request):
    """Get the embedding for an item."""
    model_manager = http_request.app.state.model_manager
    predictor = model_manager.get_predictor()
    
    if not predictor:
        raise HTTPException(503, "No model available")
    
    try:
        embedding = predictor.get_item_embedding(item_id)
        return {
            "item_id": item_id,
            "embedding": embedding.tolist(),
            "dimension": len(embedding)
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to get embedding: {str(e)}")

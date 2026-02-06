"""
FastAPI wrapper for Feast Feature Store.

Provides REST API endpoints for feature operations.
"""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

import feast
from feast import FeatureStore


# Global feature store instance
_store: Optional[FeatureStore] = None


def get_store() -> FeatureStore:
    """Get or initialize the Feast store."""
    global _store
    if _store is None:
        repo_path = os.getenv("FEAST_REPO_PATH", "/app/feature_repo")
        _store = FeatureStore(repo_path=repo_path)
    return _store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Feast on startup."""
    global _store
    repo_path = os.getenv("FEAST_REPO_PATH", "/app/feature_repo")
    _store = FeatureStore(repo_path=repo_path)
    print(f"Feast initialized from {repo_path}")
    yield
    print("Feast shutdown")


app = FastAPI(
    title="Feature Store Service",
    description="Feast-powered feature store for ML Platform",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
FEATURE_OPS = Counter(
    'feast_operations_total',
    'Feast operations',
    ['operation', 'status']
)


# =============================================================================
# Request/Response Models
# =============================================================================

class FeatureRequest(BaseModel):
    """Request for online features."""
    entity_rows: List[Dict[str, Any]]
    features: List[str]


class MaterializeRequest(BaseModel):
    """Request to materialize features."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    feature_views: Optional[List[str]] = None


class HistoricalFeaturesRequest(BaseModel):
    """Request for historical features (training data)."""
    entity_df: List[Dict[str, Any]]
    features: List[str]


# =============================================================================
# Health & Metrics
# =============================================================================

@app.get("/health")
async def health():
    """Health check."""
    try:
        store = get_store()
        return {"status": "healthy", "feast_version": feast.__version__}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =============================================================================
# Feature Retrieval
# =============================================================================

@app.post("/api/v1/features/online")
async def get_online_features(request: FeatureRequest):
    """
    Get features from the online store.
    
    Example:
        {
            "entity_rows": [{"user_id": "user_001"}],
            "features": ["user_features:interaction_count", "user_features:avg_rating"]
        }
    """
    try:
        store = get_store()
        
        result = store.get_online_features(
            entity_rows=request.entity_rows,
            features=request.features,
        )
        
        FEATURE_OPS.labels(operation="get_online", status="success").inc()
        
        return {
            "features": result.to_dict()
        }
        
    except Exception as e:
        FEATURE_OPS.labels(operation="get_online", status="error").inc()
        raise HTTPException(500, f"Failed to get features: {str(e)}")


@app.get("/api/v1/user/{user_id}")
async def get_user_features(user_id: str):
    """Get features for a specific user."""
    try:
        store = get_store()
        
        result = store.get_online_features(
            entity_rows=[{"user_id": user_id}],
            features=[
                "user_features:interaction_count",
                "user_features:avg_rating",
                "user_features:unique_items",
                "user_features:last_active_days",
            ],
        )
        
        features = result.to_dict()
        
        return {
            "user_id": user_id,
            "features": {
                k: v[0] for k, v in features.items() if k != "user_id"
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to get user features: {str(e)}")


@app.get("/api/v1/item/{item_id}")
async def get_item_features(item_id: str):
    """Get features for a specific item."""
    try:
        store = get_store()
        
        result = store.get_online_features(
            entity_rows=[{"item_id": item_id}],
            features=[
                "item_features:interaction_count",
                "item_features:avg_rating",
                "item_features:unique_users",
                "item_features:last_active_days",
            ],
        )
        
        features = result.to_dict()
        
        return {
            "item_id": item_id,
            "features": {
                k: v[0] for k, v in features.items() if k != "item_id"
            }
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to get item features: {str(e)}")


# =============================================================================
# Feature Materialization
# =============================================================================

@app.post("/api/v1/materialize")
async def materialize_features(request: Optional[MaterializeRequest] = None):
    """
    Materialize features to the online store.
    
    This pushes computed features from the offline store to Redis.
    """
    try:
        store = get_store()
        
        end_date = datetime.utcnow()
        start_date = datetime(2020, 1, 1)
        feature_views = None
        
        if request:
            end_date = request.end_date or end_date
            start_date = request.start_date or start_date
            feature_views = request.feature_views
        
        if feature_views:
            store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=feature_views,
            )
        else:
            store.materialize(
                start_date=start_date,
                end_date=end_date,
            )
        
        FEATURE_OPS.labels(operation="materialize", status="success").inc()
        
        return {
            "success": True,
            "message": f"Materialized features from {start_date} to {end_date}"
        }
        
    except Exception as e:
        FEATURE_OPS.labels(operation="materialize", status="error").inc()
        raise HTTPException(500, f"Materialization failed: {str(e)}")


# =============================================================================
# Feature Store Management
# =============================================================================

@app.post("/api/v1/apply")
async def apply_feature_definitions():
    """Apply feature definitions to the registry."""
    try:
        store = get_store()
        store.apply([])  # Apply from repo
        
        return {"success": True, "message": "Feature definitions applied"}
        
    except Exception as e:
        raise HTTPException(500, f"Apply failed: {str(e)}")


@app.get("/api/v1/registry")
async def list_registry():
    """List all registered features."""
    try:
        store = get_store()
        
        entities = [e.name for e in store.list_entities()]
        feature_views = [fv.name for fv in store.list_feature_views()]
        
        return {
            "entities": entities,
            "feature_views": feature_views
        }
        
    except Exception as e:
        raise HTTPException(500, f"Failed to list registry: {str(e)}")


@app.get("/api/v1/stats")
async def get_stats():
    """Get feature store statistics."""
    try:
        store = get_store()
        
        return {
            "feast_version": feast.__version__,
            "entities": len(store.list_entities()),
            "feature_views": len(store.list_feature_views()),
            "online_store": "redis",
            "offline_store": "postgres"
        }
        
    except Exception as e:
        raise HTTPException(500, str(e))

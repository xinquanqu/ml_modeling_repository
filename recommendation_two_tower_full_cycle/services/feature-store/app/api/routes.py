"""
API routes for feature store.
"""

import json
from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from prometheus_client import Counter

from app.storage.postgres import get_db_connection
from app.storage.redis_cache import get_redis
from app.features.transformations import compute_user_features, compute_item_features

router = APIRouter()

# Metrics
FEATURE_OPS = Counter(
    'feature_operations_total',
    'Feature store operations',
    ['operation', 'entity_type', 'status']
)


class Feature(BaseModel):
    """Feature definition."""
    name: str
    dtype: str = "float32"
    value: Any
    version: str = "1.0"


class FeatureSet(BaseModel):
    """Collection of features."""
    entity_id: str
    entity_type: str
    features: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FeatureRequest(BaseModel):
    """Request for feature retrieval."""
    entity_ids: List[str]
    entity_type: str  # "user" or "item"
    feature_names: Optional[List[str]] = None


class FeatureResponse(BaseModel):
    """Response containing features."""
    entity_id: str
    entity_type: str
    features: Dict[str, Any]
    source: str = "cache"  # "cache" or "computed"


class FeatureRegistration(BaseModel):
    """Register a new feature."""
    name: str
    entity_type: str
    dtype: str = "float32"
    description: Optional[str] = None


# =============================================================================
# Feature Retrieval Endpoints
# =============================================================================

@router.post("/get", response_model=List[FeatureResponse])
async def get_features(request: FeatureRequest):
    """
    Get features for a list of entities.
    
    First checks Redis cache, then falls back to PostgreSQL.
    """
    results = []
    redis = await get_redis()
    
    for entity_id in request.entity_ids:
        cache_key = f"features:{request.entity_type}:{entity_id}"
        
        # Try Redis first
        cached = await redis.get(cache_key)
        if cached:
            features = json.loads(cached)
            if request.feature_names:
                features = {k: v for k, v in features.items() if k in request.feature_names}
            
            FEATURE_OPS.labels(operation="get", entity_type=request.entity_type, status="cache_hit").inc()
            results.append(FeatureResponse(
                entity_id=entity_id,
                entity_type=request.entity_type,
                features=features,
                source="cache"
            ))
            continue
        
        # Fall back to PostgreSQL
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                table = "user_features" if request.entity_type == "user" else "item_features"
                id_col = "user_id" if request.entity_type == "user" else "item_id"
                
                cur.execute(f"SELECT features FROM {table} WHERE {id_col} = %s", (entity_id,))
                row = cur.fetchone()
                
                if row:
                    features = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                    if request.feature_names:
                        features = {k: v for k, v in features.items() if k in request.feature_names}
                    
                    # Cache for next time
                    await redis.setex(cache_key, 3600, json.dumps(features))
                    
                    FEATURE_OPS.labels(operation="get", entity_type=request.entity_type, status="db_hit").inc()
                    results.append(FeatureResponse(
                        entity_id=entity_id,
                        entity_type=request.entity_type,
                        features=features,
                        source="computed"
                    ))
                else:
                    FEATURE_OPS.labels(operation="get", entity_type=request.entity_type, status="miss").inc()
                    results.append(FeatureResponse(
                        entity_id=entity_id,
                        entity_type=request.entity_type,
                        features={},
                        source="computed"
                    ))
        finally:
            conn.close()
    
    return results


@router.get("/user/{user_id}")
async def get_user_features(user_id: str, feature_names: Optional[str] = None):
    """Get features for a specific user."""
    names = feature_names.split(",") if feature_names else None
    request = FeatureRequest(entity_ids=[user_id], entity_type="user", feature_names=names)
    results = await get_features(request)
    return results[0] if results else HTTPException(404, "User not found")


@router.get("/item/{item_id}")
async def get_item_features(item_id: str, feature_names: Optional[str] = None):
    """Get features for a specific item."""
    names = feature_names.split(",") if feature_names else None
    request = FeatureRequest(entity_ids=[item_id], entity_type="item", feature_names=names)
    results = await get_features(request)
    return results[0] if results else HTTPException(404, "Item not found")


# =============================================================================
# Feature Storage Endpoints
# =============================================================================

@router.post("/set")
async def set_features(feature_set: FeatureSet):
    """
    Store features for an entity.
    
    Updates both PostgreSQL (persistent) and Redis (cache).
    """
    conn = get_db_connection()
    redis = await get_redis()
    
    try:
        with conn.cursor() as cur:
            table = "user_features" if feature_set.entity_type == "user" else "item_features"
            id_col = "user_id" if feature_set.entity_type == "user" else "item_id"
            
            cur.execute(f"""
                INSERT INTO {table} ({id_col}, features, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT ({id_col}) DO UPDATE
                SET features = EXCLUDED.features,
                    updated_at = NOW()
            """, (feature_set.entity_id, json.dumps(feature_set.features)))
            
            conn.commit()
        
        # Update cache
        cache_key = f"features:{feature_set.entity_type}:{feature_set.entity_id}"
        await redis.setex(cache_key, 3600, json.dumps(feature_set.features))
        
        FEATURE_OPS.labels(operation="set", entity_type=feature_set.entity_type, status="success").inc()
        
        return {"success": True, "entity_id": feature_set.entity_id}
        
    except Exception as e:
        FEATURE_OPS.labels(operation="set", entity_type=feature_set.entity_type, status="error").inc()
        raise HTTPException(500, str(e))
    finally:
        conn.close()


@router.post("/batch-set")
async def batch_set_features(feature_sets: List[FeatureSet]):
    """Store features for multiple entities."""
    results = []
    for fs in feature_sets:
        try:
            result = await set_features(fs)
            results.append(result)
        except Exception as e:
            results.append({"success": False, "entity_id": fs.entity_id, "error": str(e)})
    
    return {"results": results}


# =============================================================================
# Feature Computation Endpoints
# =============================================================================

@router.post("/compute/user/{user_id}")
async def compute_and_store_user_features(user_id: str):
    """
    Compute features for a user from raw data.
    """
    try:
        features = await compute_user_features(user_id)
        
        feature_set = FeatureSet(
            entity_id=user_id,
            entity_type="user",
            features=features
        )
        await set_features(feature_set)
        
        return {"success": True, "user_id": user_id, "features": features}
        
    except Exception as e:
        raise HTTPException(500, f"Feature computation failed: {str(e)}")


@router.post("/compute/item/{item_id}")
async def compute_and_store_item_features(item_id: str):
    """
    Compute features for an item from raw data.
    """
    try:
        features = await compute_item_features(item_id)
        
        feature_set = FeatureSet(
            entity_id=item_id,
            entity_type="item",
            features=features
        )
        await set_features(feature_set)
        
        return {"success": True, "item_id": item_id, "features": features}
        
    except Exception as e:
        raise HTTPException(500, f"Feature computation failed: {str(e)}")


# =============================================================================
# Feature Registry Endpoints
# =============================================================================

@router.post("/registry/register")
async def register_feature(registration: FeatureRegistration):
    """Register a new feature in the registry."""
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feature_registry (name, entity_type, dtype, description)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                SET entity_type = EXCLUDED.entity_type,
                    dtype = EXCLUDED.dtype,
                    description = EXCLUDED.description,
                    updated_at = NOW()
                RETURNING id
            """, (registration.name, registration.entity_type, registration.dtype, registration.description))
            
            feature_id = cur.fetchone()[0]
            conn.commit()
            
        return {"success": True, "feature_id": feature_id, "name": registration.name}
        
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        conn.close()


@router.get("/registry/list")
async def list_features(entity_type: Optional[str] = None):
    """List all registered features."""
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            if entity_type:
                cur.execute("""
                    SELECT id, name, entity_type, dtype, description, version, is_active
                    FROM feature_registry
                    WHERE entity_type = %s AND is_active = true
                """, (entity_type,))
            else:
                cur.execute("""
                    SELECT id, name, entity_type, dtype, description, version, is_active
                    FROM feature_registry
                    WHERE is_active = true
                """)
            
            rows = cur.fetchall()
            
        features = []
        for row in rows:
            features.append({
                "id": row[0],
                "name": row[1],
                "entity_type": row[2],
                "dtype": row[3],
                "description": row[4],
                "version": row[5],
                "is_active": row[6]
            })
        
        return {"features": features}
        
    finally:
        conn.close()


@router.get("/stats")
async def get_feature_stats():
    """Get feature store statistics."""
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM user_features")
            user_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM item_features")
            item_count = cur.fetchone()[0]
            
            cur.execute("SELECT COUNT(*) FROM feature_registry WHERE is_active = true")
            feature_count = cur.fetchone()[0]
            
        return {
            "user_features_count": user_count,
            "item_features_count": item_count,
            "registered_features": feature_count
        }
        
    finally:
        conn.close()

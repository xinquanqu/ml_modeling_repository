"""
Feature Store Service - Main FastAPI Application

Provides feature computation, storage, and retrieval for ML models.
Supports both online (Redis) and offline (PostgreSQL) feature stores.
"""

import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.api.routes import router
from app.storage.postgres import init_db_pool
from app.storage.redis_cache import init_redis

# Prometheus metrics
FEATURE_REQUESTS = Counter(
    'feature_store_requests_total',
    'Total feature requests',
    ['operation', 'entity_type', 'status']
)
FEATURE_LATENCY = Histogram(
    'feature_store_latency_seconds',
    'Feature retrieval latency',
    ['operation']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    print("Starting Feature Store Service...")
    init_db_pool()
    await init_redis()
    yield
    print("Shutting down Feature Store Service...")


app = FastAPI(
    title="Feature Store Service",
    description="Feature computation and storage for ML models",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["features"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.storage.postgres import get_db_connection
    from app.storage.redis_cache import get_redis
    
    status = {"status": "healthy", "service": "feature-store"}
    
    try:
        # Check PostgreSQL
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        status["postgres"] = "connected"
    except Exception as e:
        status["postgres"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    try:
        # Check Redis
        redis = await get_redis()
        await redis.ping()
        status["redis"] = "connected"
    except Exception as e:
        status["redis"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    return status


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "feature-store",
        "version": "1.0.0",
        "docs": "/docs"
    }

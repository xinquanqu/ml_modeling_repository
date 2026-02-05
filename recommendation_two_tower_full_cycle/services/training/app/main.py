"""
Training Service - Main FastAPI Application

Manages ML training jobs with PyTorch and MLflow integration.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.api.routes import router
from app.utils.database import init_db_pool

# Prometheus metrics
TRAINING_JOBS = Counter(
    'training_jobs_total',
    'Total training jobs',
    ['status']
)
TRAINING_DURATION = Histogram(
    'training_duration_seconds',
    'Training job duration',
    ['model_type']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    print("Starting Training Service...")
    init_db_pool()
    
    # Set MLflow tracking URI
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    
    yield
    print("Shutting down Training Service...")


app = FastAPI(
    title="Training Service",
    description="ML model training with PyTorch and MLflow",
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
app.include_router(router, prefix="/api/v1", tags=["training"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    from app.utils.database import get_db_connection
    
    status = {"status": "healthy", "service": "training"}
    
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        status["postgres"] = "connected"
    except Exception as e:
        status["postgres"] = f"error: {str(e)}"
        status["status"] = "degraded"
    
    try:
        import mlflow
        mlflow.search_experiments(max_results=1)
        status["mlflow"] = "connected"
    except Exception as e:
        status["mlflow"] = f"error: {str(e)}"
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
        "service": "training",
        "version": "1.0.0",
        "docs": "/docs"
    }

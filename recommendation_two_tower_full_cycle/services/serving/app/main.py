"""
Model Serving Service - Main FastAPI Application

Provides model inference with KServe-compatible interface.
"""

import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.api.routes import router
from app.inference.model_manager import ModelManager

# Prometheus metrics
PREDICTION_REQUESTS = Counter(
    'serving_prediction_requests_total',
    'Total prediction requests',
    ['model_version', 'status']
)
PREDICTION_LATENCY = Histogram(
    'serving_prediction_latency_seconds',
    'Prediction latency',
    ['model_version'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Global model manager
model_manager: Optional[ModelManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global model_manager
    
    print("Starting Model Serving Service...")
    
    # Initialize model manager
    model_manager = ModelManager()
    await model_manager.initialize()
    
    # Store in app state
    app.state.model_manager = model_manager
    
    yield
    
    print("Shutting down Model Serving Service...")


app = FastAPI(
    title="Model Serving Service",
    description="ML model inference with KServe-compatible interface",
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
app.include_router(router, prefix="/api/v1", tags=["inference"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    global model_manager
    
    status = {
        "status": "healthy",
        "service": "serving",
        "models_loaded": 0
    }
    
    if model_manager:
        status["models_loaded"] = model_manager.get_loaded_models_count()
        if status["models_loaded"] == 0:
            status["status"] = "degraded"
            status["message"] = "No models loaded"
    
    return status


@app.get("/health/ready")
async def readiness_check():
    """Readiness check for Kubernetes."""
    global model_manager
    
    if model_manager and model_manager.get_loaded_models_count() > 0:
        return {"ready": True}
    
    return {"ready": False, "reason": "No models loaded"}


@app.get("/health/live")
async def liveness_check():
    """Liveness check for Kubernetes."""
    return {"live": True}


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
        "service": "serving",
        "version": "1.0.0",
        "docs": "/docs"
    }

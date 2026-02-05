"""
Data Ingestion Service - Main FastAPI Application

Handles file uploads (CSV, JSON, Parquet) and streaming data ingestion
into the ML platform's data warehouse.
"""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import psycopg2
from psycopg2.extras import execute_values
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from app.api.routes import router
from app.utils.database import get_db_connection, init_db_pool
from app.utils.metrics import setup_metrics

# Prometheus metrics
REQUESTS_TOTAL = Counter(
    'data_ingestion_requests_total',
    'Total number of ingestion requests',
    ['method', 'endpoint', 'status']
)
INGESTION_LATENCY = Histogram(
    'data_ingestion_latency_seconds',
    'Latency of data ingestion operations',
    ['operation']
)
RECORDS_INGESTED = Counter(
    'data_ingestion_records_total',
    'Total number of records ingested',
    ['source']
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    print("Starting Data Ingestion Service...")
    init_db_pool()
    yield
    # Shutdown
    print("Shutting down Data Ingestion Service...")


app = FastAPI(
    title="Data Ingestion Service",
    description="Upload and stream data into the ML platform",
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
app.include_router(router, prefix="/api/v1", tags=["ingestion"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return {"status": "healthy", "service": "data-ingestion"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


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
        "service": "data-ingestion",
        "version": "1.0.0",
        "docs": "/docs"
    }

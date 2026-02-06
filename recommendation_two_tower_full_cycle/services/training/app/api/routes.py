"""
API routes for training service.
"""

import uuid
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from prometheus_client import Counter

from app.trainers.trainer import run_training_async
from app.registry import ModelRegistry, initialize_registry
from app.utils.database import get_db_connection, return_db_connection

router = APIRouter()

# Metrics
TRAINING_JOBS_COUNTER = Counter(
    'training_jobs_submitted',
    'Training jobs submitted',
    ['status']
)


class TrainingConfig(BaseModel):
    """Configuration for a training job."""
    # Model selection (key change: model is now configurable)
    model_name: str = "two_tower"  # two_tower, matrix_factorization, ncf
    
    # Model architecture (overrides YAML config)
    user_embedding_dim: Optional[int] = None
    item_embedding_dim: Optional[int] = None
    hidden_dims: Optional[List[int]] = None
    dropout: Optional[float] = None
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 10
    train_split: float = 0.8
    
    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.001
    
    # Evaluation
    metrics: List[str] = Field(default_factory=lambda: ["auc", "precision", "recall"])
    
    # MLflow
    experiment_name: str = "recommendation_model"
    run_name: Optional[str] = None


class TrainingJobRequest(BaseModel):
    """Request to start a training job."""
    config: TrainingConfig = Field(default_factory=TrainingConfig)


class TrainingJobResponse(BaseModel):
    """Response for training job creation."""
    job_id: str
    status: str
    message: str
    model_name: str


class TrainingJobStatus(BaseModel):
    """Status of a training job."""
    job_id: str
    status: str
    current_epoch: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    metrics: Dict[str, Any]
    mlflow_run_id: Optional[str]
    model_uri: Optional[str]
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]


@router.get("/models")
async def list_available_models():
    """
    List all available models in the registry.
    """
    try:
        initialize_registry(config_dir="model_configs")
        models = ModelRegistry.list_models()
        configs = ModelRegistry.list_configs()
        
        return {
            "models": models,
            "configs": configs,
            "available_models": [
                {
                    "name": name,
                    "has_config": name in configs
                }
                for name in models
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list models: {str(e)}")


@router.get("/models/{model_name}/config")
async def get_model_config(model_name: str):
    """
    Get the configuration for a model.
    """
    try:
        initialize_registry(config_dir="model_configs")
        config = ModelRegistry.get_config(model_name)
        
        if not config:
            raise HTTPException(404, f"Config for model '{model_name}' not found")
        
        return config
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to get config: {str(e)}")


@router.post("/jobs", response_model=TrainingJobResponse)
async def create_training_job(
    request: TrainingJobRequest,
    background_tasks: BackgroundTasks
):
    """
    Create and start a new training job.
    """
    job_id = str(uuid.uuid4())
    config = request.config.model_dump(exclude_none=True)
    
    # Build early stopping config
    config["early_stopping"] = {
        "enabled": config.pop("early_stopping_enabled", True),
        "patience": config.pop("early_stopping_patience", 5),
        "min_delta": config.pop("early_stopping_min_delta", 0.001)
    }
    
    # Insert job into database
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO training_jobs (job_id, config, status)
                VALUES (%s, %s, 'pending')
                """,
                (job_id, json.dumps(config))
            )
            conn.commit()
    finally:
        return_db_connection(conn)
    
    # Start training in background
    background_tasks.add_task(run_training_async, job_id, config)
    
    TRAINING_JOBS_COUNTER.labels(status="submitted").inc()
    
    return TrainingJobResponse(
        job_id=job_id,
        status="pending",
        message="Training job submitted successfully",
        model_name=config.get("model_name", "two_tower")
    )


@router.get("/jobs/{job_id}", response_model=TrainingJobStatus)
async def get_training_job(job_id: str):
    """
    Get the status of a training job.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT job_id, status, current_epoch, train_loss, val_loss,
                       metrics, mlflow_run_id, model_uri, created_at,
                       started_at, completed_at, error_message
                FROM training_jobs
                WHERE job_id = %s
                """,
                (job_id,)
            )
            row = cur.fetchone()
            
            if not row:
                raise HTTPException(404, "Job not found")
            
            return TrainingJobStatus(
                job_id=row[0],
                status=row[1],
                current_epoch=row[2] or 0,
                train_loss=row[3],
                val_loss=row[4],
                metrics=row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
                mlflow_run_id=row[6],
                model_uri=row[7],
                created_at=row[8],
                started_at=row[9],
                completed_at=row[10],
                error_message=row[11]
            )
    finally:
        return_db_connection(conn)


@router.get("/jobs", response_model=List[TrainingJobStatus])
async def list_training_jobs(
    status: Optional[str] = None,
    limit: int = 20
):
    """
    List training jobs.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if status:
                cur.execute(
                    """
                    SELECT job_id, status, current_epoch, train_loss, val_loss,
                           metrics, mlflow_run_id, model_uri, created_at,
                           started_at, completed_at, error_message
                    FROM training_jobs
                    WHERE status = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (status, limit)
                )
            else:
                cur.execute(
                    """
                    SELECT job_id, status, current_epoch, train_loss, val_loss,
                           metrics, mlflow_run_id, model_uri, created_at,
                           started_at, completed_at, error_message
                    FROM training_jobs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,)
                )
            
            jobs = []
            for row in cur.fetchall():
                jobs.append(TrainingJobStatus(
                    job_id=row[0],
                    status=row[1],
                    current_epoch=row[2] or 0,
                    train_loss=row[3],
                    val_loss=row[4],
                    metrics=row[5] if isinstance(row[5], dict) else json.loads(row[5] or "{}"),
                    mlflow_run_id=row[6],
                    model_uri=row[7],
                    created_at=row[8],
                    started_at=row[9],
                    completed_at=row[10],
                    error_message=row[11]
                ))
            
            return jobs
    finally:
        return_db_connection(conn)


@router.delete("/jobs/{job_id}")
async def cancel_training_job(job_id: str):
    """
    Cancel a training job.
    """
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE training_jobs
                SET status = 'cancelled', completed_at = NOW()
                WHERE job_id = %s AND status IN ('pending', 'running')
                RETURNING job_id
                """,
                (job_id,)
            )
            
            if not cur.fetchone():
                raise HTTPException(404, "Job not found or already completed")
            
            conn.commit()
            
        return {"success": True, "job_id": job_id, "status": "cancelled"}
    finally:
        return_db_connection(conn)


@router.get("/experiments")
async def list_experiments():
    """
    List MLflow experiments.
    """
    import mlflow
    
    try:
        experiments = mlflow.search_experiments()
        return {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "artifact_location": exp.artifact_location,
                    "lifecycle_stage": exp.lifecycle_stage
                }
                for exp in experiments
            ]
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to list experiments: {str(e)}")


@router.get("/experiments/{experiment_name}/runs")
async def list_experiment_runs(experiment_name: str, limit: int = 20):
    """
    List runs for an experiment.
    """
    import mlflow
    
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if not experiment:
            raise HTTPException(404, f"Experiment '{experiment_name}' not found")
        
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=limit,
            order_by=["start_time DESC"]
        )
        
        return {
            "experiment": experiment_name,
            "runs": runs.to_dict(orient="records") if len(runs) > 0 else []
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to list runs: {str(e)}")

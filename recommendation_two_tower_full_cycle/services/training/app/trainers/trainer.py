"""
Generic Training Engine.

Configuration-driven trainer that works with any registered model.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import mlflow
import mlflow.pytorch

from app.models.base import BaseRecommendationModel
from app.registry import ModelRegistry, ModelFactory, initialize_registry
from app.data import DataPipeline, DataConfig
from app.evaluation import Evaluator, EvaluationConfig
from app.utils.database import get_db_connection


# Thread pool for async training
_executor = ThreadPoolExecutor(max_workers=2)


class TrainingEngine:
    """
    Configuration-driven training engine.
    
    Works with any model that implements BaseRecommendationModel.
    """
    
    def __init__(
        self,
        model: BaseRecommendationModel,
        config: Dict[str, Any],
        job_id: str
    ):
        self.model = model
        self.config = config
        self.job_id = job_id
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Get optimizer and loss from model (config-driven)
        self.optimizer = model.get_optimizer(config.get("learning_rate"))
        self.criterion = model.get_loss_function()
        
        # Evaluator
        eval_config = EvaluationConfig(
            metrics=config.get("metrics", ["auc", "precision", "recall"]),
            top_k=config.get("top_k", 10)
        )
        self.evaluator = Evaluator(eval_config)
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        self.best_model_state = None
        
        # Early stopping
        self.early_stopping_enabled = config.get("early_stopping", {}).get("enabled", False)
        self.patience = config.get("early_stopping", {}).get("patience", 5)
        self.min_delta = config.get("early_stopping", {}).get("min_delta", 0.001)
        self.patience_counter = 0
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            user_ids, item_ids, labels = batch
            user_ids = user_ids.to(self.device)
            item_ids = item_ids.to(self.device)
            labels = labels.float().to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(user_ids, item_ids)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        val_loss = self.evaluator.get_loss(
            self.model, val_loader, self.criterion, self.device
        )
        
        metrics = self.evaluator.evaluate(
            self.model, val_loader, self.device
        )
        
        return val_loss, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        experiment_name: str = "recommendation_model",
        run_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Full training loop with MLflow logging.
        """
        mlflow.set_experiment(experiment_name)
        
        run_name = run_name or f"train_{self.job_id[:8]}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log model info
            model_info = self.model.get_model_info()
            mlflow.log_params({
                "model_type": model_info["model_type"],
                "model_name": model_info["model_name"],
                "num_parameters": model_info["num_parameters"],
                "learning_rate": self.config.get("learning_rate", 0.001),
                "batch_size": self.config.get("batch_size", 256),
                "epochs": epochs,
                **{f"arch_{k}": str(v) for k, v in self.model.config.items() 
                   if k not in ["num_users", "num_items"]}
            })
            
            # Update job status
            self._update_job_status("running")
            
            for epoch in range(epochs):
                self.current_epoch = epoch + 1
                
                # Train
                train_loss = self.train_epoch(train_loader)
                self.train_losses.append(train_loss)
                
                # Validate
                val_loss, metrics = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Log to MLflow
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **metrics
                }, step=epoch)
                
                # Update job progress
                self._update_job_progress(
                    epoch=self.current_epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    metrics=metrics
                )
                
                # Save best model
                if val_loss < self.best_val_loss - self.min_delta:
                    self.best_val_loss = val_loss
                    self.best_model_state = self.model.state_dict().copy()
                    self.patience_counter = 0
                    
                    model_path = f"/app/models/{self.job_id}_best.pt"
                    torch.save({
                        "model_state_dict": self.model.state_dict(),
                        "config": self.model.config,
                        "epoch": epoch
                    }, model_path)
                else:
                    self.patience_counter += 1
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, "
                      f"Val Loss: {val_loss:.4f}, AUC: {metrics.get('auc', 0):.4f}")
                
                # Early stopping
                if self.early_stopping_enabled and self.patience_counter >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            # Load best model
            if self.best_model_state:
                self.model.load_state_dict(self.best_model_state)
            
            # Save final model
            model_path = f"/app/models/{self.job_id}_final.pt"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config,
                "training_config": self.config
            }, model_path)
            
            mlflow.log_artifact(model_path)
            mlflow.pytorch.log_model(self.model, "model")
            
            model_uri = f"runs:/{run.info.run_id}/model"
            
            self._update_job_status(
                "completed",
                mlflow_run_id=run.info.run_id,
                model_uri=model_uri
            )
            
            return {
                "run_id": run.info.run_id,
                "model_uri": model_uri,
                "final_train_loss": self.train_losses[-1],
                "final_val_loss": self.val_losses[-1],
                "best_val_loss": self.best_val_loss,
                "metrics": metrics,
                "epochs_trained": self.current_epoch
            }
    
    def _update_job_status(
        self,
        status: str,
        mlflow_run_id: Optional[str] = None,
        model_uri: Optional[str] = None
    ):
        """Update job status in database."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                update_fields = ["status = %s"]
                values = [status]
                
                if status == "running":
                    update_fields.append("started_at = NOW()")
                elif status == "completed":
                    update_fields.append("completed_at = NOW()")
                
                if mlflow_run_id:
                    update_fields.append("mlflow_run_id = %s")
                    values.append(mlflow_run_id)
                
                if model_uri:
                    update_fields.append("model_uri = %s")
                    values.append(model_uri)
                
                values.append(self.job_id)
                
                cur.execute(
                    f"UPDATE training_jobs SET {', '.join(update_fields)} WHERE job_id = %s",
                    values
                )
                conn.commit()
        finally:
            conn.close()
    
    def _update_job_progress(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        metrics: Dict[str, float]
    ):
        """Update job progress in database."""
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE training_jobs 
                    SET current_epoch = %s, 
                        train_loss = %s, 
                        val_loss = %s,
                        metrics = %s
                    WHERE job_id = %s
                    """,
                    (epoch, train_loss, val_loss, json.dumps(metrics), self.job_id)
                )
                conn.commit()
        finally:
            conn.close()


async def run_training_async(
    job_id: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run training in a background thread."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _run_training_sync,
        job_id,
        config
    )


def _run_training_sync(job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous training function with pluggable models.
    """
    try:
        # Initialize registry
        initialize_registry(config_dir="model_configs")
        
        # Load data
        data_pipeline = DataPipeline()
        train_loader, val_loader, num_users, num_items = data_pipeline.load_data(config)
        
        # Get model name from config
        model_name = config.get("model_name", "two_tower")
        
        # Try to load from YAML config first
        config_path = f"model_configs/{model_name}.yaml"
        
        if os.path.exists(config_path):
            model = ModelFactory.create_from_yaml(
                config_path,
                num_users=num_users,
                num_items=num_items,
                **config
            )
        else:
            # Direct instantiation
            model = ModelFactory.create_by_name(
                model_name,
                num_users=num_users,
                num_items=num_items,
                **config
            )
        
        # Create training engine
        engine = TrainingEngine(model, config, job_id)
        
        # Run training
        result = engine.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=config.get("epochs", 10),
            experiment_name=config.get("experiment_name", "recommendation_model"),
            run_name=config.get("run_name")
        )
        
        return result
        
    except Exception as e:
        # Update job as failed
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE training_jobs 
                    SET status = 'failed', error_message = %s, completed_at = NOW()
                    WHERE job_id = %s
                    """,
                    (str(e), job_id)
                )
                conn.commit()
        finally:
            conn.close()
        
        raise


# Keep backward compatibility
class Trainer(TrainingEngine):
    """Alias for backward compatibility."""
    pass

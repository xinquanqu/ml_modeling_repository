"""
Training loop and MLflow integration.
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
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mlflow
import mlflow.pytorch

from app.models.two_tower import TwoTowerModel
from app.utils.database import get_db_connection


# Thread pool for async training
_executor = ThreadPoolExecutor(max_workers=2)


class Trainer:
    """
    PyTorch model trainer with MLflow integration.
    """
    
    def __init__(
        self,
        model: TwoTowerModel,
        config: Dict[str, Any],
        job_id: str
    ):
        self.model = model
        self.config = config
        self.job_id = job_id
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.get("learning_rate", 0.001)
        )
        
        # Loss function (BCE for binary classification)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Training state
        self.current_epoch = 0
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")
        
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> float:
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
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Tuple[float, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids, item_ids, labels = batch
                user_ids = user_ids.to(self.device)
                item_ids = item_ids.to(self.device)
                labels = labels.float().to(self.device)
                
                outputs = self.model(user_ids, item_ids)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # AUC
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.5
        
        binary_preds = (all_preds > 0.5).astype(int)
        
        try:
            precision = precision_score(all_labels, binary_preds, zero_division=0)
            recall = recall_score(all_labels, binary_preds, zero_division=0)
        except:
            precision = 0.0
            recall = 0.0
        
        metrics = {
            "auc": auc,
            "precision": precision,
            "recall": recall
        }
        
        return avg_loss, metrics
    
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
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
        
        run_name = run_name or f"train_{self.job_id[:8]}"
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_params({
                "model_type": "two_tower",
                "learning_rate": self.config.get("learning_rate", 0.001),
                "batch_size": self.config.get("batch_size", 256),
                "epochs": epochs,
                "num_users": self.model.config.get("num_users"),
                "num_items": self.model.config.get("num_items"),
                "embedding_dim": self.model.config.get("user_embedding_dim"),
                "hidden_dims": str(self.model.config.get("hidden_dims")),
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
                
                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "auc": metrics["auc"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"]
                }, step=epoch)
                
                # Update job progress
                self._update_job_progress(
                    epoch=self.current_epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    metrics=metrics
                )
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    model_path = f"/app/models/{self.job_id}_best.pt"
                    torch.save(self.model.state_dict(), model_path)
                
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, AUC: {metrics['auc']:.4f}")
            
            # Save final model to MLflow
            model_path = f"/app/models/{self.job_id}_final.pt"
            torch.save({
                "model_state_dict": self.model.state_dict(),
                "config": self.model.config,
                "training_config": self.config
            }, model_path)
            
            mlflow.log_artifact(model_path)
            
            # Log model
            mlflow.pytorch.log_model(self.model, "model")
            
            # Get model URI
            model_uri = f"runs:/{run.info.run_id}/model"
            
            # Final status update
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
                "metrics": metrics
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
    """
    Run training in a background thread.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        _run_training_sync,
        job_id,
        config
    )


def _run_training_sync(job_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous training function.
    """
    try:
        # Load data
        train_loader, val_loader, num_users, num_items = _load_training_data(config)
        
        # Create model
        model = TwoTowerModel(
            num_users=num_users,
            num_items=num_items,
            user_embedding_dim=config.get("user_embedding_dim", 64),
            item_embedding_dim=config.get("item_embedding_dim", 64),
            hidden_dims=config.get("hidden_dims", [128, 64]),
            dropout=config.get("dropout", 0.1)
        )
        
        # Create trainer
        trainer = Trainer(model, config, job_id)
        
        # Run training
        result = trainer.train(
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


def _load_training_data(
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader, int, int]:
    """
    Load training data from database.
    """
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Get unique users and items
            cur.execute("SELECT DISTINCT user_id FROM raw_data ORDER BY user_id")
            user_ids = [row[0] for row in cur.fetchall()]
            user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
            
            cur.execute("SELECT DISTINCT item_id FROM raw_data ORDER BY item_id")
            item_ids = [row[0] for row in cur.fetchall()]
            item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
            
            num_users = len(user_ids)
            num_items = len(item_ids)
            
            # Load interactions
            cur.execute("""
                SELECT user_id, item_id, COALESCE(label, 1.0) as label
                FROM raw_data
            """)
            
            users = []
            items = []
            labels = []
            
            for row in cur.fetchall():
                if row[0] in user_to_idx and row[1] in item_to_idx:
                    users.append(user_to_idx[row[0]])
                    items.append(item_to_idx[row[1]])
                    labels.append(row[2])
    
    finally:
        conn.close()
    
    # Convert to tensors
    users = torch.tensor(users, dtype=torch.long)
    items = torch.tensor(items, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.float)
    
    # Split into train/val
    n = len(users)
    train_split = config.get("train_split", 0.8)
    
    indices = torch.randperm(n)
    train_size = int(n * train_split)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create datasets
    train_dataset = TensorDataset(
        users[train_indices],
        items[train_indices],
        labels[train_indices]
    )
    
    val_dataset = TensorDataset(
        users[val_indices],
        items[val_indices],
        labels[val_indices]
    )
    
    batch_size = config.get("batch_size", 256)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, num_users, num_items

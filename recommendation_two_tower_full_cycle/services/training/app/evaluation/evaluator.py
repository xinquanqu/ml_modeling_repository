"""
Model Evaluator for standardized metrics.

Provides consistent evaluation across all model types.
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    mean_squared_error,
    log_loss
)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: List[str] = None
    top_k: int = 10
    threshold: float = 0.5
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["auc", "precision", "recall"]


class Evaluator:
    """
    Standardized model evaluation.
    
    Supports multiple metric types for recommendation models.
    """
    
    SUPPORTED_METRICS = {
        "auc", "precision", "recall", "f1",
        "rmse", "mse", "log_loss",
        "ndcg@k", "hit_rate@k", "mrr"
    }
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
    
    def evaluate(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        device: torch.device
    ) -> Dict[str, float]:
        """
        Evaluate model on a dataset.
        
        Args:
            model: The model to evaluate
            data_loader: Validation data loader
            device: Device to run evaluation on
            
        Returns:
            Dictionary of metric names to values
        """
        model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                user_ids, item_ids, labels = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                labels = labels.float().to(device)
                
                outputs = model(user_ids, item_ids)
                
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                
                num_batches += 1
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate requested metrics
        results = {}
        
        for metric in self.config.metrics:
            if metric.startswith("ndcg@") or metric.startswith("hit_rate@"):
                # Handle @k metrics
                k = int(metric.split("@")[1])
                metric_name = metric.split("@")[0]
                results[metric] = self._compute_ranking_metric(
                    metric_name, all_preds, all_labels, k
                )
            else:
                results[metric] = self._compute_metric(
                    metric, all_preds, all_labels
                )
        
        return results
    
    def _compute_metric(
        self,
        metric: str,
        preds: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute a single metric."""
        try:
            if metric == "auc":
                return roc_auc_score(labels, preds)
            
            elif metric == "precision":
                binary_preds = (preds > self.config.threshold).astype(int)
                return precision_score(labels, binary_preds, zero_division=0)
            
            elif metric == "recall":
                binary_preds = (preds > self.config.threshold).astype(int)
                return recall_score(labels, binary_preds, zero_division=0)
            
            elif metric == "f1":
                binary_preds = (preds > self.config.threshold).astype(int)
                prec = precision_score(labels, binary_preds, zero_division=0)
                rec = recall_score(labels, binary_preds, zero_division=0)
                if prec + rec == 0:
                    return 0.0
                return 2 * prec * rec / (prec + rec)
            
            elif metric == "rmse":
                return np.sqrt(mean_squared_error(labels, preds))
            
            elif metric == "mse":
                return mean_squared_error(labels, preds)
            
            elif metric == "log_loss":
                return log_loss(labels, preds)
            
            elif metric == "mrr":
                return self._compute_mrr(preds, labels)
            
            else:
                print(f"Warning: Unknown metric {metric}")
                return 0.0
                
        except Exception as e:
            print(f"Warning: Failed to compute {metric}: {e}")
            return 0.0
    
    def _compute_ranking_metric(
        self,
        metric: str,
        preds: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """Compute ranking metrics like NDCG@k."""
        try:
            if metric == "ndcg":
                return self._compute_ndcg(preds, labels, k)
            elif metric == "hit_rate":
                return self._compute_hit_rate(preds, labels, k)
            else:
                return 0.0
        except Exception as e:
            print(f"Warning: Failed to compute {metric}@{k}: {e}")
            return 0.0
    
    def _compute_ndcg(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """Compute NDCG@k."""
        # Get top-k indices
        top_k_indices = np.argsort(preds)[-k:][::-1]
        
        # DCG
        dcg = 0.0
        for i, idx in enumerate(top_k_indices):
            if labels[idx] > 0:
                dcg += 1.0 / np.log2(i + 2)
        
        # Ideal DCG
        sorted_labels = np.sort(labels)[::-1][:k]
        idcg = 0.0
        for i, rel in enumerate(sorted_labels):
            if rel > 0:
                idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def _compute_hit_rate(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> float:
        """Compute Hit Rate@k."""
        top_k_indices = np.argsort(preds)[-k:][::-1]
        hits = sum(labels[idx] > 0 for idx in top_k_indices)
        return hits / k
    
    def _compute_mrr(
        self,
        preds: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        sorted_indices = np.argsort(preds)[::-1]
        
        for rank, idx in enumerate(sorted_indices, 1):
            if labels[idx] > 0:
                return 1.0 / rank
        
        return 0.0
    
    def get_loss(
        self,
        model: torch.nn.Module,
        data_loader: DataLoader,
        loss_fn: torch.nn.Module,
        device: torch.device
    ) -> float:
        """Calculate average loss on a dataset."""
        model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in data_loader:
                user_ids, item_ids, labels = batch
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                labels = labels.float().to(device)
                
                outputs = model(user_ids, item_ids)
                loss = loss_fn(outputs, labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)

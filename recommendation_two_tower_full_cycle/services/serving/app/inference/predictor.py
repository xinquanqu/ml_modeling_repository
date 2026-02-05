"""
Model predictor for inference.
"""

import time
from typing import Dict, Any, List, Optional, Tuple

import torch
import numpy as np
from prometheus_client import Histogram


# Metrics
INFERENCE_LATENCY = Histogram(
    'model_inference_latency_seconds',
    'Model inference latency',
    ['operation'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25]
)


class TwoTowerPredictor:
    """
    Predictor for the Two-Tower recommendation model.
    
    Provides inference methods for:
    - User-item similarity scoring
    - Top-K recommendations
    - Batch predictions
    """
    
    def __init__(self, model: torch.nn.Module, version: str):
        self.model = model
        self.version = version
        self.model_type = "two_tower"
        
        # Set to evaluation mode
        self.model.eval()
        
        # Move to appropriate device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Cache for embeddings
        self._user_embedding_cache: Dict[int, torch.Tensor] = {}
        self._item_embedding_cache: Dict[int, torch.Tensor] = {}
        
    @torch.no_grad()
    def predict(
        self,
        user_id: int,
        item_ids: List[int]
    ) -> List[Tuple[int, float]]:
        """
        Predict scores for a user and a list of items.
        
        Args:
            user_id: User ID
            item_ids: List of item IDs to score
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        start_time = time.time()
        
        # Get user embedding
        user_tensor = torch.tensor([user_id], dtype=torch.long, device=self.device)
        user_emb = self.model.get_user_embeddings(user_tensor)
        
        # Get item embeddings
        item_tensor = torch.tensor(item_ids, dtype=torch.long, device=self.device)
        item_embs = self.model.get_item_embeddings(item_tensor)
        
        # Compute scores
        scores = self.model.compute_scores(user_emb, item_embs)
        scores = torch.sigmoid(scores).cpu().numpy()
        
        # Create result pairs
        results = [(item_id, float(score)) for item_id, score in zip(item_ids, scores)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        INFERENCE_LATENCY.labels(operation="predict").observe(time.time() - start_time)
        
        return results
    
    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        num_items: int,
        candidate_items: Optional[List[int]] = None,
        exclude_items: Optional[List[int]] = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-K recommendations for a user.
        
        Args:
            user_id: User ID
            num_items: Number of recommendations to return
            candidate_items: Optional list of candidate items (if None, uses all items)
            exclude_items: Items to exclude from recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        start_time = time.time()
        
        if candidate_items is None:
            # Use all items
            num_total_items = self.model.config.get("num_items", 1000)
            candidate_items = list(range(num_total_items))
        
        if exclude_items:
            exclude_set = set(exclude_items)
            candidate_items = [i for i in candidate_items if i not in exclude_set]
        
        results = self.predict(user_id, candidate_items)
        
        INFERENCE_LATENCY.labels(operation="recommend").observe(time.time() - start_time)
        
        return results[:num_items]
    
    @torch.no_grad()
    def batch_predict(
        self,
        user_ids: List[int],
        item_ids: List[int]
    ) -> np.ndarray:
        """
        Batch prediction for user-item pairs.
        
        Args:
            user_ids: List of user IDs (same length as item_ids)
            item_ids: List of item IDs (same length as user_ids)
            
        Returns:
            Array of scores
        """
        start_time = time.time()
        
        user_tensor = torch.tensor(user_ids, dtype=torch.long, device=self.device)
        item_tensor = torch.tensor(item_ids, dtype=torch.long, device=self.device)
        
        scores = self.model(user_tensor, item_tensor)
        scores = torch.sigmoid(scores).cpu().numpy()
        
        INFERENCE_LATENCY.labels(operation="batch_predict").observe(time.time() - start_time)
        
        return scores
    
    @torch.no_grad()
    def get_user_embedding(self, user_id: int) -> np.ndarray:
        """Get the embedding for a user."""
        if user_id in self._user_embedding_cache:
            return self._user_embedding_cache[user_id].cpu().numpy()
        
        user_tensor = torch.tensor([user_id], dtype=torch.long, device=self.device)
        embedding = self.model.get_user_embeddings(user_tensor)
        
        self._user_embedding_cache[user_id] = embedding
        
        return embedding.cpu().numpy().squeeze()
    
    @torch.no_grad()
    def get_item_embedding(self, item_id: int) -> np.ndarray:
        """Get the embedding for an item."""
        if item_id in self._item_embedding_cache:
            return self._item_embedding_cache[item_id].cpu().numpy()
        
        item_tensor = torch.tensor([item_id], dtype=torch.long, device=self.device)
        embedding = self.model.get_item_embeddings(item_tensor)
        
        self._item_embedding_cache[item_id] = embedding
        
        return embedding.cpu().numpy().squeeze()
    
    def clear_cache(self):
        """Clear embedding caches."""
        self._user_embedding_cache.clear()
        self._item_embedding_cache.clear()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata."""
        return {
            "version": self.version,
            "model_type": self.model_type,
            "config": self.model.config,
            "device": str(self.device),
            "cache_size": {
                "users": len(self._user_embedding_cache),
                "items": len(self._item_embedding_cache)
            }
        }

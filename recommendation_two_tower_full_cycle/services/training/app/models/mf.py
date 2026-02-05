"""
Matrix Factorization Model.

Classic recommendation model using factorized user-item matrices.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from app.models.base import BaseRecommendationModel
from app.registry.model_registry import ModelRegistry


@ModelRegistry.register("matrix_factorization")
class MatrixFactorizationModel(BaseRecommendationModel):
    """
    Matrix Factorization for collaborative filtering.
    
    Simple and effective baseline model.
    """
    
    model_type = "embedding"
    model_name = "MatrixFactorization"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        embedding_dim = config.get("embedding_dim", 64)
        
        self.user_embedding = nn.Embedding(
            config["num_users"], 
            embedding_dim
        )
        self.item_embedding = nn.Embedding(
            config["num_items"], 
            embedding_dim
        )
        
        # User and item biases
        self.user_bias = nn.Embedding(config["num_users"], 1)
        self.item_bias = nn.Embedding(config["num_items"], 1)
        
        # Global bias
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
    
    def _validate_config(self) -> None:
        required = ["num_users", "num_items"]
        for key in required:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")
    
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Dot product
        dot = torch.sum(user_emb * item_emb, dim=1)
        
        # Add biases
        u_bias = self.user_bias(user_ids).squeeze()
        i_bias = self.item_bias(item_ids).squeeze()
        
        return dot + u_bias + i_bias + self.global_bias
    
    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.user_embedding(user_ids)
    
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.item_embedding(item_ids)

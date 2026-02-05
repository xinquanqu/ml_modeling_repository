"""
Two-Tower Recommendation Model.

A neural network architecture for recommendation systems that learns
separate embeddings for users and items.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any

from app.models.base import BaseRecommendationModel
from app.registry.model_registry import ModelRegistry


class UserTower(nn.Module):
    """User tower of the two-tower model."""
    
    def __init__(
        self,
        num_users: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        num_user_features: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        input_dim = embedding_dim + num_user_features
        layers = []
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else embedding_dim
        
    def forward(
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.user_embedding(user_ids)
        
        if user_features is not None:
            x = torch.cat([x, user_features], dim=1)
        
        x = self.feature_layers(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x


class ItemTower(nn.Module):
    """Item tower of the two-tower model."""
    
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        num_item_features: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        input_dim = embedding_dim + num_item_features
        layers = []
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1] if hidden_dims else embedding_dim
        
    def forward(
        self,
        item_ids: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.item_embedding(item_ids)
        
        if item_features is not None:
            x = torch.cat([x, item_features], dim=1)
        
        x = self.feature_layers(x)
        x = F.normalize(x, p=2, dim=1)
        
        return x


@ModelRegistry.register("two_tower")
class TwoTowerModel(BaseRecommendationModel):
    """
    Two-Tower Recommendation Model.
    
    Combines user and item towers to predict user-item affinity.
    """
    
    model_type = "embedding"
    model_name = "TwoTower"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        self.user_tower = UserTower(
            num_users=config["num_users"],
            embedding_dim=config.get("user_embedding_dim", 64),
            hidden_dims=config.get("hidden_dims", [128, 64]),
            num_user_features=config.get("num_user_features", 0),
            dropout=config.get("dropout", 0.1)
        )
        
        self.item_tower = ItemTower(
            num_items=config["num_items"],
            embedding_dim=config.get("item_embedding_dim", 64),
            hidden_dims=config.get("hidden_dims", [128, 64]),
            num_item_features=config.get("num_item_features", 0),
            dropout=config.get("dropout", 0.1)
        )
        
        self.temperature = config.get("temperature", 0.1)
    
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
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        
        similarity = torch.sum(user_emb * item_emb, dim=1) / self.temperature
        
        return similarity
    
    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.user_tower(user_ids, user_features)
    
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.item_tower(item_ids, item_features)

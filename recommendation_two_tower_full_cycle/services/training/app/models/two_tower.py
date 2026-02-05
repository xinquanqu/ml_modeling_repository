"""
Two-Tower Recommendation Model.

A neural network architecture for recommendation systems that learns
separate embeddings for users and items.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict, Any


class UserTower(nn.Module):
    """
    User tower of the two-tower model.
    
    Processes user features and produces user embeddings.
    """
    
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
        
        # User feature processing layers
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
        """
        Forward pass for user tower.
        
        Args:
            user_ids: Tensor of user IDs [batch_size]
            user_features: Optional tensor of user features [batch_size, num_features]
            
        Returns:
            User embeddings [batch_size, output_dim]
        """
        x = self.user_embedding(user_ids)
        
        if user_features is not None:
            x = torch.cat([x, user_features], dim=1)
        
        x = self.feature_layers(x)
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x


class ItemTower(nn.Module):
    """
    Item tower of the two-tower model.
    
    Processes item features and produces item embeddings.
    """
    
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
        
        # Item feature processing layers
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
        """
        Forward pass for item tower.
        
        Args:
            item_ids: Tensor of item IDs [batch_size]
            item_features: Optional tensor of item features [batch_size, num_features]
            
        Returns:
            Item embeddings [batch_size, output_dim]
        """
        x = self.item_embedding(item_ids)
        
        if item_features is not None:
            x = torch.cat([x, item_features], dim=1)
        
        x = self.feature_layers(x)
        
        # L2 normalize for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
        return x


class TwoTowerModel(nn.Module):
    """
    Two-Tower Recommendation Model.
    
    Combines user and item towers to predict user-item affinity.
    """
    
    def __init__(
        self,
        num_users: int,
        num_items: int,
        user_embedding_dim: int = 64,
        item_embedding_dim: int = 64,
        hidden_dims: List[int] = [128, 64],
        num_user_features: int = 0,
        num_item_features: int = 0,
        dropout: float = 0.1,
        temperature: float = 0.1
    ):
        super().__init__()
        
        self.user_tower = UserTower(
            num_users=num_users,
            embedding_dim=user_embedding_dim,
            hidden_dims=hidden_dims,
            num_user_features=num_user_features,
            dropout=dropout
        )
        
        self.item_tower = ItemTower(
            num_items=num_items,
            embedding_dim=item_embedding_dim,
            hidden_dims=hidden_dims,
            num_item_features=num_item_features,
            dropout=dropout
        )
        
        self.temperature = temperature
        
        # Model metadata
        self.config = {
            "num_users": num_users,
            "num_items": num_items,
            "user_embedding_dim": user_embedding_dim,
            "item_embedding_dim": item_embedding_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "temperature": temperature
        }
        
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass computing similarity scores.
        
        Args:
            user_ids: User ID tensor [batch_size]
            item_ids: Item ID tensor [batch_size]
            user_features: Optional user features [batch_size, num_user_features]
            item_features: Optional item features [batch_size, num_item_features]
            
        Returns:
            Similarity scores [batch_size]
        """
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        
        # Cosine similarity (embeddings are already normalized)
        similarity = torch.sum(user_emb * item_emb, dim=1) / self.temperature
        
        return similarity
    
    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get user embeddings for inference."""
        return self.user_tower(user_ids, user_features)
    
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get item embeddings for inference."""
        return self.item_tower(item_ids, item_features)
    
    def compute_scores(
        self,
        user_embedding: torch.Tensor,
        item_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scores between a user and multiple items.
        
        Args:
            user_embedding: Single user embedding [1, dim] or [dim]
            item_embeddings: Item embeddings [num_items, dim]
            
        Returns:
            Scores [num_items]
        """
        if user_embedding.dim() == 1:
            user_embedding = user_embedding.unsqueeze(0)
        
        # Cosine similarity
        scores = torch.mm(user_embedding, item_embeddings.t()) / self.temperature
        
        return scores.squeeze(0)

"""
Two-Tower model for serving.
Copy of training model for independent deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class UserTower(nn.Module):
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
        
    def forward(self, user_ids, user_features=None):
        x = self.user_embedding(user_ids)
        if user_features is not None:
            x = torch.cat([x, user_features], dim=1)
        x = self.feature_layers(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ItemTower(nn.Module):
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
        
    def forward(self, item_ids, item_features=None):
        x = self.item_embedding(item_ids)
        if item_features is not None:
            x = torch.cat([x, item_features], dim=1)
        x = self.feature_layers(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class TwoTowerModel(nn.Module):
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
        self.config = {
            "num_users": num_users,
            "num_items": num_items,
            "user_embedding_dim": user_embedding_dim,
            "item_embedding_dim": item_embedding_dim,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "temperature": temperature
        }
        
    def forward(self, user_ids, item_ids, user_features=None, item_features=None):
        user_emb = self.user_tower(user_ids, user_features)
        item_emb = self.item_tower(item_ids, item_features)
        similarity = torch.sum(user_emb * item_emb, dim=1) / self.temperature
        return similarity
    
    def get_user_embeddings(self, user_ids, user_features=None):
        return self.user_tower(user_ids, user_features)
    
    def get_item_embeddings(self, item_ids, item_features=None):
        return self.item_tower(item_ids, item_features)
    
    def compute_scores(self, user_embedding, item_embeddings):
        if user_embedding.dim() == 1:
            user_embedding = user_embedding.unsqueeze(0)
        scores = torch.mm(user_embedding, item_embeddings.t()) / self.temperature
        return scores.squeeze(0)

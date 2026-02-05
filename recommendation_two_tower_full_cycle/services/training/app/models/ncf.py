"""
Neural Collaborative Filtering (NCF) Model.

Combines matrix factorization with neural networks for
improved recommendation performance.

Reference: He et al., "Neural Collaborative Filtering" (WWW 2017)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from app.models.base import BaseRecommendationModel
from app.registry.model_registry import ModelRegistry


@ModelRegistry.register("ncf")
class NeuralCollaborativeFiltering(BaseRecommendationModel):
    """
    Neural Collaborative Filtering.
    
    Combines GMF (Generalized Matrix Factorization) and MLP
    for learning user-item interactions.
    """
    
    model_type = "hybrid"
    model_name = "NeuralCF"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        gmf_dim = config.get("gmf_embedding_dim", 32)
        mlp_dim = config.get("mlp_embedding_dim", 32)
        mlp_layers = config.get("mlp_layers", [64, 32, 16])
        dropout = config.get("dropout", 0.1)
        
        # GMF embeddings
        self.user_gmf_embedding = nn.Embedding(config["num_users"], gmf_dim)
        self.item_gmf_embedding = nn.Embedding(config["num_items"], gmf_dim)
        
        # MLP embeddings
        self.user_mlp_embedding = nn.Embedding(config["num_users"], mlp_dim)
        self.item_mlp_embedding = nn.Embedding(config["num_items"], mlp_dim)
        
        # MLP layers
        mlp_input_dim = mlp_dim * 2
        layers = []
        
        for hidden_dim in mlp_layers:
            layers.extend([
                nn.Linear(mlp_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            mlp_input_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Final prediction layer
        final_dim = gmf_dim + mlp_layers[-1]
        self.prediction = nn.Linear(final_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embeddings with small random values."""
        for embedding in [
            self.user_gmf_embedding, self.item_gmf_embedding,
            self.user_mlp_embedding, self.item_mlp_embedding
        ]:
            nn.init.normal_(embedding.weight, std=0.01)
    
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
        # GMF path
        user_gmf = self.user_gmf_embedding(user_ids)
        item_gmf = self.item_gmf_embedding(item_ids)
        gmf_output = user_gmf * item_gmf
        
        # MLP path
        user_mlp = self.user_mlp_embedding(user_ids)
        item_mlp = self.item_mlp_embedding(item_ids)
        mlp_input = torch.cat([user_mlp, item_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Combine GMF and MLP
        combined = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Final prediction
        output = self.prediction(combined).squeeze()
        
        return output
    
    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get combined GMF + MLP user embeddings."""
        gmf_emb = self.user_gmf_embedding(user_ids)
        mlp_emb = self.user_mlp_embedding(user_ids)
        return torch.cat([gmf_emb, mlp_emb], dim=1)
    
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get combined GMF + MLP item embeddings."""
        gmf_emb = self.item_gmf_embedding(item_ids)
        mlp_emb = self.item_mlp_embedding(item_ids)
        return torch.cat([gmf_emb, mlp_emb], dim=1)

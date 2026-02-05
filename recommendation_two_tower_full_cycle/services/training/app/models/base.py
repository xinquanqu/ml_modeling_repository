"""
Base model interface for all recommendation models.

All models must inherit from BaseRecommendationModel to be compatible
with the training framework.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn


class BaseRecommendationModel(nn.Module, ABC):
    """
    Abstract base class for recommendation models.
    
    All models in the registry must implement this interface.
    """
    
    # Model metadata
    model_type: str = "base"
    model_name: str = "BaseModel"
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate that required config keys are present."""
        pass
    
    @abstractmethod
    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass computing prediction scores.
        
        Args:
            user_ids: User ID tensor [batch_size]
            item_ids: Item ID tensor [batch_size]
            user_features: Optional user features [batch_size, num_features]
            item_features: Optional item features [batch_size, num_features]
            
        Returns:
            Prediction scores [batch_size]
        """
        pass
    
    @abstractmethod
    def get_user_embeddings(
        self,
        user_ids: torch.Tensor,
        user_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get user embeddings for inference."""
        pass
    
    @abstractmethod
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
        item_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Get item embeddings for inference."""
        pass
    
    def get_loss_function(self) -> nn.Module:
        """Get the loss function for this model."""
        loss_type = self.config.get("loss", "bce_with_logits")
        
        if loss_type == "bce_with_logits":
            return nn.BCEWithLogitsLoss()
        elif loss_type == "bce":
            return nn.BCELoss()
        elif loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def get_optimizer(self, learning_rate: Optional[float] = None) -> torch.optim.Optimizer:
        """Get the optimizer for this model."""
        lr = learning_rate or self.config.get("learning_rate", 0.001)
        optimizer_type = self.config.get("optimizer", "adam")
        
        if optimizer_type == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr)
        elif optimizer_type == "sgd":
            return torch.optim.SGD(
                self.parameters(), 
                lr=lr,
                momentum=self.config.get("momentum", 0.9)
            )
        elif optimizer_type == "adamw":
            return torch.optim.AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=self.config.get("weight_decay", 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
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
        
        temperature = self.config.get("temperature", 1.0)
        scores = torch.mm(user_embedding, item_embeddings.t()) / temperature
        
        return scores.squeeze(0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model metadata for logging."""
        return {
            "model_type": self.model_type,
            "model_name": self.model_name,
            "config": self.config,
            "num_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseRecommendationModel":
        """Create model instance from configuration."""
        return cls(config)

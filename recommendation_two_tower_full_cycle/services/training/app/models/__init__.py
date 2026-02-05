"""Models package with registry integration."""

from app.models.base import BaseRecommendationModel
from app.models.two_tower import TwoTowerModel
from app.models.mf import MatrixFactorizationModel
from app.models.ncf import NeuralCollaborativeFiltering

__all__ = [
    "BaseRecommendationModel",
    "TwoTowerModel",
    "MatrixFactorizationModel",
    "NeuralCollaborativeFiltering"
]

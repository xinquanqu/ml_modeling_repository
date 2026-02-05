"""
Model manager for loading and serving models.
"""

import os
import json
from typing import Dict, Optional, Any, List
from datetime import datetime

import torch
import numpy as np
import mlflow
import redis.asyncio as redis

from app.inference.predictor import TwoTowerPredictor


class ModelManager:
    """
    Manages model loading, versioning, and serving.
    """
    
    def __init__(self):
        self.models: Dict[str, TwoTowerPredictor] = {}
        self.active_model: Optional[str] = None
        self.redis: Optional[redis.Redis] = None
        
        # Configuration
        self.mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        self.model_cache_dir = os.getenv("MODEL_CACHE_DIR", "/app/models")
        
    async def initialize(self):
        """Initialize the model manager."""
        # Set up MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Connect to Redis for caching
        redis_host = os.getenv("REDIS_HOST", "redis")
        redis_port = int(os.getenv("REDIS_PORT", "6379"))
        
        try:
            self.redis = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True
            )
            await self.redis.ping()
            print(f"Redis connected: {redis_host}:{redis_port}")
        except Exception as e:
            print(f"Redis connection failed: {e}")
            self.redis = None
        
        # Try to load the latest model
        await self.load_latest_model()
        
    async def load_latest_model(self) -> bool:
        """Load the latest production model."""
        try:
            # Search for completed training runs
            runs = mlflow.search_runs(
                experiment_names=["recommendation_model"],
                filter_string="status = 'FINISHED'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if len(runs) > 0:
                run_id = runs.iloc[0]["run_id"]
                model_uri = f"runs:/{run_id}/model"
                
                await self.load_model(model_uri, version=run_id[:8])
                self.active_model = run_id[:8]
                print(f"Loaded latest model: {run_id[:8]}")
                return True
                
        except Exception as e:
            print(f"Failed to load latest model: {e}")
        
        return False
    
    async def load_model(
        self,
        model_uri: str,
        version: str
    ) -> bool:
        """
        Load a model from MLflow.
        
        Args:
            model_uri: MLflow model URI (e.g., runs:/run_id/model)
            version: Version identifier for this model
        """
        try:
            # Load model from MLflow
            model = mlflow.pytorch.load_model(model_uri)
            
            # Create predictor
            predictor = TwoTowerPredictor(model, version)
            
            # Store in cache
            self.models[version] = predictor
            
            print(f"Model loaded: version={version}, uri={model_uri}")
            return True
            
        except Exception as e:
            print(f"Failed to load model {model_uri}: {e}")
            return False
    
    async def load_model_from_path(
        self,
        model_path: str,
        version: str
    ) -> bool:
        """Load a model from a local file path."""
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            
            # Reconstruct model from config
            from app.models.two_tower import TwoTowerModel
            
            config = checkpoint.get("config", {})
            model = TwoTowerModel(
                num_users=config.get("num_users", 1000),
                num_items=config.get("num_items", 1000),
                user_embedding_dim=config.get("user_embedding_dim", 64),
                item_embedding_dim=config.get("item_embedding_dim", 64),
                hidden_dims=config.get("hidden_dims", [128, 64])
            )
            
            model.load_state_dict(checkpoint["model_state_dict"])
            
            predictor = TwoTowerPredictor(model, version)
            self.models[version] = predictor
            
            print(f"Model loaded from path: {model_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load model from path: {e}")
            return False
    
    def get_predictor(self, version: Optional[str] = None) -> Optional[TwoTowerPredictor]:
        """Get a predictor for a specific version or the active version."""
        if version:
            return self.models.get(version)
        
        if self.active_model:
            return self.models.get(self.active_model)
        
        # Return any available model
        if self.models:
            return next(iter(self.models.values()))
        
        return None
    
    def set_active_model(self, version: str) -> bool:
        """Set the active model version."""
        if version in self.models:
            self.active_model = version
            return True
        return False
    
    def get_loaded_models_count(self) -> int:
        """Get the number of loaded models."""
        return len(self.models)
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all loaded models."""
        return [
            {
                "version": version,
                "is_active": version == self.active_model,
                "model_type": predictor.model_type
            }
            for version, predictor in self.models.items()
        ]
    
    async def unload_model(self, version: str) -> bool:
        """Unload a model from memory."""
        if version in self.models:
            del self.models[version]
            if self.active_model == version:
                self.active_model = next(iter(self.models.keys()), None) if self.models else None
            return True
        return False

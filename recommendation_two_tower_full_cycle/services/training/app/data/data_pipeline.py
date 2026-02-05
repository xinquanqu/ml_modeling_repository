"""
Data Pipeline for training.

Provides unified data loading with Feature Store integration.
"""

import os
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import requests

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from app.utils.database import get_db_connection


@dataclass
class DataConfig:
    """Configuration for data loading."""
    batch_size: int = 256
    train_split: float = 0.8
    shuffle: bool = True
    num_workers: int = 0
    use_feature_store: bool = True
    feature_store_url: str = "http://feature-store:8000"


class FeatureStoreClient:
    """
    Client for Feature Store service.
    
    Retrieves pre-computed features for users and items.
    """
    
    def __init__(self, base_url: str = "http://feature-store:8000"):
        self.base_url = base_url
        self.timeout = 30
    
    def get_user_features(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get features for a user."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/user/{user_id}",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("features", {})
        except Exception as e:
            print(f"Warning: Failed to get user features: {e}")
        return None
    
    def get_item_features(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get features for an item."""
        try:
            response = requests.get(
                f"{self.base_url}/api/v1/item/{item_id}",
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("features", {})
        except Exception as e:
            print(f"Warning: Failed to get item features: {e}")
        return None
    
    def get_batch_features(
        self,
        entity_type: str,
        entity_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Get features for multiple entities."""
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/batch-get",
                json={
                    "entity_type": entity_type,
                    "entity_ids": entity_ids
                },
                timeout=self.timeout
            )
            if response.status_code == 200:
                return response.json().get("features", {})
        except Exception as e:
            print(f"Warning: Failed to get batch features: {e}")
        return {}


class DataPipeline:
    """
    Unified data pipeline for training.
    
    Loads data from database and optionally enriches with
    features from Feature Store.
    """
    
    def __init__(self, config: Optional[DataConfig] = None):
        self.config = config or DataConfig()
        self.feature_client = FeatureStoreClient(self.config.feature_store_url)
        
        # ID mappings
        self.user_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_item: Dict[int, str] = {}
    
    def load_data(
        self,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Tuple[DataLoader, DataLoader, int, int]:
        """
        Load training data.
        
        Args:
            training_config: Optional config overrides
            
        Returns:
            train_loader, val_loader, num_users, num_items
        """
        config = training_config or {}
        batch_size = config.get("batch_size", self.config.batch_size)
        train_split = config.get("train_split", self.config.train_split)
        
        # Load raw data from database
        users, items, labels = self._load_from_database()
        
        num_users = len(self.user_to_idx)
        num_items = len(self.item_to_idx)
        
        # Create tensors
        user_tensor = torch.tensor(users, dtype=torch.long)
        item_tensor = torch.tensor(items, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.float)
        
        # Train/val split
        n = len(users)
        indices = torch.randperm(n)
        train_size = int(n * train_split)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Create datasets
        train_dataset = TensorDataset(
            user_tensor[train_indices],
            item_tensor[train_indices],
            label_tensor[train_indices]
        )
        
        val_dataset = TensorDataset(
            user_tensor[val_indices],
            item_tensor[val_indices],
            label_tensor[val_indices]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        return train_loader, val_loader, num_users, num_items
    
    def _load_from_database(self) -> Tuple[List[int], List[int], List[float]]:
        """Load raw interactions from database."""
        conn = get_db_connection()
        
        try:
            with conn.cursor() as cur:
                # Get unique users
                cur.execute("SELECT DISTINCT user_id FROM raw_data ORDER BY user_id")
                user_ids = [row[0] for row in cur.fetchall()]
                self.user_to_idx = {uid: idx for idx, uid in enumerate(user_ids)}
                self.idx_to_user = {idx: uid for uid, idx in self.user_to_idx.items()}
                
                # Get unique items
                cur.execute("SELECT DISTINCT item_id FROM raw_data ORDER BY item_id")
                item_ids = [row[0] for row in cur.fetchall()]
                self.item_to_idx = {iid: idx for idx, iid in enumerate(item_ids)}
                self.idx_to_item = {idx: iid for iid, idx in self.item_to_idx.items()}
                
                # Load interactions
                cur.execute("""
                    SELECT user_id, item_id, COALESCE(label, 1.0) as label
                    FROM raw_data
                """)
                
                users = []
                items = []
                labels = []
                
                for row in cur.fetchall():
                    if row[0] in self.user_to_idx and row[1] in self.item_to_idx:
                        users.append(self.user_to_idx[row[0]])
                        items.append(self.item_to_idx[row[1]])
                        labels.append(float(row[2]))
        
        finally:
            conn.close()
        
        return users, items, labels
    
    def load_with_features(
        self,
        training_config: Optional[Dict[str, Any]] = None,
        user_feature_names: Optional[List[str]] = None,
        item_feature_names: Optional[List[str]] = None
    ) -> Tuple[DataLoader, DataLoader, int, int, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Load data with features from Feature Store.
        
        Returns:
            train_loader, val_loader, num_users, num_items, user_features, item_features
        """
        # First load basic data
        train_loader, val_loader, num_users, num_items = self.load_data(training_config)
        
        user_features = None
        item_features = None
        
        if self.config.use_feature_store and user_feature_names:
            user_features = self._load_user_features(user_feature_names)
        
        if self.config.use_feature_store and item_feature_names:
            item_features = self._load_item_features(item_feature_names)
        
        return train_loader, val_loader, num_users, num_items, user_features, item_features
    
    def _load_user_features(self, feature_names: List[str]) -> Optional[torch.Tensor]:
        """Load user features from Feature Store."""
        num_users = len(self.user_to_idx)
        num_features = len(feature_names)
        
        features = torch.zeros(num_users, num_features)
        
        for user_id, idx in self.user_to_idx.items():
            user_feats = self.feature_client.get_user_features(user_id)
            if user_feats:
                for i, feat_name in enumerate(feature_names):
                    features[idx, i] = float(user_feats.get(feat_name, 0.0))
        
        return features
    
    def _load_item_features(self, feature_names: List[str]) -> Optional[torch.Tensor]:
        """Load item features from Feature Store."""
        num_items = len(self.item_to_idx)
        num_features = len(feature_names)
        
        features = torch.zeros(num_items, num_features)
        
        for item_id, idx in self.item_to_idx.items():
            item_feats = self.feature_client.get_item_features(item_id)
            if item_feats:
                for i, feat_name in enumerate(feature_names):
                    features[idx, i] = float(item_feats.get(feat_name, 0.0))
        
        return features
    
    def get_id_mappings(self) -> Dict[str, Dict]:
        """Get user and item ID mappings."""
        return {
            "user_to_idx": self.user_to_idx,
            "idx_to_user": self.idx_to_user,
            "item_to_idx": self.item_to_idx,
            "idx_to_item": self.idx_to_item
        }

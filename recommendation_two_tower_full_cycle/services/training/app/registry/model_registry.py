"""
Model Registry for dynamic model discovery and instantiation.

The registry pattern allows models to be:
- Registered by name
- Loaded from YAML configuration files
- Instantiated dynamically at runtime
"""

import os
import importlib
from typing import Dict, Any, Type, Optional, List
from pathlib import Path

import yaml

from app.models.base import BaseRecommendationModel


class ModelRegistry:
    """
    Central registry for recommendation models.
    
    Manages model registration, discovery, and instantiation.
    """
    
    _models: Dict[str, Type[BaseRecommendationModel]] = {}
    _configs: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class.
        
        Usage:
            @ModelRegistry.register("two_tower")
            class TwoTowerModel(BaseRecommendationModel):
                ...
        """
        def decorator(model_class: Type[BaseRecommendationModel]):
            cls._models[name] = model_class
            return model_class
        return decorator
    
    @classmethod
    def register_model(cls, name: str, model_class: Type[BaseRecommendationModel]):
        """Programmatically register a model class."""
        cls._models[name] = model_class
    
    @classmethod
    def get_model_class(cls, name: str) -> Type[BaseRecommendationModel]:
        """Get a registered model class by name."""
        if name not in cls._models:
            raise ValueError(f"Model '{name}' not found in registry. Available: {list(cls._models.keys())}")
        return cls._models[name]
    
    @classmethod
    def list_models(cls) -> List[str]:
        """List all registered model names."""
        return list(cls._models.keys())
    
    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """Load model configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    @classmethod
    def load_configs_from_directory(cls, config_dir: str) -> None:
        """Load all YAML configs from a directory."""
        config_path = Path(config_dir)
        
        if not config_path.exists():
            return
        
        for yaml_file in config_path.glob("*.yaml"):
            try:
                config = cls.load_config(str(yaml_file))
                model_name = config.get("model", {}).get("name", yaml_file.stem)
                cls._configs[model_name] = config
            except Exception as e:
                print(f"Warning: Failed to load config {yaml_file}: {e}")
    
    @classmethod
    def get_config(cls, model_name: str) -> Optional[Dict[str, Any]]:
        """Get a loaded configuration by model name."""
        return cls._configs.get(model_name)
    
    @classmethod
    def list_configs(cls) -> List[str]:
        """List all loaded configuration names."""
        return list(cls._configs.keys())


class ModelFactory:
    """
    Factory for creating model instances from configuration.
    """
    
    @staticmethod
    def create_from_config(
        config: Dict[str, Any],
        num_users: int,
        num_items: int,
        **overrides
    ) -> BaseRecommendationModel:
        """
        Create a model instance from configuration.
        
        Args:
            config: Model configuration dict (from YAML)
            num_users: Number of unique users
            num_items: Number of unique items
            **overrides: Override any config values
            
        Returns:
            Instantiated model
        """
        model_info = config.get("model", {})
        model_name = model_info.get("name", "two_tower")
        
        # Get architecture config
        arch_config = config.get("architecture", {})
        training_config = config.get("training", {})
        
        # Build full config for model
        full_config = {
            "num_users": num_users,
            "num_items": num_items,
            **arch_config,
            **training_config,
            **overrides
        }
        
        # Get model class
        model_class = ModelRegistry.get_model_class(model_name)
        
        return model_class(full_config)
    
    @staticmethod
    def create_from_yaml(
        yaml_path: str,
        num_users: int,
        num_items: int,
        **overrides
    ) -> BaseRecommendationModel:
        """
        Create a model instance from a YAML config file.
        
        Args:
            yaml_path: Path to YAML configuration file
            num_users: Number of unique users
            num_items: Number of unique items
            **overrides: Override any config values
            
        Returns:
            Instantiated model
        """
        config = ModelRegistry.load_config(yaml_path)
        return ModelFactory.create_from_config(config, num_users, num_items, **overrides)
    
    @staticmethod
    def create_by_name(
        model_name: str,
        num_users: int,
        num_items: int,
        **config_overrides
    ) -> BaseRecommendationModel:
        """
        Create a model by name using registered config or defaults.
        
        Args:
            model_name: Registered model name
            num_users: Number of unique users
            num_items: Number of unique items
            **config_overrides: Override config values
            
        Returns:
            Instantiated model
        """
        # Try to get config from registry
        config = ModelRegistry.get_config(model_name)
        
        if config:
            return ModelFactory.create_from_config(
                config, num_users, num_items, **config_overrides
            )
        
        # Fall back to direct instantiation with defaults
        model_class = ModelRegistry.get_model_class(model_name)
        default_config = {
            "num_users": num_users,
            "num_items": num_items,
            **config_overrides
        }
        
        return model_class(default_config)


def discover_models(models_dir: str = "app/models") -> None:
    """
    Discover and import all model modules to trigger registration.
    """
    models_path = Path(models_dir)
    
    if not models_path.exists():
        return
    
    for py_file in models_path.glob("*.py"):
        if py_file.stem.startswith("_"):
            continue
        
        module_name = f"app.models.{py_file.stem}"
        try:
            importlib.import_module(module_name)
        except Exception as e:
            print(f"Warning: Failed to import {module_name}: {e}")


def initialize_registry(config_dir: str = "model_configs") -> None:
    """
    Initialize the model registry.
    
    - Discovers and imports all model classes
    - Loads all configuration files
    """
    discover_models()
    ModelRegistry.load_configs_from_directory(config_dir)

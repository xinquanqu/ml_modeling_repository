"""
Centralized configuration management for ML Platform.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    host: str = os.getenv("POSTGRES_HOST", "postgres")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    user: str = os.getenv("POSTGRES_USER", "mlplatform")
    password: str = os.getenv("POSTGRES_PASSWORD", "mlplatform_secret")
    database: str = os.getenv("POSTGRES_DB", "mlplatform")

    @property
    def connection_string(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class RedisConfig:
    host: str = os.getenv("REDIS_HOST", "redis")
    port: int = int(os.getenv("REDIS_PORT", "6379"))

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}"


@dataclass
class MLflowConfig:
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    backend_store_uri: Optional[str] = os.getenv("MLFLOW_BACKEND_STORE_URI")
    artifact_root: str = os.getenv("MLFLOW_ARTIFACT_ROOT", "s3://mlflow-artifacts")


@dataclass
class MinIOConfig:
    endpoint: str = os.getenv("MINIO_ENDPOINT", "http://minio:9000")
    access_key: str = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")


@dataclass
class ServiceConfig:
    """Configuration for individual service."""
    name: str
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")


# Singleton configs
db_config = DatabaseConfig()
redis_config = RedisConfig()
mlflow_config = MLflowConfig()
minio_config = MinIOConfig()

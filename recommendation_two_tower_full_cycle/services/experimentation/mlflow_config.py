"""
MLflow configuration and utilities.
"""

import os


def get_mlflow_config():
    """Get MLflow configuration from environment."""
    return {
        "backend_store_uri": os.getenv(
            "MLFLOW_BACKEND_STORE_URI",
            "postgresql://mlplatform:mlplatform_secret@postgres:5432/mlflow"
        ),
        "artifact_root": os.getenv(
            "MLFLOW_ARTIFACT_ROOT",
            "s3://mlflow-artifacts"
        ),
        "s3_endpoint_url": os.getenv(
            "MLFLOW_S3_ENDPOINT_URL",
            "http://minio:9000"
        ),
    }


def setup_s3_credentials():
    """Set up S3 credentials for MLflow artifact storage."""
    # These are set via environment variables in docker-compose
    pass

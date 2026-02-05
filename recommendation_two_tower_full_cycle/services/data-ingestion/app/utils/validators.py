"""
Data validation utilities.
"""

from typing import List, Optional
import pandas as pd


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Validate a DataFrame for ingestion.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Validated DataFrame
        
    Raises:
        DataValidationError: If validation fails
    """
    if required_columns is None:
        required_columns = ["user_id", "item_id"]
    
    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise DataValidationError(
            f"Missing required columns: {missing_cols}"
        )
    
    # Check for empty DataFrame
    if len(df) == 0:
        raise DataValidationError("DataFrame is empty")
    
    # Check for null values in required columns
    for col in required_columns:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            raise DataValidationError(
                f"Column '{col}' contains {null_count} null values"
            )
    
    # Convert types
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    
    # Handle optional columns
    if "label" in df.columns:
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
    
    if "features" not in df.columns:
        df["features"] = "{}"
    
    return df


def validate_record(record: dict) -> dict:
    """
    Validate a single record.
    
    Args:
        record: Dictionary record to validate
        
    Returns:
        Validated record
        
    Raises:
        DataValidationError: If validation fails
    """
    required_fields = ["user_id", "item_id"]
    
    for field in required_fields:
        if field not in record:
            raise DataValidationError(f"Missing required field: {field}")
        if record[field] is None or str(record[field]).strip() == "":
            raise DataValidationError(f"Field '{field}' cannot be empty")
    
    # Normalize
    record["user_id"] = str(record["user_id"])
    record["item_id"] = str(record["item_id"])
    
    if "features" not in record:
        record["features"] = {}
    
    return record

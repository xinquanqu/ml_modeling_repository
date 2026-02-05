"""
API routes for data ingestion.
"""

import io
import json
import uuid
from datetime import datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter

from app.utils.database import get_db_connection, return_db_connection
from app.utils.validators import validate_dataframe, DataValidationError

router = APIRouter()

# Metrics
RECORDS_INGESTED = Counter(
    'records_ingested_total',
    'Total records ingested',
    ['source', 'status']
)


class DataRecord(BaseModel):
    """Single data record for ingestion."""
    user_id: str
    item_id: str
    features: dict = Field(default_factory=dict)
    label: Optional[float] = None
    timestamp: Optional[datetime] = None


class BatchIngestRequest(BaseModel):
    """Batch ingestion request."""
    records: List[DataRecord]
    source: str = "api"


class UploadResponse(BaseModel):
    """Response for upload operations."""
    success: bool
    batch_id: str
    records_processed: int
    errors: List[str] = Field(default_factory=list)
    message: str


@router.post("/upload/csv", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    """
    Upload a CSV file for ingestion.
    
    Required columns: user_id, item_id
    Optional columns: label, features (JSON string), timestamp
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    batch_id = str(uuid.uuid4())
    errors = []
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Validate
        try:
            df = validate_dataframe(df)
        except DataValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Insert into database
        records_inserted = await _insert_records(df, batch_id, "csv")
        
        RECORDS_INGESTED.labels(source="csv", status="success").inc(records_inserted)
        
        return UploadResponse(
            success=True,
            batch_id=batch_id,
            records_processed=records_inserted,
            errors=errors,
            message=f"Successfully ingested {records_inserted} records"
        )
        
    except Exception as e:
        RECORDS_INGESTED.labels(source="csv", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/upload/json", response_model=UploadResponse)
async def upload_json(
    file: UploadFile = File(...),
):
    """
    Upload a JSON file for ingestion.
    
    Expected format: Array of objects with user_id, item_id fields.
    """
    if not file.filename.endswith('.json'):
        raise HTTPException(status_code=400, detail="File must be a JSON")
    
    batch_id = str(uuid.uuid4())
    
    try:
        contents = await file.read()
        data = json.loads(contents)
        
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="JSON must be an array of records")
        
        df = pd.DataFrame(data)
        
        try:
            df = validate_dataframe(df)
        except DataValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        records_inserted = await _insert_records(df, batch_id, "json")
        
        RECORDS_INGESTED.labels(source="json", status="success").inc(records_inserted)
        
        return UploadResponse(
            success=True,
            batch_id=batch_id,
            records_processed=records_inserted,
            errors=[],
            message=f"Successfully ingested {records_inserted} records"
        )
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")
    except Exception as e:
        RECORDS_INGESTED.labels(source="json", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/batch", response_model=UploadResponse)
async def ingest_batch(request: BatchIngestRequest):
    """
    Ingest a batch of records via API.
    """
    batch_id = str(uuid.uuid4())
    
    try:
        # Convert to DataFrame
        records = [r.model_dump() for r in request.records]
        df = pd.DataFrame(records)
        
        records_inserted = await _insert_records(df, batch_id, request.source)
        
        RECORDS_INGESTED.labels(source="api", status="success").inc(records_inserted)
        
        return UploadResponse(
            success=True,
            batch_id=batch_id,
            records_processed=records_inserted,
            errors=[],
            message=f"Successfully ingested {records_inserted} records"
        )
        
    except Exception as e:
        RECORDS_INGESTED.labels(source="api", status="error").inc()
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/batches/{batch_id}")
async def get_batch_info(batch_id: str):
    """Get information about a specific batch."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*) as record_count, 
                       MIN(created_at) as first_record,
                       MAX(created_at) as last_record,
                       source
                FROM raw_data 
                WHERE batch_id = %s
                GROUP BY source
                """,
                (batch_id,)
            )
            result = cur.fetchone()
        return_db_connection(conn)
        
        if not result:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return {
            "batch_id": batch_id,
            "record_count": result[0],
            "first_record": result[1],
            "last_record": result[2],
            "source": result[3]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_ingestion_stats():
    """Get overall ingestion statistics."""
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT 
                    source,
                    COUNT(*) as record_count,
                    COUNT(DISTINCT batch_id) as batch_count,
                    MIN(created_at) as earliest,
                    MAX(created_at) as latest
                FROM raw_data
                GROUP BY source
                """
            )
            results = cur.fetchall()
        return_db_connection(conn)
        
        stats = []
        for row in results:
            stats.append({
                "source": row[0],
                "record_count": row[1],
                "batch_count": row[2],
                "earliest_record": row[3],
                "latest_record": row[4]
            })
        
        return {"stats": stats}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _insert_records(df: pd.DataFrame, batch_id: str, source: str) -> int:
    """Insert records into the database."""
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Prepare data for insertion
            records = []
            for _, row in df.iterrows():
                features = row.get('features', {})
                if isinstance(features, str):
                    features = json.loads(features)
                
                records.append((
                    batch_id,
                    row['user_id'],
                    row['item_id'],
                    json.dumps(features) if features else '{}',
                    row.get('label'),
                    source
                ))
            
            # Batch insert
            insert_query = """
                INSERT INTO raw_data (batch_id, user_id, item_id, features, label, source)
                VALUES %s
                ON CONFLICT (batch_id, user_id, item_id) DO UPDATE
                SET features = EXCLUDED.features,
                    label = EXCLUDED.label
            """
            from psycopg2.extras import execute_values
            execute_values(cur, insert_query, records)
            conn.commit()
            
            return len(records)
            
    finally:
        return_db_connection(conn)

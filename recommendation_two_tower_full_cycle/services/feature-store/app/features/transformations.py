"""
Feature transformation logic.

Computes features from raw data for users and items.
"""

import json
from typing import Dict, Any, List
from datetime import datetime, timedelta

from app.storage.postgres import get_db_connection


async def compute_user_features(user_id: str) -> Dict[str, Any]:
    """
    Compute features for a user based on their interaction history.
    
    Features computed:
    - interaction_count: Total number of interactions
    - avg_label: Average label/rating
    - unique_items: Number of unique items interacted with
    - recency_days: Days since last interaction
    - activity_frequency: Interactions per day (lifetime)
    """
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Get user interaction statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as interaction_count,
                    AVG(label) as avg_label,
                    COUNT(DISTINCT item_id) as unique_items,
                    MIN(created_at) as first_interaction,
                    MAX(created_at) as last_interaction
                FROM raw_data
                WHERE user_id = %s
            """, (user_id,))
            
            row = cur.fetchone()
            
            if not row or row[0] == 0:
                return {
                    "interaction_count": 0,
                    "avg_label": 0.0,
                    "unique_items": 0,
                    "recency_days": -1,
                    "activity_frequency": 0.0,
                    "is_new_user": True
                }
            
            interaction_count = row[0]
            avg_label = float(row[1]) if row[1] else 0.0
            unique_items = row[2]
            first_interaction = row[3]
            last_interaction = row[4]
            
            # Calculate derived features
            now = datetime.utcnow()
            recency_days = (now - last_interaction.replace(tzinfo=None)).days if last_interaction else -1
            
            lifetime_days = (last_interaction - first_interaction).days + 1 if first_interaction and last_interaction else 1
            activity_frequency = interaction_count / max(lifetime_days, 1)
            
            # Get category preferences
            cur.execute("""
                SELECT 
                    features->>'category' as category,
                    COUNT(*) as count
                FROM raw_data
                WHERE user_id = %s AND features->>'category' IS NOT NULL
                GROUP BY features->>'category'
                ORDER BY count DESC
                LIMIT 5
            """, (user_id,))
            
            top_categories = [row[0] for row in cur.fetchall()]
            
            return {
                "interaction_count": interaction_count,
                "avg_label": round(avg_label, 4),
                "unique_items": unique_items,
                "recency_days": recency_days,
                "activity_frequency": round(activity_frequency, 4),
                "is_new_user": interaction_count < 5,
                "top_categories": top_categories,
                "computed_at": datetime.utcnow().isoformat()
            }
            
    finally:
        conn.close()


async def compute_item_features(item_id: str) -> Dict[str, Any]:
    """
    Compute features for an item based on interaction history.
    
    Features computed:
    - interaction_count: Total number of interactions
    - avg_label: Average label/rating
    - unique_users: Number of unique users
    - popularity_score: Normalized popularity
    - recency: Days since last interaction
    """
    conn = get_db_connection()
    
    try:
        with conn.cursor() as cur:
            # Get item interaction statistics
            cur.execute("""
                SELECT 
                    COUNT(*) as interaction_count,
                    AVG(label) as avg_label,
                    COUNT(DISTINCT user_id) as unique_users,
                    MIN(created_at) as first_interaction,
                    MAX(created_at) as last_interaction
                FROM raw_data
                WHERE item_id = %s
            """, (item_id,))
            
            row = cur.fetchone()
            
            if not row or row[0] == 0:
                return {
                    "interaction_count": 0,
                    "avg_label": 0.0,
                    "unique_users": 0,
                    "popularity_score": 0.0,
                    "recency_days": -1,
                    "is_cold_start": True
                }
            
            interaction_count = row[0]
            avg_label = float(row[1]) if row[1] else 0.0
            unique_users = row[2]
            first_interaction = row[3]
            last_interaction = row[4]
            
            # Get global stats for normalization
            cur.execute("""
                SELECT MAX(item_count) FROM (
                    SELECT COUNT(*) as item_count
                    FROM raw_data
                    GROUP BY item_id
                ) t
            """)
            max_interactions = cur.fetchone()[0] or 1
            
            # Calculate derived features
            now = datetime.utcnow()
            recency_days = (now - last_interaction.replace(tzinfo=None)).days if last_interaction else -1
            popularity_score = interaction_count / max_interactions
            
            # Get item metadata from first record
            cur.execute("""
                SELECT features
                FROM raw_data
                WHERE item_id = %s
                LIMIT 1
            """, (item_id,))
            
            metadata_row = cur.fetchone()
            item_metadata = {}
            if metadata_row and metadata_row[0]:
                item_metadata = metadata_row[0] if isinstance(metadata_row[0], dict) else json.loads(metadata_row[0])
            
            return {
                "interaction_count": interaction_count,
                "avg_label": round(avg_label, 4),
                "unique_users": unique_users,
                "popularity_score": round(popularity_score, 4),
                "recency_days": recency_days,
                "is_cold_start": interaction_count < 10,
                "category": item_metadata.get("category"),
                "computed_at": datetime.utcnow().isoformat()
            }
            
    finally:
        conn.close()


async def batch_compute_features(entity_ids: List[str], entity_type: str) -> Dict[str, Dict[str, Any]]:
    """
    Batch compute features for multiple entities.
    """
    results = {}
    
    compute_fn = compute_user_features if entity_type == "user" else compute_item_features
    
    for entity_id in entity_ids:
        try:
            results[entity_id] = await compute_fn(entity_id)
        except Exception as e:
            results[entity_id] = {"error": str(e)}
    
    return results

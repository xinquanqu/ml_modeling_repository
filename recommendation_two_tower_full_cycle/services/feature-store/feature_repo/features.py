"""
Feast Entity and Feature View definitions for the ML Platform.
Compatible with Feast 0.47+
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float32, Int64
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)


# =============================================================================
# Entities
# =============================================================================

user = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity for the recommendation system",
)

item = Entity(
    name="item",
    join_keys=["item_id"],
    description="Item entity for the recommendation system",
)


# =============================================================================
# Data Sources
# =============================================================================

user_source = PostgreSQLSource(
    name="user_features_source",
    query="""
        SELECT 
            user_id,
            CAST(COUNT(*) AS BIGINT) as interaction_count,
            CAST(COALESCE(AVG(label), 0) AS REAL) as avg_rating,
            CAST(COUNT(DISTINCT item_id) AS BIGINT) as unique_items,
            CAST(EXTRACT(EPOCH FROM (NOW() - MAX(created_at))) / 86400 AS REAL) as last_active_days,
            NOW() as event_timestamp
        FROM raw_data
        GROUP BY user_id
    """,
    timestamp_field="event_timestamp",
)

item_source = PostgreSQLSource(
    name="item_features_source",
    query="""
        SELECT 
            item_id,
            CAST(COUNT(*) AS BIGINT) as interaction_count,
            CAST(COALESCE(AVG(label), 0) AS REAL) as avg_rating,
            CAST(COUNT(DISTINCT user_id) AS BIGINT) as unique_users,
            CAST(EXTRACT(EPOCH FROM (NOW() - MAX(created_at))) / 86400 AS REAL) as last_active_days,
            NOW() as event_timestamp
        FROM raw_data
        GROUP BY item_id
    """,
    timestamp_field="event_timestamp",
)


# =============================================================================
# Feature Views
# =============================================================================

user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="interaction_count", dtype=Int64),
        Field(name="avg_rating", dtype=Float32),
        Field(name="unique_items", dtype=Int64),
        Field(name="last_active_days", dtype=Float32),
    ],
    source=user_source,
    online=True,
    description="User behavioral features",
)

item_features = FeatureView(
    name="item_features",
    entities=[item],
    ttl=timedelta(days=1),
    schema=[
        Field(name="interaction_count", dtype=Int64),
        Field(name="avg_rating", dtype=Float32),
        Field(name="unique_users", dtype=Int64),
        Field(name="last_active_days", dtype=Float32),
    ],
    source=item_source,
    online=True,
    description="Item popularity features",
)

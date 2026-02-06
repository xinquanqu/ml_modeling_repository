"""
Feast Entity and Feature View definitions for the ML Platform.

This module defines the feature schema for users and items
in the recommendation system.
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, PushSource
from feast.types import Float32, Int64, String
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

# User features computed from raw interactions
user_source = PostgreSQLSource(
    name="user_features_source",
    query="""
        SELECT 
            user_id,
            COUNT(*) as interaction_count,
            COALESCE(AVG(label), 0) as avg_rating,
            COUNT(DISTINCT item_id) as unique_items,
            EXTRACT(EPOCH FROM (NOW() - MAX(created_at))) / 86400 as last_active_days,
            NOW() as event_timestamp
        FROM raw_data
        GROUP BY user_id
    """,
    timestamp_field="event_timestamp",
)

# Item features computed from raw interactions
item_source = PostgreSQLSource(
    name="item_features_source",
    query="""
        SELECT 
            item_id,
            COUNT(*) as interaction_count,
            COALESCE(AVG(label), 0) as avg_rating,
            COUNT(DISTINCT user_id) as unique_users,
            EXTRACT(EPOCH FROM (NOW() - MAX(created_at))) / 86400 as last_active_days,
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
    description="User behavioral features computed from interactions",
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
    description="Item popularity and engagement features",
)


# =============================================================================
# Push Sources for Real-time Updates
# =============================================================================

user_push_source = PushSource(
    name="user_push_source",
    batch_source=user_source,
)

item_push_source = PushSource(
    name="item_push_source",
    batch_source=item_source,
)

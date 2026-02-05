"""
Redis cache backend for online feature serving.
"""

import os
from typing import Optional

import redis.asyncio as redis

# Redis client
_redis_client: Optional[redis.Redis] = None


async def init_redis():
    """Initialize Redis connection."""
    global _redis_client
    
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    
    try:
        _redis_client = redis.Redis(
            host=host,
            port=port,
            decode_responses=True
        )
        await _redis_client.ping()
        print(f"Redis connected: {host}:{port}")
    except Exception as e:
        print(f"Failed to connect to Redis: {e}")
        raise


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    global _redis_client
    
    if _redis_client is None:
        await init_redis()
    
    return _redis_client


async def close_redis():
    """Close Redis connection."""
    global _redis_client
    
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None

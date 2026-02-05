"""
Database utilities for data ingestion service.
"""

import os
from typing import Optional

import psycopg2
from psycopg2 import pool

# Connection pool
_connection_pool: Optional[pool.ThreadedConnectionPool] = None


def get_db_config() -> dict:
    """Get database configuration from environment."""
    return {
        "host": os.getenv("POSTGRES_HOST", "postgres"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "mlplatform"),
        "password": os.getenv("POSTGRES_PASSWORD", "mlplatform_secret"),
        "database": os.getenv("POSTGRES_DB", "mlplatform"),
    }


def init_db_pool(min_conn: int = 2, max_conn: int = 10):
    """Initialize the database connection pool."""
    global _connection_pool
    
    config = get_db_config()
    
    try:
        _connection_pool = pool.ThreadedConnectionPool(
            min_conn,
            max_conn,
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"]
        )
        print(f"Database connection pool initialized: {config['host']}:{config['port']}")
    except Exception as e:
        print(f"Failed to initialize database pool: {e}")
        raise


def get_db_connection():
    """Get a database connection from the pool."""
    global _connection_pool
    
    if _connection_pool is None:
        # Fallback to direct connection if pool not initialized
        config = get_db_config()
        return psycopg2.connect(
            host=config["host"],
            port=config["port"],
            user=config["user"],
            password=config["password"],
            database=config["database"]
        )
    
    return _connection_pool.getconn()


def return_db_connection(conn):
    """Return a connection to the pool."""
    global _connection_pool
    
    if _connection_pool is not None:
        _connection_pool.putconn(conn)
    else:
        conn.close()


def close_db_pool():
    """Close all connections in the pool."""
    global _connection_pool
    
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None

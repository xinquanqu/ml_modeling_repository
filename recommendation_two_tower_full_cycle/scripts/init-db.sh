#!/bin/bash

# =============================================================================
# Database Initialization Script
# =============================================================================

echo "🗄️  Initializing database..."

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL..."
until docker-compose exec -T postgres pg_isready -U mlplatform; do
    sleep 2
done

echo "✅ Database is ready!"

# Run the init SQL
docker-compose exec -T postgres psql -U mlplatform -d mlplatform -f /docker-entrypoint-initdb.d/init.sql

echo "✅ Database initialized successfully!"

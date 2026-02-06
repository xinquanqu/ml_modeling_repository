#!/bin/bash
# Startup script that generates Feast config from environment variables

set -e

cd /app

cat > /app/feature_repo/feature_store.yaml << EOF
project: ml_platform_features
provider: local
registry: /app/feature_repo/data/registry.db
online_store:
  type: redis
  redis_type: redis
  connection_string: ${REDIS_HOST:-redis}:6379
offline_store:
  type: postgres
  host: ${POSTGRES_HOST:-postgres}
  port: 5432
  database: ${POSTGRES_DB:-mlplatform}
  db_schema: public
  user: ${POSTGRES_USER:-mlplatform}
  password: ${POSTGRES_PASSWORD:-mlplatform_secret}
entity_key_serialization_version: 2
EOF

echo "Feast config generated from environment variables"

# Apply feature definitions
echo "Applying Feast feature definitions..."
feast -c /app/feature_repo apply 2>&1 || echo "Feast apply failed (may be waiting for DB)"

# Start the FastAPI application
exec uvicorn app.main:app --host 0.0.0.0 --port 8002

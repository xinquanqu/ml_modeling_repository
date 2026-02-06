#!/bin/bash
# Startup script that generates Feast config from environment variables

cat > /app/feature_repo/feature_store.yaml << EOF
project: ml_platform_features
provider: local
registry: /app/feature_repo/data/registry.db
online_store:
  type: redis
  connection_string: redis://${REDIS_HOST:-redis}:6379
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

# Start the FastAPI application
exec uvicorn app.main:app --host 0.0.0.0 --port 8002

#!/bin/bash

# =============================================================================
# ML Platform Start Script
# =============================================================================

set -e

echo "🚀 Starting ML Platform..."

# Check if .env exists, if not copy from example
if [ ! -f .env ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
fi

# Create required directories
mkdir -p services/observability/grafana/dashboards
mkdir -p services/observability/grafana/provisioning/datasources
mkdir -p services/observability/grafana/provisioning/dashboards

# Build and start all services
echo "🔨 Building services..."
docker-compose build

echo "▶️  Starting services..."
docker-compose up -d

echo ""
echo "✅ ML Platform started successfully!"
echo ""
echo "📊 Service URLs:"
echo "   - Data Ingestion: http://localhost:8001"
echo "   - Feature Store:  http://localhost:8002"
echo "   - Training:       http://localhost:8003"
echo "   - Model Serving:  http://localhost:8004"
echo "   - MLflow UI:      http://localhost:5000"
echo "   - Grafana:        http://localhost:3000 (admin/admin)"
echo "   - Prometheus:     http://localhost:9090"
echo "   - MinIO Console:  http://localhost:9001 (minioadmin/minioadmin123)"
echo ""
echo "📋 Useful commands:"
echo "   docker-compose logs -f           # View all logs"
echo "   docker-compose ps                # Check service status"
echo "   ./scripts/stop.sh                # Stop all services"
echo ""

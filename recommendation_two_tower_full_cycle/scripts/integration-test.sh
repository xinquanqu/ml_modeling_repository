#!/bin/bash

# =============================================================================
# Integration Test Script
# =============================================================================

set -e

echo "🧪 Running ML Platform Integration Tests..."
echo ""

BASE_URL="http://localhost"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_service() {
    local name=$1
    local url=$2
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "  ✅ ${GREEN}$name${NC}"
        return 0
    else
        echo -e "  ❌ ${RED}$name${NC}"
        return 1
    fi
}

# 1. Check all services are running
echo "📍 Step 1: Checking service health..."
check_service "Data Ingestion" "$BASE_URL:8001/health"
check_service "Feature Store" "$BASE_URL:8002/health"
check_service "Training" "$BASE_URL:8003/health"
check_service "Model Serving" "$BASE_URL:8004/health"
check_service "MLflow" "$BASE_URL:5000"
check_service "Prometheus" "$BASE_URL:9090/-/healthy"
check_service "Grafana" "$BASE_URL:3000/api/health"
echo ""

# 2. Test data ingestion
echo "📍 Step 2: Testing data ingestion..."
BATCH_RESPONSE=$(curl -s -X POST "$BASE_URL:8001/api/v1/ingest/batch" \
    -H "Content-Type: application/json" \
    -d '{
        "records": [
            {"user_id": "user_1", "item_id": "item_1", "label": 1.0},
            {"user_id": "user_1", "item_id": "item_2", "label": 0.0},
            {"user_id": "user_2", "item_id": "item_1", "label": 1.0}
        ],
        "source": "integration_test"
    }')

if echo "$BATCH_RESPONSE" | grep -q '"success":true'; then
    echo -e "  ✅ ${GREEN}Data ingestion successful${NC}"
else
    echo -e "  ❌ ${RED}Data ingestion failed${NC}"
    echo "  Response: $BATCH_RESPONSE"
fi
echo ""

# 3. Test feature store
echo "📍 Step 3: Testing feature store..."
FEATURE_RESPONSE=$(curl -s -X POST "$BASE_URL:8002/api/v1/set" \
    -H "Content-Type: application/json" \
    -d '{
        "entity_id": "user_1",
        "entity_type": "user",
        "features": {"age": 25, "premium": true}
    }')

if echo "$FEATURE_RESPONSE" | grep -q '"success":true'; then
    echo -e "  ✅ ${GREEN}Feature store write successful${NC}"
else
    echo -e "  ❌ ${RED}Feature store write failed${NC}"
fi

GET_FEATURE=$(curl -s "$BASE_URL:8002/api/v1/user/user_1")
if echo "$GET_FEATURE" | grep -q '"entity_id":"user_1"'; then
    echo -e "  ✅ ${GREEN}Feature store read successful${NC}"
else
    echo -e "  ❌ ${RED}Feature store read failed${NC}"
fi
echo ""

# 4. Check training service API
echo "📍 Step 4: Testing training service..."
JOBS_RESPONSE=$(curl -s "$BASE_URL:8003/api/v1/jobs")
if echo "$JOBS_RESPONSE" | grep -q '\['; then
    echo -e "  ✅ ${GREEN}Training jobs endpoint working${NC}"
else
    echo -e "  ❌ ${RED}Training jobs endpoint failed${NC}"
fi
echo ""

# 5. Check serving service
echo "📍 Step 5: Testing serving service..."
MODELS_RESPONSE=$(curl -s "$BASE_URL:8004/api/v1/models")
if echo "$MODELS_RESPONSE" | grep -q '"models"'; then
    echo -e "  ✅ ${GREEN}Model serving endpoint working${NC}"
else
    echo -e "  ❌ ${RED}Model serving endpoint failed${NC}"
fi
echo ""

# 6. Check metrics endpoints
echo "📍 Step 6: Checking Prometheus metrics..."
PROM_TARGETS=$(curl -s "$BASE_URL:9090/api/v1/targets" | grep -c '"health":"up"')
echo "  📊 Active Prometheus targets: $PROM_TARGETS"
echo ""

echo "🎉 Integration tests completed!"

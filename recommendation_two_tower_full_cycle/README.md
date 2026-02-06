# ML Platform Microservices

A comprehensive machine learning platform with microservices architecture for the full ML lifecycle.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ML Platform                               │
├─────────────────────────────────────────────────────────────────┤
│  Data Layer          │  ML Layer           │  Serving Layer     │
│  ─────────────────   │  ──────────────     │  ────────────────  │
│  • Data Ingestion    │  • Training         │  • Model Serving   │
│  • Data Warehouse    │  • Experimentation  │    (FastAPI+KServe)│
│  • Feature Store     │    (MLflow)         │                    │
├─────────────────────────────────────────────────────────────────┤
│                    Observability (Grafana + Prometheus)          │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Copy environment file
cp .env.example .env

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Data Ingestion | 8001 | File upload & streaming |
| Data Warehouse | 5433 | PostgreSQL database |
| Feature Store | 8002 | Feast-powered feature serving |
| Training | 8003 | PyTorch training jobs |
| MLflow | 5001 | Experiment tracking UI |
| Model Serving | 8004 | Inference API |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards (admin/admin) |

---

## 🚀 End-to-End Training Example

This example walks through the complete ML lifecycle: **Data Ingestion → Feature Store → Training → Serving**.

### Step 1: Ingest Data

Upload training data to the data warehouse:

```bash
# Upload sample CSV data
curl -X POST "http://localhost:8001/api/v1/upload/csv" \
  -F "file=@data/sample_data.csv"

# Or ingest batch via API
curl -X POST "http://localhost:8001/api/v1/ingest/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "source": "api",
    "records": [
      {"user_id": "user_001", "item_id": "item_101", "label": 1.0},
      {"user_id": "user_001", "item_id": "item_102", "label": 0.0},
      {"user_id": "user_002", "item_id": "item_101", "label": 1.0}
    ]
  }'
```

### Step 2: Materialize Features (Feast)

Push features to the online store (Redis):

```bash
# Materialize all features to online store
curl -X POST "http://localhost:8002/api/v1/materialize" \
  -H "Content-Type: application/json" \
  -d '{}'

# Get user features
curl "http://localhost:8002/api/v1/user/user_001"

# Get item features
curl "http://localhost:8002/api/v1/item/item_101"

# Check feature store registry
curl "http://localhost:8002/api/v1/registry"
```

### Step 3: Train a Model

Choose a model and start training:

```bash
# List available models
curl "http://localhost:8003/api/v1/models"

# Train Two-Tower model (default)
curl -X POST "http://localhost:8003/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model_name": "two_tower",
      "epochs": 10,
      "learning_rate": 0.001,
      "batch_size": 256,
      "experiment_name": "recommendation_v1"
    }
  }'

# Train Neural Collaborative Filtering
curl -X POST "http://localhost:8003/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "config": {
      "model_name": "ncf",
      "epochs": 15,
      "experiment_name": "ncf_experiment"
    }
  }'

# Check training status
curl "http://localhost:8003/api/v1/jobs/{job_id}"
```

### Step 4: View Experiments in MLflow

Open MLflow UI to compare experiments:

```bash
open http://localhost:5001
```

- Compare metrics across runs
- View training curves
- Download model artifacts

### Step 5: Load Model for Serving

Deploy the trained model:

```bash
# Load model from MLflow
curl -X POST "http://localhost:8004/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_uri": "runs:/{run_id}/model",
    "model_name": "recommendation_v1"
  }'

# Set as active model
curl -X POST "http://localhost:8004/api/v1/models/recommendation_v1/activate"
```

### Step 6: Get Predictions

Make inference requests:

```bash
# Get recommendations for a user
curl -X POST "http://localhost:8004/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "num_items": 10
  }'

# Batch predictions
curl -X POST "http://localhost:8004/api/v1/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"user_id": 1, "item_id": 101},
      {"user_id": 1, "item_id": 102},
      {"user_id": 2, "item_id": 101}
    ]
  }'

# KServe-compatible endpoint
curl -X POST "http://localhost:8004/api/v1/v2/models/recommendation/infer" \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [{"name": "user_id", "data": [1]}, {"name": "item_ids", "data": [101, 102, 103]}]
  }'
```

---

## Available Models

| Model | Config | Description |
|-------|--------|-------------|
| `two_tower` | `model_configs/two_tower.yaml` | Dual-tower with separate user/item encoders |
| `matrix_factorization` | `model_configs/matrix_factorization.yaml` | Classic MF baseline |
| `ncf` | `model_configs/ncf.yaml` | Neural CF (GMF + MLP) |

### Adding a Custom Model

1. Create model class inheriting `BaseRecommendationModel`:
```python
from app.models.base import BaseRecommendationModel
from app.registry import ModelRegistry

@ModelRegistry.register("my_model")
class MyModel(BaseRecommendationModel):
    def forward(self, user_ids, item_ids, ...):
        ...
```

2. Add YAML config in `model_configs/my_model.yaml`

3. Restart training service

---

## Directory Structure

```
├── docker-compose.yml      # Orchestration
├── services/
│   ├── data-ingestion/     # FastAPI data service
│   ├── data-warehouse/     # PostgreSQL + migrations
│   ├── feature-store/      # Feature engineering
│   ├── training/           # PyTorch training + model configs
│   │   ├── app/
│   │   │   ├── models/     # Model implementations
│   │   │   ├── registry/   # Model registry
│   │   │   ├── data/       # Data pipeline
│   │   │   └── evaluation/ # Metrics
│   │   └── model_configs/  # YAML model definitions
│   ├── experimentation/    # MLflow server
│   ├── serving/            # FastAPI + KServe
│   └── observability/      # Grafana + Prometheus
├── shared/                 # Shared utilities
├── scripts/                # Helper scripts
└── data/                   # Sample data
```

## Development

```bash
# Rebuild training service after code changes
docker-compose up -d --build training

# View logs
docker-compose logs -f training

# Run integration tests
./scripts/integration-test.sh

# Stop all services
docker-compose down
```

## Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **MLflow**: http://localhost:5001

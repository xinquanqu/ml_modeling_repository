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
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| Data Ingestion | 8001 | File upload & streaming |
| Data Warehouse | 5432 | PostgreSQL database |
| Feature Store | 8002 | Feature computation & storage |
| Training | 8003 | PyTorch training jobs |
| MLflow | 5000 | Experiment tracking UI |
| Model Serving | 8004 | Inference API |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards (admin/admin) |

## Directory Structure

```
├── docker-compose.yml      # Orchestration
├── services/
│   ├── data-ingestion/     # FastAPI data service
│   ├── data-warehouse/     # PostgreSQL + migrations
│   ├── feature-store/      # Feature engineering
│   ├── training/           # PyTorch training
│   ├── experimentation/    # MLflow server
│   ├── serving/            # FastAPI + KServe
│   └── observability/      # Grafana + Prometheus
├── shared/                 # Shared utilities
└── scripts/                # Helper scripts
```

## Development

```bash
# Run tests
docker-compose -f docker-compose.test.yml up --build

# Restart single service
docker-compose restart training

# Rebuild after code changes
docker-compose up -d --build training
```

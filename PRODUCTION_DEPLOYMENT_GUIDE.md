# Production Deployment Guide - Nuru Yield Prediction API

This guide covers deploying the yield prediction model for real-time production use.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Deployment Steps](#deployment-steps)
4. [Model Promotion](#model-promotion)
5. [API Usage](#api-usage)
6. [Monitoring & Observability](#monitoring--observability)
7. [Scaling & Performance](#scaling--performance)
8. [Security Considerations](#security-considerations)
9. [CI/CD Pipeline](#cicd-pipeline)
10. [Troubleshooting](#troubleshooting)

---

## Architecture Overview

### Production Components

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT APPLICATIONS                    │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│              INFERENCE API (FastAPI)                      │
│  - Model Loading (MLflow)                                │
│  - Feature Serving (S3/MinIO or Feast)                   │
│  - Prediction Endpoints                                  │
│  - Health Checks                                         │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   MLflow     │   │   MinIO/S3   │   │  PostgreSQL  │
│  Registry    │   │  (Features) │   │  (Metadata)  │
└──────────────┘   └──────────────┘   └──────────────┘
```

### Key Components

1. **Inference API Service** (`inference-api`)
   - FastAPI application
   - Loads models from MLflow registry
   - Serves features from S3/MinIO
   - Provides REST endpoints for predictions

2. **Model Registry** (MLflow)
   - Stores trained models
   - Version management
   - Stage transitions (Staging → Production)

3. **Feature Store** (S3/MinIO)
   - Pre-computed plot features
   - Historical data access
   - Optional: Feast for online serving

---

## Prerequisites

### 1. Trained Models in MLflow

Ensure you have trained models logged to MLflow:

```bash
# Check MLflow UI: http://localhost:5000
# Verify experiment "YieldPrediction_MultiModel" has runs
```

### 2. Features Available in S3/MinIO

Ensure plot features are materialized:

```bash
# Check S3/MinIO for:
# nuru-yield/features/{YYYY-MM-DD}/plot_features.parquet
```

### 3. Environment Variables

Create/update `.env` file:

```bash
# MLflow
MLFLOW_TRACKING_URI=http://mlflow-server:5000
MLFLOW_DB_PASSWORD=your_password

# Storage
S3_BUCKET_NAME=nuru-yield
S3_ENDPOINT_URL=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# API (optional)
API_KEY=your_api_key  # For authentication
```

---

## Deployment Steps

### Step 1: Promote Model to Production

Before deploying, promote your best model to Production stage in MLflow:

```bash
# Option 1: Using script
python scripts/promote_model_to_production.py \
    --run-id <your_run_id> \
    --model-name Ensemble_VotingRegressor \
    --stage Production

# Option 2: Using MLflow UI
# 1. Go to http://localhost:5000
# 2. Navigate to experiment "YieldPrediction_MultiModel"
# 3. Select best run
# 4. Click "Register Model" → Create new "Ensemble_VotingRegressor"
# 5. Transition version to "Production" stage
```

### Step 2: Start Inference Service

```bash
# Start all services including inference API
docker-compose up -d

# Or start only inference API
docker-compose up -d inference-api

# Check logs
docker-compose logs -f inference-api
```

### Step 3: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "models_loaded": ["ensemble", "xgboost", "lightgbm", ...],
#   "mlflow_connected": true,
#   "feature_store_available": true,
#   "timestamp": "2025-12-09T..."
# }
```

---

## Model Promotion

### Automatic Promotion (Recommended)

Add to your training DAG to auto-promote best model:

```python
# In model_training_dag.py, after training:
from mlflow.tracking import MlflowClient

client = MlflowClient()
best_model_run_id = ensemble_run.info.run_id

# Register and promote
mv = client.create_model_version(
    name="Ensemble_VotingRegressor",
    source=f"runs:/{best_model_run_id}/ensemble_model",
    run_id=best_model_run_id
)

client.transition_model_version_stage(
    name="Ensemble_VotingRegressor",
    version=mv.version,
    stage="Production"
)
```

### Manual Promotion

Use the provided script:

```bash
python scripts/promote_model_to_production.py \
    --run-id abc123def456 \
    --model-name Ensemble_VotingRegressor
```

---

## API Usage

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "plot_id": 1,
    "latitude": -0.499127,
    "longitude": 37.612253,
    "planting_date": "2021-10-07",
    "season": "Short Rains",
    "altitude": 1252.08,
    "model_name": "ensemble"
  }'
```

**Response:**
```json
{
  "plot_id": 1,
  "predicted_yield_kg_per_ha": 2450.75,
  "model_name": "ensemble",
  "model_version": "abc123def456",
  "confidence_score": null,
  "features_used": {
    "precip_total": 450.2,
    "gdd_sum": 1200.5,
    "mean_ndvi": 0.65,
    ...
  },
  "timestamp": "2025-12-09T10:30:00Z"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "plots": [
      {
        "plot_id": 1,
        "latitude": -0.499127,
        "longitude": 37.612253,
        "planting_date": "2021-10-07",
        "season": "Short Rains"
      },
      {
        "plot_id": 2,
        "latitude": -0.501234,
        "longitude": 37.614567,
        "planting_date": "2021-10-10",
        "season": "Short Rains"
      }
    ],
    "model_name": "ensemble"
  }'
```

### List Available Models

```bash
curl http://localhost:8000/models
```

### Reload Models

```bash
curl -X POST http://localhost:8000/models/reload
```

---

## Monitoring & Observability

### 1. Health Checks

Monitor API health:

```bash
# Kubernetes/Docker health check
curl http://localhost:8000/health

# Prometheus metrics (if added)
curl http://localhost:8000/metrics
```

### 2. Logging

API logs are available via:

```bash
# Docker logs
docker-compose logs -f inference-api

# Log files (if configured)
tail -f logs/inference_api.log
```

### 3. MLflow Model Monitoring

- Track model performance in production
- Monitor prediction distributions
- Set up alerts for model drift

### 4. Application Performance Monitoring (APM)

Recommended tools:
- **Prometheus + Grafana** - Metrics and dashboards
- **ELK Stack** - Log aggregation
- **Sentry** - Error tracking

---

## Scaling & Performance

### Horizontal Scaling

```yaml
# docker-compose.yml
inference-api:
  deploy:
    replicas: 3
  # Or use Kubernetes:
  # kubectl scale deployment inference-api --replicas=3
```

### Load Balancing

Use nginx or Traefik:

```nginx
# nginx.conf
upstream inference_api {
    server inference-api-1:8000;
    server inference-api-2:8000;
    server inference-api-3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://inference_api;
    }
}
```

### Caching

- **Model Caching**: Models loaded in memory (already implemented)
- **Feature Caching**: Redis for frequently accessed features
- **Response Caching**: Cache predictions for same inputs

### Performance Optimization

1. **Async Processing**: Use FastAPI async endpoints for I/O-bound operations
2. **Batch Processing**: Use `/predict/batch` for multiple predictions
3. **Model Optimization**: Consider ONNX conversion for faster inference
4. **Feature Pre-computation**: Pre-compute features to reduce latency

---

## Security Considerations

### 1. Authentication

Add API key authentication:

```python
# In src/inference/api.py
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != os.environ.get("API_KEY"):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

@app.post("/predict")
async def predict_yield(request: PredictionRequest, api_key: str = Depends(verify_api_key)):
    ...
```

### 2. Rate Limiting

Add rate limiting middleware:

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/predict")
@limiter.limit("10/minute")
async def predict_yield(request: Request, ...):
    ...
```

### 3. Input Validation

Already implemented via Pydantic schemas, but add additional checks:

```python
# Validate feature ranges
if request.latitude < -90 or request.latitude > 90:
    raise HTTPException(status_code=400, detail="Invalid latitude")
```

### 4. HTTPS/TLS

Use reverse proxy (nginx/Traefik) with SSL certificates:

```nginx
server {
    listen 443 ssl;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ...
}
```

### 5. Network Security

- Use private networks in Docker
- Restrict API access via firewall rules
- Use VPN for internal access

---

## CI/CD Pipeline

### GitHub Actions Example

```yaml
# .github/workflows/deploy.yml
name: Deploy Inference API

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -f Dockerfile.inference -t inference-api:${{ github.sha }} .
      
      - name: Run tests
        run: pytest tests/
      
      - name: Deploy to production
        run: |
          docker tag inference-api:${{ github.sha }} inference-api:latest
          docker-compose up -d inference-api
```

### Model Deployment Workflow

1. **Train Model** → MLflow experiment
2. **Evaluate** → Compare metrics
3. **Promote** → Register to Production stage
4. **Deploy** → Restart inference API (auto-reloads models)
5. **Monitor** → Track performance and drift

---

## Production Checklist

### Pre-Deployment

- [ ] Models trained and logged to MLflow
- [ ] Best model promoted to Production stage
- [ ] Features materialized in S3/MinIO
- [ ] Environment variables configured
- [ ] Health checks passing
- [ ] Load testing completed
- [ ] Security measures implemented

### Deployment

- [ ] Inference API service started
- [ ] Models loaded successfully
- [ ] API endpoints responding
- [ ] Health check endpoint working
- [ ] Logs accessible and monitored

### Post-Deployment

- [ ] Monitor prediction latency
- [ ] Track prediction distributions
- [ ] Set up alerts for failures
- [ ] Document API usage
- [ ] Train team on API usage

---

## Troubleshooting

### Models Not Loading

**Problem:** API starts but models not loaded

**Solutions:**
1. Check MLflow connection:
   ```bash
   curl http://mlflow-server:5000/health
   ```

2. Verify model exists in registry:
   ```python
   from mlflow.tracking import MlflowClient
   client = MlflowClient()
   client.get_latest_versions("Ensemble_VotingRegressor", stages=["Production"])
   ```

3. Check S3/MinIO credentials in environment

### Features Not Found

**Problem:** 404 errors for plot features

**Solutions:**
1. Verify features exist in S3:
   ```bash
   # Check MinIO UI: http://localhost:9001
   # Navigate to: nuru-yield/features/
   ```

2. Run feature materialization DAG:
   ```bash
   # Trigger feature_materialization DAG in Airflow
   ```

3. Check feature serving logs:
   ```bash
   docker-compose logs inference-api | grep "feature"
   ```

### High Latency

**Problem:** Predictions taking too long

**Solutions:**
1. Use batch endpoint for multiple predictions
2. Enable feature caching (Redis)
3. Scale horizontally (multiple API instances)
4. Optimize model (ONNX conversion)

### Memory Issues

**Problem:** API running out of memory

**Solutions:**
1. Load only ensemble model (not all models)
2. Increase container memory limits
3. Use model quantization
4. Implement lazy loading

---

## Advanced Features

### 1. A/B Testing

Test multiple model versions:

```python
# Route traffic between models
if plot_id % 2 == 0:
    model_name = "ensemble_v1"
else:
    model_name = "ensemble_v2"
```

### 2. Shadow Mode

Run new model alongside production without affecting results:

```python
# Make predictions with both models
prod_pred = model_loader.predict(X, "ensemble")
shadow_pred = model_loader.predict(X, "ensemble_v2")
# Log both, compare later
```

### 3. Feature Store Integration (Feast)

Enable Feast for real-time feature serving:

```python
from feast import FeatureStore

fs = FeatureStore(repo_path="src/feast_repo")
features = fs.get_online_features(...)
```

### 4. Prediction Monitoring

Log all predictions for analysis:

```python
# Log to S3/MinIO or database
prediction_log = {
    "plot_id": plot_id,
    "prediction": predicted_yield,
    "features": features_dict,
    "timestamp": datetime.utcnow()
}
# Save to monitoring/predictions/{date}/predictions.parquet
```

---

## Next Steps

1. **Set up monitoring dashboards** (Grafana)
2. **Implement automated retraining** triggers
3. **Add prediction confidence intervals**
4. **Set up alerting** for model drift
5. **Create API documentation** (Swagger/OpenAPI)
6. **Implement feature store** (Feast online serving)
7. **Add model explainability** (SHAP values)

---

## Support

For issues or questions:
- Check logs: `docker-compose logs inference-api`
- Review MLflow UI: http://localhost:5000
- Check Airflow DAGs for data pipeline issues

---

**End of Production Deployment Guide**


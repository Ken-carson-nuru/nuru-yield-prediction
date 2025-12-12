# Quick Start: Production Deployment

This is a quick guide to get your inference API running in production.

## 🚀 Quick Start (5 minutes)

### 1. Ensure Models are Trained

```bash
# Check MLflow UI: http://localhost:5000
# Verify you have runs in "YieldPrediction_MultiModel" experiment
```

### 2. Promote Best Model to Production

**Option A: Using MLflow UI (Easiest)**
1. Go to http://localhost:5000
2. Navigate to experiment "YieldPrediction_MultiModel"
3. Find your best run (lowest RMSE)
4. Click "Register Model" → Create "Ensemble_VotingRegressor"
5. Click on the model → Transition version to "Production"

**Option B: Using Script**
```bash
# Find your run ID from MLflow UI
python scripts/promote_model_to_production.py \
    --run-id <your_run_id> \
    --model-name Ensemble_VotingRegressor
```

### 3. Start Inference API

```bash
# Start all services
docker-compose up -d

# Or just the inference API
docker-compose up -d inference-api

# Check it's running
docker-compose ps inference-api
```

### 4. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Make a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "plot_id": 1,
    "latitude": -0.499127,
    "longitude": 37.612253,
    "planting_date": "2021-10-07",
    "season": "Short Rains",
    "altitude": 1252.08
  }'
```

### 5. Run Test Suite

```bash
# Install requests if needed
pip install requests

# Run tests
python scripts/test_inference_api.py
```

## 📋 What Was Created

### New Files

1. **`src/inference/api.py`** - FastAPI inference service
2. **`src/inference/model_loader.py`** - MLflow model loading
3. **`src/inference/feature_serving.py`** - Feature retrieval from S3
4. **`Dockerfile.inference`** - Docker image for API
5. **`scripts/promote_model_to_production.py`** - Model promotion script
6. **`scripts/test_inference_api.py`** - API test suite
7. **`PRODUCTION_DEPLOYMENT_GUIDE.md`** - Full deployment guide

### Updated Files

1. **`docker-compose.yml`** - Added `inference-api` service

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/models` | GET | List available models |
| `/models/reload` | POST | Reload models from MLflow |

## 📝 Example Usage

### Python Client

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "plot_id": 1,
        "latitude": -0.499127,
        "longitude": 37.612253,
        "planting_date": "2021-10-07",
        "season": "Short Rains",
        "altitude": 1252.08
    }
)
print(response.json())
# {
#   "plot_id": 1,
#   "predicted_yield_kg_per_ha": 2450.75,
#   "model_name": "ensemble",
#   "timestamp": "2025-12-09T10:30:00Z"
# }
```

### JavaScript/TypeScript Client

```javascript
const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    plot_id: 1,
    latitude: -0.499127,
    longitude: 37.612253,
    planting_date: '2021-10-07',
    season: 'Short Rains',
    altitude: 1252.08
  })
});

const result = await response.json();
console.log(result.predicted_yield_kg_per_ha);
```

## 🐛 Troubleshooting

### API not starting?

```bash
# Check logs
docker-compose logs inference-api

# Common issues:
# - MLflow not accessible: Check MLFLOW_TRACKING_URI
# - Models not found: Promote model to Production first
# - S3/MinIO connection: Check credentials in .env
```

### Models not loading?

```bash
# Verify model exists in MLflow
curl http://localhost:5000/api/2.0/mlflow/model-versions/search?name=Ensemble_VotingRegressor

# Check model loader logs
docker-compose logs inference-api | grep "model"
```

### Features not found?

```bash
# Ensure features are materialized
# Trigger feature_materialization_dag in Airflow

# Check S3/MinIO
# Go to http://localhost:9001
# Navigate to: nuru-yield/features/
```

## 📚 Next Steps

1. **Read Full Guide**: See `PRODUCTION_DEPLOYMENT_GUIDE.md` for:
   - Security (authentication, rate limiting)
   - Scaling (horizontal scaling, load balancing)
   - Monitoring (metrics, logging, alerts)
   - CI/CD pipeline setup

2. **Add Authentication**: Implement API keys (see guide)

3. **Set Up Monitoring**: Add Prometheus/Grafana

4. **Enable Caching**: Add Redis for feature caching

5. **Scale Horizontally**: Deploy multiple API instances

## 🎯 Production Checklist

- [ ] Model promoted to Production in MLflow
- [ ] Features materialized in S3/MinIO
- [ ] API health check passing
- [ ] Test predictions working
- [ ] Logs accessible
- [ ] Environment variables configured
- [ ] Security measures implemented (see full guide)

---

**For detailed information, see `PRODUCTION_DEPLOYMENT_GUIDE.md`**


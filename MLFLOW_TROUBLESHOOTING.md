# MLflow Permission Error Fix Guide

## Problem

When trying to register a model in MLflow UI, you get:
```
Error registering model Unable to fetch model from model URI source artifact location 'models:/m-622a3f83324c43368ad2747097289e36'.
Error: [Errno 13] Permission denied: './mlruns'
```

## Root Cause

The MLflow client is trying to use a **local `./mlruns` directory** instead of the **remote MLflow server** with S3/MinIO storage. This happens when:

1. MLflow client is not configured to use the remote tracking URI
2. Local `./mlruns` directory exists and conflicts
3. S3/MinIO credentials are not set before model registration

## Solutions

### Solution 1: Use the Registration Script (Recommended)

Use the provided script that ensures proper configuration:

```bash
# Set environment variables first
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Register ensemble model
python scripts/register_model_fix.py \
    --run-id <your_run_id> \
    --model-name Ensemble_VotingRegressor \
    --artifact-path ensemble_model \
    --stage Production
```

### Solution 2: Fix MLflow UI Registration

If you want to use the MLflow UI, you need to ensure your browser/client is configured correctly:

#### Step 1: Remove Local mlruns Directory

```bash
# If you have a local mlruns directory, remove or rename it
rm -rf ./mlruns
# Or on Windows:
rmdir /s /q mlruns
```

#### Step 2: Set Environment Variables

**Before opening MLflow UI**, set these environment variables:

```bash
# Windows PowerShell
$env:MLFLOW_TRACKING_URI="http://localhost:5000"
$env:MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"
$env:AWS_ACCESS_KEY_ID="minioadmin"
$env:AWS_SECRET_ACCESS_KEY="minioadmin"

# Linux/Mac
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
```

#### Step 3: Register Model via MLflow UI

1. Go to http://localhost:5000
2. Navigate to your experiment
3. Click on the run you want to register
4. Scroll to "Artifacts" section
5. Find your model (e.g., `ensemble_model/`)
6. Click on the model folder
7. Click "Register Model" button
8. Create new model name or select existing
9. The model should now register to S3/MinIO

### Solution 3: Use MLflow CLI

```bash
# Set environment
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Register model
mlflow models register-model \
    --model-uri "runs:/<run_id>/ensemble_model" \
    --name "Ensemble_VotingRegressor"

# Transition to Production
mlflow models transition-stage \
    --name "Ensemble_VotingRegressor" \
    --version 1 \
    --stage Production
```

### Solution 4: Register from Python (In Jupyter/Notebook)

```python
import os
import mlflow
from mlflow.tracking import MlflowClient

# CRITICAL: Set these BEFORE any MLflow operations
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ACCESS_KEY_ID"] = "minioadmin"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minioadmin"

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Register model
client = MlflowClient()
run_id = "your_run_id_here"

mv = client.create_model_version(
    name="Ensemble_VotingRegressor",
    source=f"runs:/{run_id}/ensemble_model",
    run_id=run_id
)

# Promote to Production
client.transition_model_version_stage(
    name="Ensemble_VotingRegressor",
    version=mv.version,
    stage="Production"
)

print(f"Model registered! Version: {mv.version}")
```

## Verification

After registration, verify the model is in the registry:

```bash
# Using script
python scripts/register_model_fix.py --run-id <run_id> --model-name Ensemble_VotingRegressor

# Or check in MLflow UI
# Go to http://localhost:5000 → Models → Ensemble_VotingRegressor
```

## Common Issues

### Issue 1: "Permission denied: './mlruns'"

**Fix:** 
- Remove local `./mlruns` directory
- Ensure `MLFLOW_TRACKING_URI` is set to remote server
- Set S3 credentials before MLflow operations

### Issue 2: "Model artifact not found"

**Fix:**
- Check the artifact path in the run
- Common paths: `ensemble_model`, `xgb_model`, `lgb_model`, `cat_model`, `rf_model`
- Verify the run ID is correct

### Issue 3: "Unable to fetch from S3"

**Fix:**
- Verify MinIO is running: `docker-compose ps minio`
- Check S3 credentials are correct
- Verify `MLFLOW_S3_ENDPOINT_URL` is set

### Issue 4: Model URI format error

**Wrong:** `models:/m-622a3f83324c43368ad2747097289e36`  
**Correct:** `runs:/<run_id>/<artifact_path>` or `models:/<model_name>/<stage>`

## Prevention

To prevent this issue in the future:

1. **Always set environment variables** before MLflow operations
2. **Use the registration script** instead of UI when possible
3. **Remove local mlruns directories** if they exist
4. **Verify MLflow server is accessible** before operations

## Quick Reference

```bash
# Environment setup (run before MLflow operations)
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin

# Register model
python scripts/register_model_fix.py --run-id <run_id> --model-name Ensemble_VotingRegressor

# Check registered models
mlflow models list
```

---

**If issues persist, check:**
1. MLflow server logs: `docker-compose logs mlflow-server`
2. MinIO is accessible: http://localhost:9001
3. Network connectivity between services


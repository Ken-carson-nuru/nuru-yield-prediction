# src/inference/api.py
"""
FastAPI inference service for yield prediction.
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import numpy as np
from loguru import logger
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from src.inference.model_loader import ModelLoader
from src.inference.feature_serving import FeatureServer
from config.schemas import PlotInput

# Initialize FastAPI app
app = FastAPI(
    title="Nuru Yield Prediction API",
    description="Production API for agricultural yield prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loader and feature server
model_loader: Optional[ModelLoader] = None
feature_server: Optional[FeatureServer] = None

# Prometheus metrics
REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total number of inference requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "inference_request_latency_seconds",
    "Latency of inference requests in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    import time
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNT.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
    return response


# Request/Response Schemas
class PredictionRequest(BaseModel):
    """Request schema for yield prediction."""
    plot_id: int = Field(..., gt=0, description="Plot identifier")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    planting_date: str = Field(..., description="Planting date (YYYY-MM-DD)")
    season: Optional[str] = Field(None, description="Season: Short Rains or Long Rains")
    altitude: Optional[float] = Field(None, ge=0, le=10000, description="Altitude in meters")
    model_name: Optional[str] = Field("ensemble", description="Model to use: ensemble, xgboost, lightgbm, catboost, randomforest")

    class Config:
        json_schema_extra = {
            "example": {
                "plot_id": 1,
                "latitude": -0.499127,
                "longitude": 37.612253,
                "planting_date": "2021-10-07",
                "season": "Short Rains",
                "altitude": 1252.08,
                "model_name": "ensemble"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""
    plots: List[PredictionRequest] = Field(..., min_items=1, max_items=1000)
    model_name: Optional[str] = Field("ensemble", description="Model to use")


class PredictionResponse(BaseModel):
    """Response schema for yield prediction."""
    plot_id: int
    predicted_yield_kg_per_ha: float = Field(..., description="Predicted dry yield in kg/ha")
    model_name: str
    model_version: Optional[str] = None
    confidence_score: Optional[float] = Field(None, ge=0, le=1)
    features_used: Optional[Dict[str, Any]] = None
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""
    predictions: List[PredictionResponse]
    total_plots: int
    successful: int
    failed: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]
    mlflow_connected: bool
    feature_store_available: bool
    timestamp: str


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and feature server on startup."""
    global model_loader, feature_server
    
    logger.info("Starting inference service...")
    
    try:
        model_loader = ModelLoader()
        model_loader.load_all_models()
        logger.success(f"Loaded models: {list(model_loader.models.keys())}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        # Service can still start but predictions will fail
        model_loader = None
    
    try:
        feature_server = FeatureServer()
        logger.success("Feature server initialized")
    except Exception as e:
        logger.error(f"Failed to initialize feature server: {e}")
        feature_server = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down inference service...")

@app.get("/metrics")
async def metrics():
    return JSONResponse(content=generate_latest().decode("utf-8"), media_type=CONTENT_TYPE_LATEST)


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "Nuru Yield Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_loaded = list(model_loader.models.keys()) if model_loader else []
    
    mlflow_connected = False
    if model_loader:
        try:
            import mlflow
            mlflow.set_tracking_uri(model_loader.tracking_uri)
            mlflow_connected = True
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if model_loader and models_loaded else "degraded",
        models_loaded=models_loaded,
        mlflow_connected=mlflow_connected,
        feature_store_available=feature_server is not None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_yield(request: PredictionRequest):
    """
    Predict yield for a single plot.
    
    Args:
        request: Prediction request with plot details
        
    Returns:
        Prediction response with yield estimate
    """
    if not model_loader:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not feature_server:
        raise HTTPException(status_code=503, detail="Feature server not available")
    
    try:
        # Get features for the plot
        features_df = feature_server.get_features_for_inference(
            plot_id=request.plot_id,
            latitude=request.latitude,
            longitude=request.longitude,
            planting_date=request.planting_date,
            season=request.season,
            altitude=request.altitude
        )
        
        if features_df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No features found for plot {request.plot_id}"
            )
        
        # Prepare features for model (drop non-feature columns)
        feature_cols = [
            c for c in features_df.columns 
            if c not in ["plot_id", "planting_date", "latitude", "longitude", "season"]
        ]
        X = features_df[feature_cols].copy()
        
        # Make prediction
        predictions = model_loader.predict(X, model_name=request.model_name)
        predicted_yield = float(predictions[0])
        
        # Get model info
        model_info = model_loader.get_model_info(request.model_name)
        
        return PredictionResponse(
            plot_id=request.plot_id,
            predicted_yield_kg_per_ha=round(predicted_yield, 2),
            model_name=request.model_name,
            model_version=model_info.get("run_id"),
            confidence_score=None,  # Could add confidence intervals
            features_used={col: float(X[col].iloc[0]) for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])},
            timestamp=datetime.utcnow().isoformat()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Prediction error for plot {request.plot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_yield_batch(request: BatchPredictionRequest):
    """
    Predict yield for multiple plots in batch.
    
    Args:
        request: Batch prediction request
        
    Returns:
        Batch prediction response
    """
    if not model_loader:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not feature_server:
        raise HTTPException(status_code=503, detail="Feature server not available")
    
    predictions = []
    successful = 0
    failed = 0
    
    for plot_request in request.plots:
        try:
            # Get features
            features_df = feature_server.get_features_for_inference(
                plot_id=plot_request.plot_id,
                latitude=plot_request.latitude,
                longitude=plot_request.longitude,
                planting_date=plot_request.planting_date,
                season=plot_request.season,
                altitude=plot_request.altitude
            )
            
            if features_df.empty:
                failed += 1
                continue
            
            # Prepare features
            feature_cols = [
                c for c in features_df.columns 
                if c not in ["plot_id", "planting_date", "latitude", "longitude", "season"]
            ]
            X = features_df[feature_cols].copy()
            
            # Predict
            preds = model_loader.predict(X, model_name=request.model_name)
            predicted_yield = float(preds[0])
            
            model_info = model_loader.get_model_info(request.model_name)
            
            predictions.append(PredictionResponse(
                plot_id=plot_request.plot_id,
                predicted_yield_kg_per_ha=round(predicted_yield, 2),
                model_name=request.model_name,
                model_version=model_info.get("run_id"),
                timestamp=datetime.utcnow().isoformat()
            ))
            successful += 1
            
        except Exception as e:
            logger.warning(f"Failed prediction for plot {plot_request.plot_id}: {e}")
            failed += 1
    
    return BatchPredictionResponse(
        predictions=predictions,
        total_plots=len(request.plots),
        successful=successful,
        failed=failed
    )


@app.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List available models and their metadata."""
    if not model_loader:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    models_info = {}
    for model_name in model_loader.models.keys():
        models_info[model_name] = model_loader.get_model_info(model_name)
    
    return {
        "available_models": list(model_loader.models.keys()),
        "model_details": models_info
    }


@app.post("/models/reload")
async def reload_models():
    """Reload models from MLflow (useful after model updates)."""
    global model_loader
    
    try:
        model_loader = ModelLoader()
        model_loader.load_all_models()
        return {
            "status": "success",
            "models_loaded": list(model_loader.models.keys()),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to reload models: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload models: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)






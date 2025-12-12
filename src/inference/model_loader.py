# src/inference/model_loader.py
"""
Model loader for production inference.
Loads models from MLflow registry and manages model versions.
"""
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from typing import Optional, Dict, Any
from loguru import logger
import pandas as pd
import numpy as np

from config.settings import get_settings

settings = get_settings()


class ModelLoader:
    """Load and manage production models from MLflow registry."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """
        Initialize model loader with MLflow tracking URI.
        
        Args:
            tracking_uri: MLflow tracking URI (defaults to env var or local)
        """
        self.tracking_uri = tracking_uri or os.environ.get(
            "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
        )
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Set S3/MinIO credentials for artifact access
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get(
            "MLFLOW_S3_ENDPOINT_URL", 
            os.environ.get("S3_ENDPOINT_URL", "http://minio:9000")
        )
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
            "S3_ACCESS_KEY", 
            os.environ.get("MINIO_ROOT_USER", "minioadmin") or "minioadmin"
        )
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
            "S3_SECRET_KEY", 
            os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin") or "minioadmin"
        )
        
        self.models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        self.experiment_name = "YieldPrediction_MultiModel"
        
    def load_production_model(
        self, 
        model_name: str = "Ensemble_VotingRegressor",
        stage: str = "Production"
    ) -> Any:
        """
        Load model from MLflow model registry by name and stage.
        
        Args:
            model_name: Name of the model in registry (default: Ensemble)
            stage: Model stage (Production, Staging, None)
            
        Returns:
            Loaded model object
        """
        try:
            # Try to load from model registry
            model_uri = f"models:/{model_name}/{stage}"
            logger.info(f"Loading model from registry: {model_uri}")
            model = mlflow.pyfunc.load_model(model_uri)
            self.models[model_name] = model
            logger.success(f"Loaded {model_name} from stage {stage}")
            return model
        except Exception as e:
            logger.warning(f"Failed to load from registry ({model_uri}): {e}")
            # Fallback: load latest run from experiment
            return self._load_latest_from_experiment(model_name)
    
    def _load_latest_from_experiment(self, run_name_prefix: str) -> Any:
        """Load latest model from experiment by run name prefix."""
        try:
            mlflow.set_experiment(self.experiment_name)
            client = mlflow.tracking.MlflowClient()
            
            # Get latest run with matching name
            runs = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(self.experiment_name).experiment_id],
                filter_string=f"tags.mlflow.runName LIKE '{run_name_prefix}%'",
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                raise ValueError(f"No runs found for {run_name_prefix}")
            
            run = runs[0]
            run_id = run.info.run_id
            
            # Load model artifacts
            if "Ensemble" in run_name_prefix:
                model_uri = f"runs:/{run_id}/ensemble_model"
            elif "XGBoost" in run_name_prefix:
                model_uri = f"runs:/{run_id}/xgb_model"
            elif "LightGBM" in run_name_prefix:
                model_uri = f"runs:/{run_id}/lgb_model"
            elif "CatBoost" in run_name_prefix:
                model_uri = f"runs:/{run_id}/cat_model"
            elif "RandomForest" in run_name_prefix:
                model_uri = f"runs:/{run_id}/rf_model"
            else:
                model_uri = f"runs:/{run_id}/model"
            
            model = mlflow.pyfunc.load_model(model_uri)
            self.models[run_name_prefix] = model
            
            # Store metadata
            self.model_metadata[run_name_prefix] = {
                "run_id": run_id,
                "run_name": run.info.run_name,
                "start_time": run.info.start_time,
                "metrics": run.data.metrics,
                "params": run.data.params
            }
            
            logger.success(f"Loaded {run_name_prefix} from experiment (run_id: {run_id})")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load from experiment: {e}")
            raise
    
    def load_all_models(self) -> Dict[str, Any]:
        """Load all available models (ensemble + individual)."""
        models = {}
        
        # Try to load ensemble first (preferred)
        try:
            models["ensemble"] = self.load_production_model("Ensemble_VotingRegressor")
        except Exception as e:
            logger.warning(f"Could not load ensemble: {e}")
        
        # Load individual models as fallback
        for model_name in ["XGBoost_GridSearch", "LightGBM_GridSearch", "CatBoost_GridSearch", "RandomForest_GridSearch"]:
            try:
                models[model_name.lower().replace("_gridsearch", "")] = self._load_latest_from_experiment(model_name)
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
        
        if not models:
            raise RuntimeError("No models could be loaded from MLflow")
        
        self.models = models
        return models
    
    def get_model(self, model_name: str = "ensemble") -> Any:
        """Get loaded model by name."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not loaded. Available: {list(self.models.keys())}")
        return self.models[model_name]
    
    def predict(self, features: pd.DataFrame, model_name: str = "ensemble") -> np.ndarray:
        """
        Make predictions using loaded model.
        
        Args:
            features: DataFrame with feature columns matching training data
            model_name: Name of model to use (default: ensemble)
            
        Returns:
            Predictions array
        """
        model = self.get_model(model_name)
        
        # Ensure features match expected format
        # MLflow pyfunc models handle preprocessing, but we ensure correct types
        features = features.copy()
        
        # Convert to numeric where needed
        for col in features.columns:
            if features[col].dtype == 'object':
                try:
                    features[col] = pd.to_numeric(features[col], errors='coerce')
                except Exception:
                    pass
        
        predictions = model.predict(features)
        return predictions
    
    def get_model_info(self, model_name: str = "ensemble") -> Dict:
        """Get metadata for a loaded model."""
        if model_name not in self.model_metadata:
            return {"status": "loaded", "model_name": model_name}
        return self.model_metadata[model_name]


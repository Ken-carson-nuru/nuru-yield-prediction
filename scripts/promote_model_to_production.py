#!/usr/bin/env python3
"""
Script to promote a model from MLflow experiment to Production stage.
Usage: python scripts/promote_model_to_production.py --run-id <run_id> --model-name <name>
"""
import argparse
import os
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

def promote_model_to_production(
    run_id: str,
    model_name: str = "Ensemble_VotingRegressor",
    stage: str = "Production"
):
    """
    Promote a model to Production stage in MLflow model registry.
    
    Args:
        run_id: MLflow run ID
        model_name: Name for the model in registry
        stage: Target stage (Production, Staging, Archived)
    """
    # Set tracking URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    # Set S3/MinIO credentials
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get(
        "MLFLOW_S3_ENDPOINT_URL", 
        os.environ.get("S3_ENDPOINT_URL", "http://localhost:9000")
    )
    os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get(
        "S3_ACCESS_KEY", 
        os.environ.get("MINIO_ROOT_USER", "minioadmin") or "minioadmin"
    )
    os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get(
        "S3_SECRET_KEY", 
        os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin") or "minioadmin"
    )
    
    client = MlflowClient()
    
    # Determine model artifact path based on model name
    if "Ensemble" in model_name:
        artifact_path = "ensemble_model"
    elif "XGBoost" in model_name:
        artifact_path = "xgb_model"
    elif "LightGBM" in model_name:
        artifact_path = "lgb_model"
    elif "CatBoost" in model_name:
        artifact_path = "cat_model"
    elif "RandomForest" in model_name:
        artifact_path = "rf_model"
    else:
        artifact_path = "model"
    
    model_uri = f"runs:/{run_id}/{artifact_path}"
    
    try:
        # Register the model
        logger.info(f"Registering model {model_name} from run {run_id}...")
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        
        logger.info(f"Model version {mv.version} created for {model_name}")
        
        # Transition to Production
        logger.info(f"Transitioning model to {stage} stage...")
        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage=stage
        )
        
        # Archive any previous Production versions
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            for pv in prod_versions:
                if pv.version != str(mv.version):
                    logger.info(f"Archiving previous production version {pv.version}")
                    client.transition_model_version_stage(
                        name=model_name,
                        version=int(pv.version),
                        stage="Archived"
                    )
        except Exception as e:
            logger.warning(f"Could not archive previous versions: {e}")
        
        logger.success(f"Model {model_name} version {mv.version} promoted to {stage}")
        return mv
        
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Promote MLflow model to Production")
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--model-name", default="Ensemble_VotingRegressor", help="Model name in registry")
    parser.add_argument("--stage", default="Production", choices=["Production", "Staging", "Archived"], help="Target stage")
    
    args = parser.parse_args()
    
    promote_model_to_production(
        run_id=args.run_id,
        model_name=args.model_name,
        stage=args.stage
    )


#!/usr/bin/env python3
"""
Script to properly register a model in MLflow with correct S3 configuration.
This fixes permission issues by ensuring MLflow uses the remote server and S3.
"""
import argparse
import os
import mlflow
from mlflow.tracking import MlflowClient
from loguru import logger

def register_model_properly(
    run_id: str,
    model_name: str = "Ensemble_VotingRegressor",
    artifact_path: str = "ensemble_model",
    stage: str = "Production"
):
    """
    Register model with proper S3/MinIO configuration.
    
    Args:
        run_id: MLflow run ID
        model_name: Name for the model in registry
        artifact_path: Path to model artifact in run (e.g., "ensemble_model", "xgb_model")
        stage: Target stage (Production, Staging, Archived)
    """
    # Set tracking URI to remote server
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"Using MLflow tracking URI: {tracking_uri}")
    
    # CRITICAL: Set S3/MinIO credentials BEFORE any MLflow operations
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
    
    # Ensure we're not using local file system
    if os.path.exists("./mlruns"):
        logger.warning("Local ./mlruns directory exists. This may cause conflicts.")
        logger.warning("Ensure MLFLOW_TRACKING_URI is set to remote server.")
    
    client = MlflowClient()
    
    # Verify run exists and get run info
    try:
        run = client.get_run(run_id)
        logger.info(f"Found run: {run.info.run_name} (ID: {run_id})")
        logger.info(f"Experiment ID: {run.info.experiment_id}")
    except Exception as e:
        logger.error(f"Run {run_id} not found: {e}")
        raise
    
    # Construct model URI - use runs:/ format
    model_uri = f"runs:/{run_id}/{artifact_path}"
    logger.info(f"Model URI: {model_uri}")
    
    # Verify artifact exists
    try:
        # Try to load the model to verify it exists
        logger.info("Verifying model artifact exists...")
        model = mlflow.pyfunc.load_model(model_uri)
        logger.success(f"Model artifact verified at {model_uri}")
    except Exception as e:
        logger.error(f"Model artifact not found at {model_uri}: {e}")
        logger.info("Available artifacts in run:")
        try:
            artifacts = client.list_artifacts(run_id)
            for artifact in artifacts:
                logger.info(f"  - {artifact.path}")
        except Exception:
            pass
        raise
    
    # Register the model
    try:
        logger.info(f"Registering model '{model_name}' from run {run_id}...")
        mv = client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=run_id
        )
        
        logger.success(f"Model version {mv.version} created for {model_name}")
        logger.info(f"Model version details:")
        logger.info(f"  - Version: {mv.version}")
        logger.info(f"  - Stage: {mv.current_stage}")
        logger.info(f"  - Source: {mv.source}")
        
        # Transition to target stage
        if stage != mv.current_stage:
            logger.info(f"Transitioning model to '{stage}' stage...")
            client.transition_model_version_stage(
                name=model_name,
                version=mv.version,
                stage=stage
            )
            logger.success(f"Model transitioned to '{stage}' stage")
        
        # Archive any previous Production versions (if promoting to Production)
        if stage == "Production":
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
        
        logger.success(f"✅ Model '{model_name}' version {mv.version} successfully registered and promoted to '{stage}'")
        return mv
        
    except Exception as e:
        logger.error(f"Failed to register model: {e}")
        logger.error("Common issues:")
        logger.error("  1. MLFLOW_TRACKING_URI not set to remote server")
        logger.error("  2. S3/MinIO credentials not configured")
        logger.error("  3. Model artifact path incorrect")
        logger.error("  4. Permission issues with local mlruns directory")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Register MLflow model with proper S3 configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register ensemble model
  python scripts/register_model_fix.py --run-id abc123 --model-name Ensemble_VotingRegressor
  
  # Register XGBoost model
  python scripts/register_model_fix.py --run-id abc123 --model-name XGBoost_Model --artifact-path xgb_model
  
  # Register to Staging first
  python scripts/register_model_fix.py --run-id abc123 --stage Staging
        """
    )
    parser.add_argument("--run-id", required=True, help="MLflow run ID")
    parser.add_argument("--model-name", default="Ensemble_VotingRegressor", help="Model name in registry")
    parser.add_argument("--artifact-path", default="ensemble_model", 
                       help="Model artifact path (ensemble_model, xgb_model, lgb_model, cat_model, rf_model)")
    parser.add_argument("--stage", default="Production", 
                       choices=["Production", "Staging", "Archived"], 
                       help="Target stage")
    
    args = parser.parse_args()
    
    # Set environment variables if not set (for local execution)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
        logger.info("MLFLOW_TRACKING_URI not set, using http://localhost:5000")
    
    register_model_properly(
        run_id=args.run_id,
        model_name=args.model_name,
        artifact_path=args.artifact_path,
        stage=args.stage
    )


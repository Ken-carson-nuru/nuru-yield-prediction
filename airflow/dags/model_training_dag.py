from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from datetime import datetime
import os
import sys
import pandas as pd
import numpy as np
from loguru import logger

# Ensure project packages (src/, config/) are importable in Airflow
sys.path.insert(0, '/opt/airflow/dags/repo')

# ML and logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

from catboost import CatBoostRegressor
import xgboost as xgb
import lightgbm as lgb

from src.storage import DataStorage
from config.settings import get_settings


default_args = {
    "owner": "nuru",
    "start_date": days_ago(1),
    "retries": 2,
}

settings = get_settings()
BUCKET = settings.S3_BUCKET_NAME


def _get_tracking_uri() -> str:
    """Resolve MLflow tracking URI for Airflow runtime."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    # Default to a local path inside the repo (persisted by the Airflow container)
    return "/opt/airflow/dags/repo/mlruns"


def _get_ds_from_airflow_env() -> str:
    """Get YYYY-MM-DD date string from Airflow context env reliably."""
    exec_dt_env = os.environ.get("AIRFLOW_CTX_EXECUTION_DATE")
    if exec_dt_env:
        try:
            return pd.to_datetime(exec_dt_env).strftime("%Y-%m-%d")
        except Exception:
            return exec_dt_env.split("T")[0]
    return datetime.utcnow().strftime("%Y-%m-%d")


def _read_parquet_from_s3(bucket: str, key: str, storage: DataStorage) -> pd.DataFrame:
    """Utility to read parquet from S3/MinIO using DataStorage's s3 client."""
    resp = storage.s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(pd.io.common.BytesIO(resp["Body"].read()))


def _resolve_dataset_path() -> str:
    """Find final_dataset.csv guided by yield_model.md usage."""
    candidates = [
        "/opt/airflow/dags/repo/final_dataset.csv",
        "/opt/airflow/dags/repo/notebooks/final_dataset.csv",
        "/opt/airflow/dags/repo/data/final_dataset.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "final_dataset.csv not found. Place it at repo root or notebooks/ or data/."
    )


def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Mirror preprocessing in yield_model.md (types, encodings, date features)."""
    # Convert Planting_Date to datetime and extract date features
    if "Planting_Date" in df.columns:
        df["Planting_Date"] = pd.to_datetime(df["Planting_Date"], errors="coerce")
        df["plant_month"] = df["Planting_Date"].dt.month
        df["plant_doy"] = df["Planting_Date"].dt.dayofyear
        # Drop original date column (same as notebook)
        df = df.drop(columns=["Planting_Date"]) 

    # Label encode crop_type to a numeric feature
    if "crop_type" in df.columns:
        le = LabelEncoder()
        df["crop_type_enc"] = le.fit_transform(df["crop_type"].astype(str))

    # Convert all to numeric (coerce errors)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop unnamed index-like column if present
    drop_cols = [c for c in ["Unnamed: 0"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def _split_features_labels(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=42)


def _eval_and_log(run_name: str, model, X_test, y_test, params: dict):
    with mlflow.start_run(run_name=run_name):
        # Log params and compute metrics
        if params:
            mlflow.log_params(params)
        preds = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        mlflow.log_metrics({"RMSE": rmse, "MAE": mae, "R2": r2})
        return {"RMSE": rmse, "MAE": mae, "R2": r2}


with DAG(
    dag_id="model_training_dag_weekly",
    default_args=default_args,
    schedule_interval="0 3 * * 1",  # Every Monday 03:00
    catchup=False,
    tags=["model", "training", "gridsearch"],
):

    @task(multiple_outputs=False)
    def load_dataset_path() -> str:
        """Locate the dataset file used in yield_model.md."""
        path = _resolve_dataset_path()
        logger.info(f"Using dataset at: {path}")
        return path

    @task(multiple_outputs=False)
    def load_plot_features_path() -> str:
        """Return S3 path for plot-level features (weather + indices combined)."""
        ds = _get_ds_from_airflow_env()
        key = f"{settings.S3_BASE_PREFIX}/features/{ds}/plot_features.parquet"
        return f"s3://{BUCKET}/{key}"

    @task(multiple_outputs=True)
    def combine_features_and_labels(labels_csv_path: str, plot_features_path: str) -> dict:
        """Read labels CSV, read plot_features from S3, and merge to form final dataset.
        Ensures weather + indices (from plot_features) are combined with yield labels.
        """
        # Load labels
        labels_df = pd.read_csv(labels_csv_path)
        # Normalize label keys
        if "plot_no" in labels_df.columns:
            labels_df = labels_df.rename(columns={"plot_no": "plot_id"})
        if "Planting_Date" in labels_df.columns:
            labels_df["Planting_Date"] = pd.to_datetime(labels_df["Planting_Date"], errors="coerce")

        # Load plot features from S3
        storage = DataStorage()
        key = plot_features_path.replace(f"s3://{BUCKET}/", "")
        features_df = _read_parquet_from_s3(BUCKET, key, storage)
        # Normalize features timestamp
        if "planting_date" in features_df.columns:
            features_df["planting_date"] = pd.to_datetime(features_df["planting_date"], errors="coerce")

        # Merge on plot_id and planting_date (fallback to plot_id only if needed)
        merged = labels_df.merge(
            features_df,
            left_on=[c for c in ["plot_id", "Planting_Date"] if c in labels_df.columns],
            right_on=[c for c in ["plot_id", "planting_date"] if c in features_df.columns],
            how="left",
        )
        if merged.isna().all(axis=1).sum() > 0:
            # Fallback: join by plot_id only
            merged = labels_df.merge(features_df, on="plot_id", how="left")

        # Return as JSON-safe records for XCom
        merged_records = merged.to_dict(orient="records")
        logger.info(f"Combined dataset rows: {len(merged_records)}")
        return merged_records

    @task(multiple_outputs=True)
    def train_models(dataset_records: list) -> dict:
        """Preprocess, split, run GridSearchCV for XGB, LGBM, CatBoost; log to MLflow.
        Accepts combined dataset records to ensure features include weather + indices.
        """
        # Set MLflow tracking
        tracking_uri = _get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("YieldPrediction_MultiModel")

        # Load and preprocess
        df = pd.DataFrame(dataset_records)
        df = _preprocess(df)
        target = "dry_harvest_kg/ha"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        X_train, X_test, y_train, y_test = _split_features_labels(df, target)

        results = {}

        # =====================
        # XGBoost GridSearchCV
        # =====================
        xgb_base = xgb.XGBRegressor(
            random_state=42,
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        xgb_grid = {
            "n_estimators": [600, 1200],
            "learning_rate": [0.03, 0.05],
            "max_depth": [6, 8, 10],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        xgb_gs = GridSearchCV(xgb_base, xgb_grid, cv=3, n_jobs=-1, scoring="neg_root_mean_squared_error")
        xgb_gs.fit(X_train, y_train)
        xgb_best = xgb_gs.best_estimator_
        xgb_metrics = _eval_and_log("XGBoost_GridSearch", xgb_best, X_test, y_test, xgb_gs.best_params_)
        mlflow.xgboost.log_model(xgb_best, name="xgb_model")
        results["XGBoost"] = {**xgb_metrics, "best_params": xgb_gs.best_params_}

        # =====================
        # LightGBM GridSearchCV
        # =====================
        lgb_base = lgb.LGBMRegressor(
            random_state=42,
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        lgb_grid = {
            "n_estimators": [600, 1200],
            "learning_rate": [0.03, 0.05],
            "num_leaves": [31, 63, 127],
            "max_depth": [-1, 8, 12],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
        }
        lgb_gs = GridSearchCV(lgb_base, lgb_grid, cv=3, n_jobs=-1, scoring="neg_root_mean_squared_error")
        lgb_gs.fit(X_train, y_train)
        lgb_best = lgb_gs.best_estimator_
        lgb_metrics = _eval_and_log("LightGBM_GridSearch", lgb_best, X_test, y_test, lgb_gs.best_params_)
        mlflow.lightgbm.log_model(lgb_best, name="lgb_model")
        results["LightGBM"] = {**lgb_metrics, "best_params": lgb_gs.best_params_}

        # =====================
        # CatBoost GridSearchCV
        # =====================
        # Use scikit wrapper and pass categorical feature indices via fit_params
        cat_base = CatBoostRegressor(
            random_seed=42,
            verbose=False,
            loss_function="RMSE",
            eval_metric="RMSE",
        )
        cat_grid = {
            "depth": [6, 8, 10],
            "learning_rate": [0.03, 0.05],
            "iterations": [1000, 2000],
        }
        cat_gs = GridSearchCV(cat_base, cat_grid, cv=3, n_jobs=-1, scoring="neg_root_mean_squared_error")

        # Determine categorical indices (prefer encoded column if present)
        cat_cols = [c for c in ["crop_type_enc"] if c in X_train.columns]
        cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
        # Fit with categorical feature indices
        cat_gs.fit(X_train, y_train, **({"cat_features": cat_idx} if cat_idx else {}))
        cat_best = cat_gs.best_estimator_
        cat_metrics = _eval_and_log("CatBoost_GridSearch", cat_best, X_test, y_test, cat_gs.best_params_)
        mlflow.catboost.log_model(cat_best, name="cat_model")
        results["CatBoost"] = {**cat_metrics, "best_params": cat_gs.best_params_}

        logger.success("Model training with GridSearchCV completed for XGB, LGBM, CatBoost")
        return results

    @task(multiple_outputs=False)
    def log_completion(metrics: dict) -> bool:
        logger.info(f"Training metrics summary: {metrics}")
        return True

    # Task graph
    dataset_path = load_dataset_path()
    plot_features_path = load_plot_features_path()
    combined_dataset = combine_features_and_labels(dataset_path, plot_features_path)
    metrics = train_models(combined_dataset)
    done = log_completion(metrics)

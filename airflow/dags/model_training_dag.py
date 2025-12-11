from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from datetime import datetime
import os
import sys
import pandas as pd
import numpy as np
from io import BytesIO
import socket
from urllib.parse import urlparse
from loguru import logger

# Ensure project packages (src/, config/) are importable in Airflow
sys.path.insert(0, '/opt/airflow/dags/repo')

# ML and logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

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
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        try:
            p = urlparse(uri)
            host = p.hostname
            if p.scheme in ("http", "https") and host:
                socket.gethostbyname(host)
                return uri
        except Exception:
            logger.warning(f"Invalid or unreachable MLFLOW_TRACKING_URI '{uri}', falling back to local path")
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
    return pd.read_parquet(BytesIO(resp["Body"].read()))


def _resolve_dataset_path() -> str:
    """Deprecated: CSV-based labels path (kept for rare fallback)."""
    candidates = [
        "/opt/airflow/dags/repo/final_dataset.csv",
        "/opt/airflow/dags/repo/notebooks/final_dataset.csv",
        "/opt/airflow/dags/repo/data/final_dataset.csv",
        "/opt/airflow/dags/repo/final_for_yield.csv",
        "/opt/airflow/dags/repo/notebooks/final_for_yield.csv",
        "/opt/airflow/dags/repo/data/processed/final_for_yield.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _find_latest_s3_parquet(bucket: str, base_prefix: str, subdir: str, filename: str, storage: DataStorage) -> str:
    """Find the latest available parquet in S3 under base_prefix/subdir/YYYY-MM-DD/filename.
    Returns full s3:// path or None.
    """
    prefix = f"{base_prefix}/{subdir}/"
    try:
        resp = storage.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        contents = resp.get("Contents", [])
        keys = [obj["Key"] for obj in contents if obj["Key"].endswith(f"/{filename}")]
        if not keys:
            return None
        dated = []
        for k in keys:
            parts = k.split("/")
            # Expect .../<subdir>/<YYYY-MM-DD>/<filename>
            ds = parts[-2] if len(parts) >= 2 else None
            try:
                dt = pd.to_datetime(ds)
                dated.append((k, dt))
            except Exception:
                continue
        if not dated:
            return None
        dated.sort(key=lambda x: x[1], reverse=True)
        latest_key = dated[0][0]
        return f"s3://{bucket}/{latest_key}"
    except Exception as e:
        logger.warning(f"Failed to list S3 objects for {prefix}: {e}")
        return None


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


def _assign_plot_id_from_coordinates(labels_df: pd.DataFrame, features_df: pd.DataFrame, tolerance_deg: float = 0.01) -> pd.DataFrame:
    """Assign plot_id in labels_df using nearest latitude/longitude from features_df when missing.

    tolerance_deg is the maximum allowed Euclidean distance in degrees to accept a match.
    Roughly, 0.01° ~ 1.1 km at the equator; adjust as needed.
    """
    df = labels_df.copy()

    # Normalize coordinate column names in labels
    if "Latitude" in df.columns and "latitude" not in df.columns:
        df = df.rename(columns={"Latitude": "latitude"})
    if "Longitude" in df.columns and "longitude" not in df.columns:
        df = df.rename(columns={"Longitude": "longitude"})

    # If plot_id already present and non-null for all rows, nothing to do
    if "plot_id" in df.columns and df["plot_id"].notna().all():
        return df

    # Require coordinates in both frames
    if not {"latitude", "longitude"}.issubset(df.columns):
        return df
    if not {"latitude", "longitude", "plot_id"}.issubset(features_df.columns):
        return df

    # Prepare coordinate arrays (dropna and ensure numeric)
    try:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    except Exception:
        return df

    features_coords = features_df[["plot_id", "latitude", "longitude"]].dropna().drop_duplicates(subset=["plot_id"]).copy()
    if features_coords.empty:
        return df

    labels_missing_mask = ~(df.get("plot_id").notna() if "plot_id" in df.columns else pd.Series([False]*len(df), index=df.index))
    labels_missing = df.loc[labels_missing_mask & df[["latitude", "longitude"]].notna().all(axis=1)]
    if labels_missing.empty:
        return df

    # Fit nearest neighbors on features coordinates
    nbrs = NearestNeighbors(n_neighbors=1, algorithm="auto")
    nbrs.fit(features_coords[["latitude", "longitude"]])
    distances, indices = nbrs.kneighbors(labels_missing[["latitude", "longitude"]])

    # Assign matches within tolerance
    matched_plot_ids = []
    for i, (dist, idx) in enumerate(zip(distances[:, 0], indices[:, 0])):
        if np.isfinite(dist) and dist <= tolerance_deg:
            pid = features_coords.iloc[idx]["plot_id"]
            matched_plot_ids.append(pid)
        else:
            matched_plot_ids.append(np.nan)

    df.loc[labels_missing.index, "plot_id"] = matched_plot_ids
    return df


with DAG(
    dag_id="model_training_dag_weekly",
    default_args=default_args,
    schedule_interval="0 3 * * 1",  # Every Monday 03:00
    catchup=False,
    tags=["model", "training", "gridsearch"],
):

    @task(multiple_outputs=False)
    def load_labels_path() -> str:
        """Return S3 path for labels parquet if available; will be used to merge with features."""
        ds = _get_ds_from_airflow_env()
        # Convention: labels stored under base prefix
        key = f"{settings.S3_BASE_PREFIX}/labels/{ds}/yield.parquet"
        return f"s3://{BUCKET}/{key}"

    @task(multiple_outputs=False)
    def load_plot_features_path() -> str:
        """Return S3 path for plot-level features (weather + indices combined)."""
        ds = _get_ds_from_airflow_env()
        key = f"{settings.S3_BASE_PREFIX}/features/{ds}/plot_features.parquet"
        return f"s3://{BUCKET}/{key}"

    @task(multiple_outputs=False)
    def combine_features_and_labels(labels_path: str, plot_features_path: str) -> dict:
        """Read labels (Parquet preferred), read plot_features from S3, and merge to form final dataset.
        Ensures weather + indices (from plot_features) are combined with yield labels.
        """
        storage = DataStorage()
        labels_df = None
        # Try S3 parquet first (provided path)
        try:
            if labels_path.startswith("s3://"):
                labels_key = labels_path.replace(f"s3://{BUCKET}/", "")
                labels_df = _read_parquet_from_s3(BUCKET, labels_key, storage)
                logger.info(f"Loaded labels from S3: {labels_path}")
        except Exception as e:
            logger.warning(f"Failed to read labels from provided S3 path {labels_path}: {e}")
            labels_df = None

        # Fallback: find latest labels parquet in S3
        if labels_df is None:
            latest_labels_s3 = _find_latest_s3_parquet(BUCKET, settings.S3_BASE_PREFIX, "labels", "yield.parquet", storage)
            if latest_labels_s3:
                try:
                    labels_key = latest_labels_s3.replace(f"s3://{BUCKET}/", "")
                    labels_df = _read_parquet_from_s3(BUCKET, labels_key, storage)
                    logger.info(f"Loaded latest available labels from S3: {latest_labels_s3}")
                except Exception as e:
                    logger.warning(f"Failed to read latest labels from S3 {latest_labels_s3}: {e}")

        # Fallback to local parquet (support older notebook outputs)
        if labels_df is None:
            parq_candidates = [
                "/opt/airflow/dags/repo/final_dataset.parquet",
                "/opt/airflow/dags/repo/notebooks/final_dataset.parquet",
                "/opt/airflow/dags/repo/data/final_dataset.parquet",
                "/opt/airflow/dags/repo/final_for_yield.parquet",
                "/opt/airflow/dags/repo/notebooks/final_for_yield.parquet",
                "/opt/airflow/dags/repo/data/processed/final_for_yield.parquet",
            ]
            for p in parq_candidates:
                if os.path.exists(p):
                    labels_df = pd.read_parquet(p)
                    logger.info(f"Loaded labels from local parquet: {p}")
                    break

        # Last resort: CSV fallback if present (not preferred)
        if labels_df is None:
            csv_path = _resolve_dataset_path()
            if csv_path:
                labels_df = pd.read_csv(csv_path)
                logger.warning(f"Falling back to CSV labels at: {csv_path}")
            else:
                # Gracefully skip if labels truly missing
                from airflow.exceptions import AirflowSkipException
                logger.warning("No labels parquet/CSV found. Skipping combine and training for this run.")
                raise AirflowSkipException("Labels not found for execution date")

        # Normalize label keys
        if "plot_no" in labels_df.columns:
            labels_df = labels_df.rename(columns={"plot_no": "plot_id"})
        if "Planting_Date" in labels_df.columns:
            # Parse as UTC and strip timezone to avoid tz-aware vs naive mismatches
            labels_df["Planting_Date"] = pd.to_datetime(labels_df["Planting_Date"], errors="coerce", utc=True).dt.tz_convert(None)

        # Load plot features from S3
        # Load plot features from S3 (provided path first)
        features_df = None
        try:
            key = plot_features_path.replace(f"s3://{BUCKET}/", "")
            features_df = _read_parquet_from_s3(BUCKET, key, storage)
            logger.info(f"Loaded plot features from S3: {plot_features_path}")
        except Exception as e:
            logger.warning(f"Failed to read plot features from provided S3 path {plot_features_path}: {e}")
            # Try latest available features parquet
            latest_features_s3 = _find_latest_s3_parquet(BUCKET, settings.S3_BASE_PREFIX, "features", "plot_features.parquet", storage)
            if latest_features_s3:
                try:
                    key = latest_features_s3.replace(f"s3://{BUCKET}/", "")
                    features_df = _read_parquet_from_s3(BUCKET, key, storage)
                    logger.info(f"Loaded latest available plot features from S3: {latest_features_s3}")
                except Exception as e2:
                    logger.warning(f"Failed to read latest plot features from S3 {latest_features_s3}: {e2}")
        if features_df is None:
            raise FileNotFoundError("Plot features parquet not found in S3. Ensure feature materialization ran.")
        # Normalize features timestamp
        if "planting_date" in features_df.columns:
            # Parse as UTC and strip timezone to avoid tz-aware vs naive mismatches
            features_df["planting_date"] = pd.to_datetime(features_df["planting_date"], errors="coerce", utc=True).dt.tz_convert(None)

        # Merge on plot_id and planting_date (fallbacks: plot_id only, then lat/long)
        # Normalize keys for robust join
        # Map lower-case columns if present
        if "plot_no" in labels_df.columns and "plot_id" not in labels_df.columns:
            labels_df = labels_df.rename(columns={"plot_no": "plot_id"})
        if "planting_date" in labels_df.columns and "Planting_Date" not in labels_df.columns:
            # unify to Title case used in training preprocessing
            labels_df = labels_df.rename(columns={"planting_date": "Planting_Date"})

        # Try to infer plot_id from coordinates when missing
        try:
            labels_df = _assign_plot_id_from_coordinates(labels_df, features_df, tolerance_deg=0.01)
        except Exception as e:
            logger.warning(f"Failed coordinate-based plot_id inference: {e}")

        # Standardize plot_id dtype across frames to improve merge reliability
        if "plot_id" in labels_df.columns:
            try:
                labels_df["plot_id"] = pd.to_numeric(labels_df["plot_id"], errors="coerce")
                # Cast float integers to pandas Int64 where appropriate
                def _to_int64_if_integer(x):
                    try:
                        return int(x) if pd.notna(x) and float(x).is_integer() else pd.NA
                    except Exception:
                        return pd.NA
                if pd.api.types.is_float_dtype(labels_df["plot_id"]):
                    labels_df["plot_id"] = labels_df["plot_id"].apply(_to_int64_if_integer).astype("Int64")
            except Exception as e:
                logger.warning(f"Failed to normalize plot_id dtype in labels: {e}")
        if "plot_id" in features_df.columns:
            try:
                features_df["plot_id"] = pd.to_numeric(features_df["plot_id"], errors="coerce")
                if pd.api.types.is_float_dtype(features_df["plot_id"]):
                    features_df["plot_id"] = features_df["plot_id"].apply(
                        lambda x: int(x) if pd.notna(x) and float(x).is_integer() else pd.NA
                    ).astype("Int64")
            except Exception as e:
                logger.warning(f"Failed to normalize plot_id dtype in features: {e}")

        # Build aligned merge keys only when both sides have the corresponding columns
        left_keys = []
        right_keys = []
        if "plot_id" in labels_df.columns and "plot_id" in features_df.columns:
            left_keys.append("plot_id")
            right_keys.append("plot_id")
        if "Planting_Date" in labels_df.columns and "planting_date" in features_df.columns:
            left_keys.append("Planting_Date")
            right_keys.append("planting_date")

        # Perform primary merge safely
        if left_keys:
            logger.info(f"Merging using keys left={left_keys} right={right_keys}")
            merged = labels_df.merge(features_df, left_on=left_keys, right_on=right_keys, how="left")
        else:
            # If no aligned keys, attempt plot_id-only if present, else leave for coordinate fallback
            if "plot_id" in labels_df.columns and "plot_id" in features_df.columns:
                logger.info("Merging using plot_id only")
                merged = labels_df.merge(features_df, on="plot_id", how="left")
            else:
                merged = labels_df.copy()

        # Detect rows where feature columns are entirely missing (labels are present, features are NaN)
        feature_indicator_cols = [
            c for c in [
                "precip_total", "gdd_sum", "gdd_peak",
                "mean_ndvi", "mean_evi", "mean_ndre", "mean_savi", "mean_ndwi", "mean_ndmi",
                "days_to_vt"
            ] if c in merged.columns
        ]
        if not feature_indicator_cols:
            logger.warning("No feature indicator columns found in merged frame; proceeding without fallback detection.")
        else:
            missing_features_mask = merged[feature_indicator_cols].isna().all(axis=1)
            missing_count = int(missing_features_mask.sum())
            total_count = int(len(merged))
            logger.info(
                f"Primary merge results: features missing for {missing_count}/{total_count} rows (by {feature_indicator_cols})."
            )

            if missing_count > 0:
                # Fallback: join by plot_id only (ignore planting date mismatches)
                merged_by_id = labels_df.merge(features_df, on="plot_id", how="left")

                if feature_indicator_cols:
                    missing_by_id = merged_by_id[feature_indicator_cols].isna().all(axis=1).sum()
                    logger.info(
                        f"plot_id-only merge: features missing for {int(missing_by_id)}/{total_count} rows."
                    )

                # If still many unmatched, attempt coordinate-based merge (rounded coords +- planting date)
                needs_coord_fallback = (
                    feature_indicator_cols and merged_by_id[feature_indicator_cols].isna().all(axis=1).any()
                )
                if needs_coord_fallback:
                    lbl = labels_df.copy()
                    feats = features_df.copy()
                    # Normalize coordinate column names in labels
                    if "Latitude" in lbl.columns and "latitude" not in lbl.columns:
                        lbl = lbl.rename(columns={"Latitude": "latitude"})
                    if "Longitude" in lbl.columns and "longitude" not in lbl.columns:
                        lbl = lbl.rename(columns={"Longitude": "longitude"})
                    if {"latitude", "longitude"}.issubset(lbl.columns) and {"latitude", "longitude"}.issubset(feats.columns):
                        # Round to 4 decimals to allow approximate match
                        for df_ in (lbl, feats):
                            df_["lat_round"] = pd.to_numeric(df_.get("latitude"), errors="coerce").round(4)
                            df_["lon_round"] = pd.to_numeric(df_.get("longitude"), errors="coerce").round(4)
                        # Prefer matching with planting date if available
                        left_on = [c for c in ["lat_round", "lon_round", "Planting_Date"] if c in lbl.columns]
                        right_on = [c for c in ["lat_round", "lon_round", "planting_date"] if c in feats.columns]
                        logger.info(f"Coordinate merge using keys left={left_on} right={right_on}")
                        if left_on and right_on:
                            merged_coords = lbl.merge(feats, left_on=left_on, right_on=right_on, how="left")
                        else:
                            merged_coords = lbl.merge(feats, on=["lat_round", "lon_round"], how="left")

                        if feature_indicator_cols:
                            missing_after_coords = merged_coords[feature_indicator_cols].isna().all(axis=1).sum()
                            logger.info(
                                f"Coordinate-based merge: features missing for {int(missing_after_coords)}/{total_count} rows."
                            )
                        merged = merged_coords
                    else:
                        logger.warning("Coordinate fallback unavailable due to missing latitude/longitude on one side.")
                        merged = merged_by_id
                else:
                    merged = merged_by_id

        def _json_safe_records(df: pd.DataFrame) -> list:
            recs = df.to_dict(orient="records")
            return [
                {k: (v.strftime("%Y-%m-%d") if isinstance(v, pd.Timestamp) else (None if pd.isna(v) else v)) for k, v in r.items()}
                for r in recs
            ]
        merged_records = _json_safe_records(merged)
        logger.info(f"Combined dataset rows: {len(merged_records)}")
        return merged_records

    @task(multiple_outputs=True)
    def train_models(dataset_records: list) -> dict:
        """Preprocess, split, run RandomizedSearchCV for XGB, LGBM, CatBoost, RandomForest; create ensemble; log to MLflow and S3.
        Accepts combined dataset records to ensure features include weather + indices.
        """
        # Set MLflow tracking
        tracking_uri = _get_tracking_uri()
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.environ.get("MLFLOW_S3_ENDPOINT_URL", os.environ.get("S3_ENDPOINT_URL", "http://minio:9000"))
        os.environ["AWS_ACCESS_KEY_ID"] = os.environ.get("S3_ACCESS_KEY", os.environ.get("MINIO_ROOT_USER", "minioadmin") or "minioadmin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.environ.get("S3_SECRET_KEY", os.environ.get("MINIO_ROOT_PASSWORD", "minioadmin") or "minioadmin")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("YieldPrediction_MultiModel")

        # Load and preprocess
        df = pd.DataFrame(dataset_records)
        df = _preprocess(df)
        target = "dry_harvest_kg/ha"
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        X_train, X_test, y_train, y_test = _split_features_labels(df, target)
        
        # Store execution date for S3 saving
        execution_date = datetime.utcnow()
        ds = _get_ds_from_airflow_env()
        try:
            execution_date = pd.to_datetime(ds).to_pydatetime()
        except Exception:
            pass

        results = {}
        trained_models = {}  # Store all trained models for ensemble
        all_predictions = {}  # Store predictions for comparison

        # =========================
        # XGBoost RandomizedSearchCV
        # =========================
        logger.info("Starting XGBoost RandomizedSearchCV...")
        xgb_base = xgb.XGBRegressor(
            random_state=42,
            n_estimators=600,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        xgb_dist = {
            "n_estimators": np.arange(400, 1501, 100),
            "learning_rate": np.linspace(0.01, 0.2, 10),
            "max_depth": np.arange(4, 13, 1),
            "subsample": np.linspace(0.6, 1.0, 9),
            "colsample_bytree": np.linspace(0.6, 1.0, 9),
        }
        xgb_rs = RandomizedSearchCV(
            xgb_base,
            xgb_dist,
            n_iter=20,
            cv=3,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            verbose=1,
        )
        xgb_rs.fit(X_train, y_train)
        xgb_best = xgb_rs.best_estimator_
        xgb_metrics = _eval_and_log("XGBoost_RandomSearch", xgb_best, X_test, y_test, xgb_rs.best_params_)
        mlflow.xgboost.log_model(xgb_best, name="xgb_model")
        results["XGBoost"] = {**xgb_metrics, "best_params": xgb_rs.best_params_}
        trained_models["xgb"] = xgb_best
        all_predictions["XGBoost"] = xgb_best.predict(X_test)
        logger.success(f"XGBoost completed - RMSE: {xgb_metrics['RMSE']:.4f}, R2: {xgb_metrics['R2']:.4f}")

        # =========================
        # LightGBM RandomizedSearchCV
        # =========================
        logger.info("Starting LightGBM RandomizedSearchCV...")
        lgb_base = lgb.LGBMRegressor(
            random_state=42,
            n_estimators=600,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
        lgb_dist = {
            "n_estimators": np.arange(400, 1501, 100),
            "learning_rate": np.linspace(0.01, 0.2, 10),
            "num_leaves": np.arange(31, 256, 16),
            "max_depth": [-1] + list(np.arange(4, 13, 1)),
            "subsample": np.linspace(0.6, 1.0, 9),
            "colsample_bytree": np.linspace(0.6, 1.0, 9),
        }
        lgb_rs = RandomizedSearchCV(
            lgb_base,
            lgb_dist,
            n_iter=20,
            cv=3,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            verbose=1,
        )
        lgb_rs.fit(X_train, y_train)
        lgb_best = lgb_rs.best_estimator_
        lgb_metrics = _eval_and_log("LightGBM_RandomSearch", lgb_best, X_test, y_test, lgb_rs.best_params_)
        mlflow.lightgbm.log_model(lgb_best, name="lgb_model")
        results["LightGBM"] = {**lgb_metrics, "best_params": lgb_rs.best_params_}
        trained_models["lgb"] = lgb_best
        all_predictions["LightGBM"] = lgb_best.predict(X_test)
        logger.success(f"LightGBM completed - RMSE: {lgb_metrics['RMSE']:.4f}, R2: {lgb_metrics['R2']:.4f}")

        # =========================
        # CatBoost RandomizedSearchCV
        # =========================
        logger.info("Starting CatBoost RandomizedSearchCV...")
        cat_base = CatBoostRegressor(
            random_seed=42,
            verbose=False,
            loss_function="RMSE",
            eval_metric="RMSE",
        )
        cat_dist = {
            "depth": np.arange(4, 11, 1),
            "learning_rate": np.linspace(0.01, 0.2, 10),
            "iterations": np.arange(500, 2001, 250),
        }
        cat_rs = RandomizedSearchCV(
            cat_base,
            cat_dist,
            n_iter=20,
            cv=3,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            verbose=1,
        )

        # Determine categorical indices (prefer encoded column if present)
        cat_cols = [c for c in ["crop_type_enc"] if c in X_train.columns]
        cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
        # Fit with categorical feature indices
        cat_rs.fit(X_train, y_train, **({"cat_features": cat_idx} if cat_idx else {}))
        cat_best = cat_rs.best_estimator_
        cat_metrics = _eval_and_log("CatBoost_RandomSearch", cat_best, X_test, y_test, cat_rs.best_params_)
        mlflow.catboost.log_model(cat_best, name="cat_model")
        results["CatBoost"] = {**cat_metrics, "best_params": cat_rs.best_params_}
        trained_models["cat"] = cat_best
        all_predictions["CatBoost"] = cat_best.predict(X_test)
        logger.success(f"CatBoost completed - RMSE: {cat_metrics['RMSE']:.4f}, R2: {cat_metrics['R2']:.4f}")

        # ============================
        # Random Forest RandomizedSearchCV
        # ============================
        logger.info("Starting Random Forest RandomizedSearchCV...")
        rf_base = RandomForestRegressor(
            random_state=42,
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
        )
        rf_dist = {
            "n_estimators": np.arange(100, 601, 50),
            "max_depth": list(np.arange(6, 15, 1)) + [None],
            "min_samples_split": np.arange(2, 11, 1),
            "min_samples_leaf": np.arange(1, 6, 1),
            "max_features": ["sqrt", "log2", 0.5, 0.7, None],
        }
        rf_rs = RandomizedSearchCV(
            rf_base,
            rf_dist,
            n_iter=20,
            cv=3,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            random_state=42,
            verbose=1,
        )
        rf_rs.fit(X_train, y_train)
        rf_best = rf_rs.best_estimator_
        rf_metrics = _eval_and_log("RandomForest_RandomSearch", rf_best, X_test, y_test, rf_rs.best_params_)
        mlflow.sklearn.log_model(rf_best, name="rf_model")
        results["RandomForest"] = {**rf_metrics, "best_params": rf_rs.best_params_}
        trained_models["rf"] = rf_best
        all_predictions["RandomForest"] = rf_best.predict(X_test)
        logger.success(f"Random Forest completed - RMSE: {rf_metrics['RMSE']:.4f}, R2: {rf_metrics['R2']:.4f}")

        # =====================
        # Ensemble Model (VotingRegressor)
        # =====================
        logger.info("Creating ensemble model (VotingRegressor)...")
        ensemble = VotingRegressor(
            estimators=[
                ("xgb", trained_models["xgb"]),
                ("lgb", trained_models["lgb"]),
                ("cat", trained_models["cat"]),
                ("rf", trained_models["rf"]),
            ],
            weights=None  # Equal weights, can be optimized later
        )
        ensemble.fit(X_train, y_train)
        ensemble_preds = ensemble.predict(X_test)
        ensemble_rmse = float(np.sqrt(mean_squared_error(y_test, ensemble_preds)))
        ensemble_mae = float(mean_absolute_error(y_test, ensemble_preds))
        ensemble_r2 = float(r2_score(y_test, ensemble_preds))
        
        # Log ensemble to MLflow
        with mlflow.start_run(run_name="Ensemble_VotingRegressor"):
            mlflow.log_params({
                "ensemble_type": "VotingRegressor",
                "base_models": "XGBoost,LightGBM,CatBoost,RandomForest",
                "weights": "equal"
            })
            mlflow.log_metrics({
                "RMSE": ensemble_rmse,
                "MAE": ensemble_mae,
                "R2": ensemble_r2
            })
            mlflow.sklearn.log_model(ensemble, name="ensemble_model")
        
        results["Ensemble"] = {
            "RMSE": ensemble_rmse,
            "MAE": ensemble_mae,
            "R2": ensemble_r2,
            "ensemble_type": "VotingRegressor",
            "base_models": ["XGBoost", "LightGBM", "CatBoost", "RandomForest"]
        }
        all_predictions["Ensemble"] = ensemble_preds
        logger.success(f"Ensemble completed - RMSE: {ensemble_rmse:.4f}, R2: {ensemble_r2:.4f}")

        # =====================
        # Create Predictions Comparison DataFrame
        # =====================
        predictions_df = pd.DataFrame({
            "actual": y_test.values,
            **{f"pred_{model}": preds for model, preds in all_predictions.items()}
        })
        
        # =====================
        # Create Model Comparison DataFrame
        # =====================
        comparison_data = []
        for model_name, model_results in results.items():
            comparison_data.append({
                "model": model_name,
                "RMSE": model_results.get("RMSE", None),
                "MAE": model_results.get("MAE", None),
                "R2": model_results.get("R2", None),
                "best_params": str(model_results.get("best_params", {})) if "best_params" in model_results else None
            })
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values("RMSE", ascending=True)
        
        # =====================
        # Save to S3/MinIO
        # =====================
        storage = DataStorage()
        s3_paths = storage.save_training_results(
            training_metrics=results,
            predictions_comparison=predictions_df,
            model_comparison=comparison_df,
            execution_date=execution_date
        )
        logger.success(f"Training results saved to S3: {s3_paths}")

        logger.success("Model training with GridSearchCV completed for XGBoost, LightGBM, CatBoost, RandomForest, and Ensemble")
        return {
            **results,
            "_s3_paths": s3_paths,
            "_comparison_df": comparison_df.to_dict(orient="records")
        }

    @task(multiple_outputs=False)
    def log_completion(metrics: dict) -> bool:
        logger.info(f"Training metrics summary: {metrics}")
        return True

    # Task graph
    plot_features_path = load_plot_features_path()
    labels_path = load_labels_path()
    combined_dataset = combine_features_and_labels(labels_path, plot_features_path)
    metrics = train_models(combined_dataset)
    done = log_completion(metrics)

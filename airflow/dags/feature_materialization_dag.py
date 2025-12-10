from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.exceptions import AirflowSkipException
import pandas as pd
import os
import sys
from datetime import datetime
from loguru import logger
from io import BytesIO

# Ensure project packages (src/, config/) are importable in Airflow
sys.path.insert(0, '/opt/airflow/dags/repo')

from src.storage import DataStorage
from src.labels import compute_labels_from_raw
from config.settings import get_settings

settings = get_settings()
BUCKET = settings.S3_BUCKET_NAME

default_args = {
    "owner": "nuru",
    "start_date": days_ago(1),
    "retries": 2,
}


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


def _compute_stage_label(row, stage_dates: dict) -> str:
    """Label the current stage for a given date using stage dates dict per plot."""
    # Stage order from vegetative to maturity
    stage_order = ["VE", "V2", "V6", "VT", "R1", "R4", "R6"]
    date = pd.to_datetime(row["date"]) if not pd.isna(row["date"]) else None
    planting_date = pd.to_datetime(stage_dates.get("planting_date")) if stage_dates.get("planting_date") else None
    if date is None:
        return None
    if planting_date and date < planting_date:
        return "pre_planting"
    current = None
    for stg in stage_order:
        sd = stage_dates.get(stg)
        if sd is None or pd.isna(sd):
            continue
        try:
            sd_dt = pd.to_datetime(sd)
        except Exception:
            continue
        if date >= sd_dt:
            current = stg
    if current is None and planting_date:
        return "post_planting_pre_VE"
    return current or "unknown"


with DAG(
    dag_id="feature_materialization",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    @task(multiple_outputs=False)
    def load_weather_path() -> str:
        storage = DataStorage()
        ds = _get_ds_from_airflow_env()

        validated_key = f"{settings.S3_BASE_PREFIX}/validated/weather/{ds}/weather_data.parquet"
        raw_key = f"{settings.S3_RAW_WEATHER_PREFIX}/{ds}/weather_data.parquet"

        try:
            storage.s3_client.head_object(Bucket=BUCKET, Key=validated_key)
            return f"s3://{BUCKET}/{validated_key}"
        except Exception:
            try:
                storage.s3_client.head_object(Bucket=BUCKET, Key=raw_key)
                return f"s3://{BUCKET}/{raw_key}"
            except Exception:
                try:
                    resp = storage.s3_client.list_objects_v2(
                        Bucket=BUCKET,
                        Prefix=f"{settings.S3_BASE_PREFIX}/validated/weather/"
                    )
                    contents = resp.get("Contents", [])
                    keys = [o["Key"] for o in contents if o["Key"].endswith("/weather_data.parquet")]
                    dated = []
                    for k in keys:
                        parts = k.split("/")
                        if len(parts) >= 2:
                            ds2 = parts[-2]
                            try:
                                dt = pd.to_datetime(ds2)
                                dated.append((k, dt))
                            except Exception:
                                pass
                    if dated:
                        dated.sort(key=lambda x: x[1], reverse=True)
                        latest_key = dated[0][0]
                        return f"s3://{BUCKET}/{latest_key}"
                except Exception:
                    pass
                try:
                    resp = storage.s3_client.list_objects_v2(
                        Bucket=BUCKET,
                        Prefix=f"{settings.S3_RAW_WEATHER_PREFIX}/"
                    )
                    contents = resp.get("Contents", [])
                    keys = [o["Key"] for o in contents if o["Key"].endswith("/weather_data.parquet")]
                    dated = []
                    for k in keys:
                        parts = k.split("/")
                        if len(parts) >= 2:
                            ds2 = parts[-2]
                            try:
                                dt = pd.to_datetime(ds2)
                                dated.append((k, dt))
                            except Exception:
                                pass
                    if dated:
                        dated.sort(key=lambda x: x[1], reverse=True)
                        latest_key = dated[0][0]
                        return f"s3://{BUCKET}/{latest_key}"
                except Exception:
                    pass
                raise AirflowSkipException("No weather data available")

    @task(multiple_outputs=False)
    def load_planting_dates_path() -> str:
        """Return S3 path for processed (inferred) planting dates."""
        ds = _get_ds_from_airflow_env()
        key = f"{settings.S3_PROCESSED_WEATHER_PREFIX}/{ds}/planting_dates.parquet"
        return f"s3://{BUCKET}/{key}"

    @task(multiple_outputs=False)
    def load_crop_stages_path() -> str:
        """Return S3 path for crop stages parquet."""
        ds = _get_ds_from_airflow_env()
        key = f"crop_stages/crop_stages_{ds}.parquet"
        return f"s3://{BUCKET}/{key}"

    @task(multiple_outputs=False)
    def load_satellite_indices_path() -> str:
        """Return S3 path for satellite indices parquet."""
        ds = _get_ds_from_airflow_env()
        key = f"{settings.S3_BASE_PREFIX}/processed/satellite/{ds}/all_indices.parquet"
        return f"s3://{BUCKET}/{key}"

    @task(multiple_outputs=False)
    def build_daily_features(weather_path: str, planting_path: str, crop_stages_path: str, satellite_path: str) -> str:
        """Join weather + satellite daily, label stages, compute GDD features, and persist daily features."""
        storage = DataStorage()

        def _key_from_path(s3_path: str) -> str:
            return s3_path.replace(f"s3://{BUCKET}/", "")

        # Load inputs
        weather_df = _read_parquet_from_s3(BUCKET, _key_from_path(weather_path), storage)
        planting_df = _read_parquet_from_s3(BUCKET, _key_from_path(planting_path), storage)
        stages_df = _read_parquet_from_s3(BUCKET, _key_from_path(crop_stages_path), storage)
        sat_df = _read_parquet_from_s3(BUCKET, _key_from_path(satellite_path), storage)

        # Normalize dtypes
        weather_df["date"] = pd.to_datetime(weather_df["date"], errors="coerce")
        sat_df["date"] = pd.to_datetime(sat_df["date"], errors="coerce")
        if "inferred_planting_date" in planting_df.columns:
            planting_df["inferred_planting_date"] = pd.to_datetime(planting_df["inferred_planting_date"], errors="coerce")
        if "planting_date" in stages_df.columns:
            stages_df["planting_date"] = pd.to_datetime(stages_df["planting_date"], errors="coerce")

        # Merge planting metadata into weather
        pfl = planting_df.rename(columns={"inferred_planting_date": "planting_date"})
        weather_merged = weather_df.merge(
            pfl[["plot_id", "planting_date", "confidence_score"]], on="plot_id", how="left"
        )

        # Compute daily GDD and cumulative since planting
        # GDD = ((Tmax + Tmin)/2) - 10.0, clipped at 0
        weather_merged["daily_gdd"] = (
            (weather_merged["max_temp_c"] + weather_merged["min_temp_c"]) / 2.0 - 10.0
        ).clip(lower=0)

        # Only accumulate from planting_date onwards
        weather_merged = weather_merged.sort_values(["plot_id", "date"])  # ensure order
        def _cum_gdd(group: pd.DataFrame) -> pd.Series:
            pd_series = group["planting_date"].iloc[0]
            if pd.isna(pd_series):
                return pd.Series([None] * len(group), index=group.index)
            mask = group["date"] >= pd_series
            cumsum = group.loc[mask, "daily_gdd"].fillna(0).cumsum()
            out = pd.Series([None] * len(group), index=group.index)
            out.loc[mask] = cumsum
            return out
        weather_merged["gdd_cumulative"] = weather_merged.groupby("plot_id", group_keys=False).apply(_cum_gdd)

        # Join satellite indices to weather by plot_id + date (left join to keep daily weather)
        daily = weather_merged.merge(sat_df, on=["plot_id", "date"], how="left")

        # Prepare stage dates mapping per plot
        stage_cols = ["VE", "V2", "V6", "VT", "R1", "R4", "R6"]
        for c in ["planting_date"] + stage_cols:
            if c in stages_df.columns:
                stages_df[c] = pd.to_datetime(stages_df[c], errors="coerce")

        stages_map = {
            int(row["plot_id"]): {**{sc: row.get(sc) for sc in stage_cols}, "planting_date": row.get("planting_date")}
            for _, row in stages_df.iterrows()
        }

        # Label stage for each daily row
        daily["current_stage"] = daily.apply(
            lambda r: _compute_stage_label(r, stages_map.get(int(r["plot_id"]), {})), axis=1
        )

        # Days since planting
        daily["days_since_planting"] = (daily["date"] - daily["planting_date"]).dt.days

        # Select feature columns (compact, training-ready)
        feature_cols = [
            "plot_id", "date", "latitude", "longitude",
            "max_temp_c", "min_temp_c", "precip_mm",
            "daily_gdd", "gdd_cumulative", "days_since_planting", "current_stage",
        ]
        # Include mean/cumulative indices if present
        for idx in ["NDVI", "EVI", "NDRE", "SAVI", "NDWI", "NDMI"]:
            lc = idx.lower()
            mean_col = f"mean_{lc}"
            cum_col = f"cumulative_{lc}"
            if mean_col in daily.columns:
                feature_cols.append(mean_col)
            if cum_col in daily.columns:
                feature_cols.append(cum_col)

        features_df = daily[feature_cols].copy()

        # Persist daily features
        try:
            s3_path = storage.save_features_daily(features_df)
        except Exception:
            logger.exception("Failed to save daily features to storage")
            raise

        return s3_path

    @task(multiple_outputs=False)
    def aggregate_plot_features(daily_features_path: str, crop_stages_path: str) -> str:
        """Aggregate plot-level features for training: NDVI/EVI means, precipitation total, GDD sum, days_to_vt."""
        storage = DataStorage()
        def _key_from_path(s3_path: str) -> str:
            return s3_path.replace(f"s3://{BUCKET}/", "")

        daily_df = _read_parquet_from_s3(BUCKET, _key_from_path(daily_features_path), storage)
        stages_df = _read_parquet_from_s3(BUCKET, _key_from_path(crop_stages_path), storage)

        # Normalize types
        daily_df["date"] = pd.to_datetime(daily_df["date"], errors="coerce")
        for c in ["planting_date", "VT"]:
            if c in stages_df.columns:
                stages_df[c] = pd.to_datetime(stages_df[c], errors="coerce")

        # Prep satellites metrics that may exist
        metric_means = {}
        for idx in ["ndvi", "evi", "ndre", "savi", "ndwi", "ndmi"]:
            col = f"mean_{idx}"
            if col in daily_df.columns:
                metric_means[idx] = col

        # Aggregations per plot
        agg_spec = {
            "precip_mm": "sum",
            "daily_gdd": "sum",
            "gdd_cumulative": "max",
            "latitude": "first",
            "longitude": "first",
        }
        for idx, col in metric_means.items():
            agg_spec[col] = "mean"

        plot_agg = daily_df.groupby("plot_id").agg(agg_spec).reset_index()

        # Join planting and VT for derived features
        # Include season and altitude to align with yield_features.md guidance
        base_cols = ["plot_id", "planting_date", "VT"]
        extra_cols = []
        if "confidence" in stages_df.columns:
            extra_cols.append("confidence")
        if "season" in stages_df.columns:
            extra_cols.append("season")
        if "altitude" in stages_df.columns:
            extra_cols.append("altitude")
        pfl = stages_df[base_cols + extra_cols].copy()
        plot_agg = plot_agg.merge(pfl, on="plot_id", how="left")

        # Derived features
        plot_agg["days_to_vt"] = (plot_agg["VT"] - plot_agg["planting_date"]).dt.days

        # Rename columns to training-friendly names
        rename_map = {
            "precip_mm": "precip_total",
            "daily_gdd": "gdd_sum",
            "gdd_cumulative": "gdd_peak",
        }
        plot_agg = plot_agg.rename(columns=rename_map)

        # Persist aggregated plot features
        try:
            s3_path = storage.save_features_plot(plot_agg)
        except Exception:
            logger.exception("Failed to save plot features to storage")
            raise

        return s3_path

    # Task graph
    weather_path = load_weather_path()
    planting_path = load_planting_dates_path()
    crop_stages_path = load_crop_stages_path()
    satellite_path = load_satellite_indices_path()

    daily_features_path = build_daily_features(weather_path, planting_path, crop_stages_path, satellite_path)
    plot_features_path = aggregate_plot_features(daily_features_path, crop_stages_path)

    @task(multiple_outputs=False)
    def load_labels_local_path() -> str:
        """Locate a local labels dataset produced by notebooks or preprocessing."""
        candidates = [
            "/opt/airflow/dags/repo/data/processed/yield.parquet",
            "/opt/airflow/dags/repo/final_for_yield.parquet",
            "/opt/airflow/dags/repo/notebooks/final_for_yield.parquet",
            "/opt/airflow/dags/repo/data/final_dataset.parquet",
            "/opt/airflow/dags/repo/data/processed/final_for_yield.parquet",
            "/opt/airflow/dags/repo/final_for_yield.csv",
            "/opt/airflow/dags/repo/notebooks/final_for_yield.csv",
            "/opt/airflow/dags/repo/final_dataset.csv",
        ]
        for p in candidates:
            if os.path.exists(p):
                logger.info(f"Found local labels dataset: {p}")
                return p
        logger.warning("No local labels dataset found in expected locations")
        return None

    @task(multiple_outputs=False)
    def export_labels_to_s3(labels_local_path: str) -> str:
        """Read local labels (parquet/csv) and export to S3 canonical labels path."""
        if not labels_local_path:
            logger.warning("Skipping labels export: no local labels file found")
            raise AirflowSkipException("No local labels file found")
        storage = DataStorage()
        try:
            if labels_local_path.endswith(".parquet"):
                df = pd.read_parquet(labels_local_path)
            elif labels_local_path.endswith(".csv"):
                df = pd.read_csv(labels_local_path)
            else:
                logger.warning(f"Unsupported labels file type: {labels_local_path}")
                raise AirflowSkipException("Unsupported labels file type")
        except Exception:
            logger.exception(f"Failed to read local labels file: {labels_local_path}")
            raise

        # Normalize keys for downstream training merge
        if "plot_no" in df.columns and "plot_id" not in df.columns:
            df = df.rename(columns={"plot_no": "plot_id"})
        # Ensure Planting_Date is present in Title case
        if "Planting_Date" in df.columns:
            df["Planting_Date"] = pd.to_datetime(df["Planting_Date"], errors="coerce")
        elif "planting_date" in df.columns:
            df["Planting_Date"] = pd.to_datetime(df["planting_date"], errors="coerce")

        s3_path = storage.save_labels(df)
        logger.success(f"Exported labels to {s3_path}")
        return s3_path

    @task(multiple_outputs=False)
    def resolve_raw_labels_source() -> str:
        """Resolve a raw labels source from S3, env URL, or local file."""
        ds = _get_ds_from_airflow_env()
        storage = DataStorage()

        # Prefer S3 raw labels if present
        s3_key_csv = f"{settings.S3_RAW_LABELS_PREFIX}/{ds}/{settings.RAW_LABELS_FILE_NAME}"
        s3_key_parquet = f"{settings.S3_RAW_LABELS_PREFIX}/{ds}/harvest.parquet"
        try:
            storage.s3_client.head_object(Bucket=BUCKET, Key=s3_key_csv)
            return f"s3://{BUCKET}/{s3_key_csv}"
        except Exception:
            try:
                storage.s3_client.head_object(Bucket=BUCKET, Key=s3_key_parquet)
                return f"s3://{BUCKET}/{s3_key_parquet}"
            except Exception:
                pass

        # Next: environment-provided CSV URL (e.g., Google Sheets export URL)
        if settings.HARVEST_SHEET_CSV_URL:
            return settings.HARVEST_SHEET_CSV_URL

        # Fallback: local raw file if present
        local_csv = "/opt/airflow/dags/repo/data/raw/harvest.csv"
        local_parquet = "/opt/airflow/dags/repo/data/raw/harvest.parquet"
        if os.path.exists(local_csv):
            return local_csv
        if os.path.exists(local_parquet):
            return local_parquet

        logger.warning("No raw labels source resolved")
        return None

    @task(multiple_outputs=False)
    def generate_labels_from_raw(raw_source: str) -> str:
        """Read raw harvest inputs, compute kg/ha labels, and save to S3."""
        if not raw_source:
            logger.warning("Skipping labels generation: no raw source available")
            raise AirflowSkipException("No raw labels source")

        storage = DataStorage()
        df = None
        try:
            if raw_source.startswith("s3://"):
                key = raw_source.replace(f"s3://{BUCKET}/", "")
                resp = storage.s3_client.get_object(Bucket=BUCKET, Key=key)
                if key.endswith(".parquet"):
                    df = pd.read_parquet(BytesIO(resp["Body"].read()))
                else:
                    df = pd.read_csv(BytesIO(resp["Body"].read()))
            elif raw_source.startswith("http://") or raw_source.startswith("https://"):
                df = pd.read_csv(raw_source)
            else:
                if raw_source.endswith(".parquet"):
                    df = pd.read_parquet(raw_source)
                else:
                    df = pd.read_csv(raw_source)
        except Exception:
            logger.exception(f"Failed to read raw labels from source: {raw_source}")
            raise

        labels_df = compute_labels_from_raw(df)
        if labels_df.empty:
            logger.warning("Computed labels dataframe is empty")
            raise AirflowSkipException("Labels computation yielded empty dataframe")

        s3_path = storage.save_labels(labels_df)
        logger.success(f"Generated and saved labels to {s3_path}")
        return s3_path

    raw_labels_source = resolve_raw_labels_source()
    labels_s3_path = generate_labels_from_raw(raw_labels_source)

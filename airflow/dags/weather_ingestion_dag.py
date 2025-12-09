# airflow/dags/weather_ingestion_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
import os
import json
import sys
from loguru import logger

# Add project root
sys.path.insert(0, '/opt/airflow/dags/repo')

# Internal modules
from src.ingestion.weather_client import WeatherAPIClient
from src.ingestion.planting_date_inference import PlantingDateInferenceEngine
from src.ingestion.crop_stage_determiner import determine_crop_stages_from_existing_data
from src.storage import DataStorage
from config.schemas import CropStageOutput, VTStageOutput, PlotInput
from config.settings import get_settings

settings = get_settings()

# -------------------------------------------------------------------------
# DAG DEFAULTS
# -------------------------------------------------------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': ['kennedy@nuru.solutions'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# -------------------------------------------------------------------------
# TASK 1: LOAD INPUT PLOTS
# -------------------------------------------------------------------------
def load_input_plots(**context):
    """
    Load initial plot list from CSV at data/processed/selected_farm_data.csv.
    Expected columns (case-insensitive, flexible):
      - plot_id (or plot_no / Plot_No)
      - latitude (or Latitude)
      - longitude (or Longitude)
      - altitude (optional)
      - season (optional; derived from planting_date if missing)
      - year (optional; derived from planting_date if missing)
      - planting_date (optional; can be Planting_Date or boxes_planting_date)
    """

    # Resolve CSV path inside the Airflow container
    repo_root = "/opt/airflow/dags/repo"
    csv_path = os.path.join(repo_root, "data", "processed", "selected_farm_data.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Plot input CSV not found at '{csv_path}'. Please place your file at data/processed/selected_farm_data.csv on the host."
        )

    df = pd.read_csv(csv_path)

    # Normalize column names
    cols = {c.lower(): c for c in df.columns}
    def has(name):
        return name in cols

    # Map identifiers
    if not has("plot_id"):
        for alt in ["plot_no", "plotid", "plot", "plot_no_id", "Plot_No"]:
            if alt.lower() in cols:
                df = df.rename(columns={cols[alt.lower()]: "plot_id"})
                break
    if "plot_id" not in df.columns:
        raise ValueError("selected_farm_data.csv must include a 'plot_id' (or 'plot_no') column")

    # Coordinates
    if not has("latitude") and "Latitude" in df.columns:
        df = df.rename(columns={"Latitude": "latitude"})
    if not has("longitude") and "Longitude" in df.columns:
        df = df.rename(columns={"Longitude": "longitude"})
    if not {"latitude", "longitude"}.issubset(df.columns):
        raise ValueError("selected_farm_data.csv must include latitude/longitude columns")

    # Planting date for deriving season/year if missing
    planting_col = None
    for cand in ["Planting_Date", "planting_date", "boxes_planting_date"]:
        if cand in df.columns:
            planting_col = cand
            break
    if planting_col:
        df[planting_col] = pd.to_datetime(df[planting_col], errors="coerce")

    if "season" in df.columns:
        def _norm_season(s):
            if pd.isna(s):
                return None
            sv = str(s).strip().lower()
            if sv in ["short rains", "short", "sr", "short_rains", "short-rains"]:
                return "Short Rains"
            if sv in ["long rains", "long", "lr", "long_rains", "long-rains"]:
                return "Long Rains"
            return None
        df["season"] = df["season"].apply(_norm_season)

    def derive_season(dt):
        if pd.isna(dt):
            return "Short Rains"
        m = dt.month
        if m in [9, 10, 11, 12]:
            return "Short Rains"
        if m in [2, 3, 4, 5]:
            return "Long Rains"
        return "Short Rains"

    if "season" in df.columns:
        if planting_col:
            df["season"] = df["season"].fillna(df[planting_col].apply(derive_season))
        else:
            df["season"] = df["season"].fillna("Short Rains")
    else:
        df["season"] = df[planting_col].apply(derive_season) if planting_col else "Short Rains"

    # Derive year if missing from planting date
    if "year" not in df.columns:
        if planting_col:
            df["year"] = df[planting_col].dt.year.bfill().ffill().fillna(2021).astype(int)
        else:
            df["year"] = 2021

    # Coerce types
    df["plot_id"] = pd.to_numeric(df["plot_id"], errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    # altitude optional (PlotInput defaults to 1200.0 if None)
    if "altitude" in df.columns:
        df["altitude"] = pd.to_numeric(df["altitude"], errors="coerce")

    # Drop rows with missing required fields
    required = ["plot_id", "latitude", "longitude", "season", "year"]
    clean = df.dropna(subset=required).copy()

    # Deduplicate per plot_id
    clean = clean.sort_values(["plot_id"]).drop_duplicates(subset=["plot_id"], keep="first")

    # Build PlotInput list
    plots_data = [
        {
            "plot_id": int(row["plot_id"]),
            "longitude": float(row["longitude"]),
            "latitude": float(row["latitude"]),
            "altitude": float(row["altitude"]) if not pd.isna(row.get("altitude", None)) else None,
            "season": str(row["season"]),
            "year": int(row["year"]),
        }
        for _, row in clean.iterrows()
    ]

    plots = [PlotInput(**p) for p in plots_data]

    # Push to XCom
    context["task_instance"].xcom_push(
        key="input_plots",
        value=[p.model_dump() for p in plots]
    )

    logger.info(f"[LOAD INPUT] Loaded {len(plots)} plots from {csv_path}")

    # Save metadata to storage
    storage = DataStorage()
    execution_date = context["execution_date"]
    date_str = execution_date.strftime("%Y%m%d")

    out_df = pd.DataFrame([p.model_dump() for p in plots])
    buf = pd.io.common.BytesIO()
    out_df.to_parquet(buf, index=False)
    buf.seek(0)

    storage.s3_client.put_object(
        Bucket=storage.settings.S3_BUCKET_NAME,
        Key=f"{storage.settings.S3_BASE_PREFIX}/metadata/plots/plots_{date_str}.parquet",
        Body=buf.getvalue()
    )

    return out_df


# -------------------------------------------------------------------------
# TASK 2: FETCH WEATHER
# -------------------------------------------------------------------------
def fetch_weather_data(**context):

    plot_dicts = context["task_instance"].xcom_pull(
        task_ids="load_input_plots", key="input_plots"
    )
    plots = [PlotInput(**p) for p in plot_dicts]

    client = WeatherAPIClient()
    weather_df = client.fetch_weather_batch(plots)

    logger.info(f"[WEATHER] Retrieved {len(weather_df)} rows")

    storage = DataStorage()
    upload_path = storage.save_raw_weather(
        weather_df, 
        context["execution_date"]
    )

    context["task_instance"].xcom_push(
        key="weather_data_path", 
        value=upload_path
    )

    return upload_path


# -------------------------------------------------------------------------
# TASK 3: VALIDATE WEATHER
# -------------------------------------------------------------------------
def validate_weather_data(**context):
    """Data quality checks"""

    raw_path = context["task_instance"].xcom_pull(
        task_ids="fetch_weather_data",
        key="weather_data_path"
    )

    storage = DataStorage()
    bucket = storage.settings.S3_BUCKET_NAME
    
    key = raw_path.replace(f"s3://{bucket}/", "")
    response = storage.s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_parquet(pd.io.common.BytesIO(response["Body"].read()))

    issues = []

    # 1. null checks
    null_cols = ["max_temp_c", "min_temp_c", "precip_mm"]
    nulls = df[null_cols].isnull().sum()
    if nulls.any():
        issues.append(f"Nulls found: {nulls.to_dict()}")

    # 2. range checks
    if (df["max_temp_c"] > 60).any() or (df["max_temp_c"] < -50).any():
        issues.append("Temperature outside global limits")

    # 3. duplicates
    dups = df.duplicated(subset=["plot_id", "date"]).sum()
    if dups > 0:
        issues.append(f"{dups} duplicates found")

    if issues:
        raise ValueError("Weather validation failed: " + json.dumps(issues, indent=2))

    logger.success("[VALIDATION] Weather data passed quality checks")

    # Save validated
    validated_key = key.replace("raw/weather", "validated/weather")
    validated_path = f"s3://{bucket}/{validated_key}"

    buf = pd.io.common.BytesIO()
    df.to_parquet(buf, index=False)
    buf.seek(0)

    storage.s3_client.put_object(
        Bucket=bucket,
        Key=validated_key,
        Body=buf.getvalue()
    )

    context["task_instance"].xcom_push(
        key="validated_weather_path",
        value=validated_path
    )

    return validated_path


# -------------------------------------------------------------------------
# TASK 4: INFER PLANTING DATE
# -------------------------------------------------------------------------
def infer_planting_dates(**context):

    weather_path = context["task_instance"].xcom_pull(
        task_ids="validate_weather_data",
        key="validated_weather_path"
    )
    plot_dicts = context["task_instance"].xcom_pull(
        task_ids="load_input_plots", key="input_plots"
    )

    storage = DataStorage()
    bucket = storage.settings.S3_BUCKET_NAME

    key = weather_path.replace(f"s3://{bucket}/", "")
    response = storage.s3_client.get_object(Bucket=bucket, Key=key)
    weather_df = pd.read_parquet(pd.io.common.BytesIO(response["Body"].read()))

    altitude_map = {p["plot_id"]: p["altitude"] for p in plot_dicts}

    engine = PlantingDateInferenceEngine()
    planting_df = engine.infer_batch(weather_df, altitude_map)

    upload_path = storage.save_inferred_planting_dates(
        planting_df,
        context["execution_date"]
    )

    context["task_instance"].xcom_push(
        key="planting_dates_path",
        value=upload_path
    )

    return upload_path


# -------------------------------------------------------------------------
# TASK 5: DETERMINE CROP STAGES
# -------------------------------------------------------------------------
def determine_crop_stages(**context):

    weather_path = context["task_instance"].xcom_pull(
        task_ids="validate_weather_data",
        key="validated_weather_path"
    )
    planting_path = context["task_instance"].xcom_pull(
        task_ids="infer_planting_dates",
        key="planting_dates_path"
    )
    plot_dicts = context["task_instance"].xcom_pull(
        task_ids="load_input_plots", key="input_plots"
    )

    storage = DataStorage()
    bucket = storage.settings.S3_BUCKET_NAME

    # Load weather
    weather_key = weather_path.replace(f"s3://{bucket}/", "")
    weather_df = pd.read_parquet(
        pd.io.common.BytesIO(
            storage.s3_client.get_object(Bucket=bucket, Key=weather_key)["Body"].read()
        )
    )

    # Load planting dates
    planting_key = planting_path.replace(f"s3://{bucket}/", "")
    planting_df = pd.read_parquet(
        pd.io.common.BytesIO(
            storage.s3_client.get_object(Bucket=bucket, Key=planting_key)["Body"].read()
        )
    )

    # Merge metadata
    plot_df = pd.DataFrame(plot_dicts)
    planting_df = planting_df.merge(
        plot_df[["plot_id", "altitude", "season"]],
        on="plot_id", 
        how="left"
    )

    # Run stage model
    stages_df, vt_df = determine_crop_stages_from_existing_data(
        weather_df,
        planting_df.rename(columns={"inferred_planting_date": "planting_date"}),
        gdd_base_temp=10.0
    )

    # Save outputs
    execution_date = context["execution_date"]
    crop_path = storage.save_crop_stages(stages_df, execution_date)
    vt_path = storage.save_vt_stages(vt_df, execution_date)

    context["task_instance"].xcom_push(key="crop_stages_path", value=crop_path)
    context["task_instance"].xcom_push(key="vt_stage_dates_path", value=vt_path)

    return {"crop": crop_path, "vt": vt_path}


# -------------------------------------------------------------------------
# TASK 6: FINAL LOGGING
# -------------------------------------------------------------------------
def log_pipeline_completion(**context):
    logger.success("Pipeline completed successfully")
    return True


# -------------------------------------------------------------------------
# DAG STRUCTURE
# -------------------------------------------------------------------------
with DAG(
    "weather_ingestion_planting_and_crop_stages",
    default_args=default_args,
    description="Weather ingestion → planting date inference → crop stage computation",
    schedule_interval="0 2 * * *",
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,
    tags=["weather", "ingestion", "crop_stages"],
):

    t1 = PythonOperator(
        task_id="load_input_plots",
        python_callable=load_input_plots,
    )

    t2 = PythonOperator(
        task_id="fetch_weather_data",
        python_callable=fetch_weather_data,
    )

    t3 = PythonOperator(
        task_id="validate_weather_data",
        python_callable=validate_weather_data,
    )

    t4 = PythonOperator(
        task_id="infer_planting_dates",
        python_callable=infer_planting_dates,
    )

    t5 = PythonOperator(
        task_id="determine_crop_stages",
        python_callable=determine_crop_stages,
    )

    t6 = PythonOperator(
        task_id="log_pipeline_completion",
        python_callable=log_pipeline_completion,
        trigger_rule="all_done",
    )

    # DAG DEPENDENCIES
    t1 >> t2 >> t3 >> t4 >> t5 >> t6

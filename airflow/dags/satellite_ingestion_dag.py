from airflow import DAG
from airflow.decorators import task
from airflow.utils.dates import days_ago
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
import pandas as pd
import os

import sys
# Ensure project packages (src/, config/) are importable in Airflow
sys.path.insert(0, '/opt/airflow/dags/repo')

from datetime import datetime

from src.ingestion.satellite_client import SatelliteClient
from src.storage import DataStorage
from config.schemas import CropStageOutput, VTStageOutput

from config.settings import get_settings
settings = get_settings()
BUCKET = settings.S3_BUCKET_NAME

# Build key inside task using Airflow context; macros don't render in Python values
# Template example: crop_stages/crop_stages_<ds>.parquet

default_args = {
    "owner": "nuru",
    "start_date": days_ago(1),
    "retries": 2,
}

with DAG(
    dag_id="satellite_ingestion",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    @task(multiple_outputs=False)
    def load_crop_stages() -> list:
        """Load crop stage parquet from S3 and validate."""
        # Use Airflow-exported env var for execution date (robust across versions)
        exec_dt_env = os.environ.get("AIRFLOW_CTX_EXECUTION_DATE")
        if exec_dt_env:
            try:
                ds = pd.to_datetime(exec_dt_env).strftime("%Y-%m-%d")
            except Exception:
                ds = (exec_dt_env.split("T")[0])
        else:
            ds = datetime.utcnow().strftime("%Y-%m-%d")

        key = f"crop_stages/crop_stages_{ds}.parquet"

        hook = S3Hook(aws_conn_id="aws_default")
        obj = hook.get_key(key=key, bucket_name=BUCKET)

        body = obj.get()["Body"].read()
        df = pd.read_parquet(pd.io.common.BytesIO(body))

        # Pydantic validation (CropStageOutput)
        validated = [
            CropStageOutput(**row)
            for row in df.to_dict(orient="records")
        ]
        return [v.model_dump() for v in validated]

    @task(multiple_outputs=False)
    def extract_vt_stage(crop_stages: list) -> list:
        """Extract VT stage and produce validated VTStageOutput schema."""
        df = pd.DataFrame(crop_stages)

        # Filter to VT only
        vt_df = df[df["VT"].notna()].copy()

        vt_rows = []
        for _, row in vt_df.iterrows():
            vt_rows.append(
                VTStageOutput(
                    plot_id=row["plot_id"],
                    vt_stage_date=row["VT"],
                    planting_date=row["planting_date"],
                    source="gdd_calculated",
                    confidence=row.get("confidence", 0.5)
                ).model_dump()
            )

        return vt_rows

    @task(multiple_outputs=False)
    def prepare_satellite_input(crop_stages: list, vt_stage_list: list) -> list:
        """Prepare input records with plot_id, latitude, longitude, planting_date, optional vt_date.
        If vt list is empty, fall back to planting_date only (SatelliteClient will default vt_date).
        """
        # Base from crop stages (has latitude/longitude)
        cs_df = pd.DataFrame(crop_stages)
        if cs_df.empty:
            return []

        # Normalize planting_date
        if "planting_date" in cs_df.columns:
            cs_df["planting_date"] = pd.to_datetime(cs_df["planting_date"], errors="coerce")

        # Select required columns
        cols = [c for c in ["plot_id", "latitude", "longitude", "planting_date"] if c in cs_df.columns]
        df = cs_df[cols].copy()

        # Attach vt_date if available
        vt_df = pd.DataFrame(vt_stage_list)
        if not vt_df.empty and "vt_stage_date" in vt_df.columns:
            vt_df["vt_stage_date"] = pd.to_datetime(vt_df["vt_stage_date"], errors="coerce")
            df = df.merge(vt_df[["plot_id", "vt_stage_date"]], on="plot_id", how="left")
            df["vt_date"] = df["vt_stage_date"]
            df.drop(columns=["vt_stage_date"], inplace=True)
        else:
            df["vt_date"] = pd.NaT

        # Drop rows without coordinates to avoid GEE geometry errors
        df = df.dropna(subset=["latitude", "longitude"])

        # Ensure JSON-serializable values for XCom (strings or nulls)
        for col in ["planting_date", "vt_date"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                df[col] = df[col].dt.strftime("%Y-%m-%d")
                df[col] = df[col].where(df[col].notna(), None)

        return df.to_dict(orient="records")

    @task(multiple_outputs=False)
    def run_sat_client(validated_plots: list) -> list:
        df = pd.DataFrame(validated_plots)
        client = SatelliteClient()

        result_df = client.batch_process_plots(df)

        return result_df.to_dict(orient="records")

    # Task execution sequence
    crop_stages = load_crop_stages()
    vt_stage_list = extract_vt_stage(crop_stages)
    sat_input = prepare_satellite_input(crop_stages, vt_stage_list)
    satellite_output = run_sat_client(sat_input)

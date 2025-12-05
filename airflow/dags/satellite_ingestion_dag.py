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

    @task
    def load_crop_stages() -> dict:
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

    @task
    def extract_vt_stage(crop_stages: list) -> dict:
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

    @task
    def prepare_satellite_input(vt_stage_list: list) -> dict:
        """Convert VTStageOutput → DataFrame ready for SatelliteClient."""
        df = pd.DataFrame(vt_stage_list)

        df["vt_date"] = pd.to_datetime(df["vt_stage_date"])
        df["planting_date"] = pd.to_datetime(df["planting_date"])

        df = df.rename(columns={
            "vt_date": "vt_date",
        })

        return df.to_dict(orient="records")

    @task
    def run_sat_client(validated_plots: list):
        df = pd.DataFrame(validated_plots)
        client = SatelliteClient()

        result_df = client.batch_process_plots(df)

        return result_df.to_dict(orient="records")

    # Task execution sequence
    crop_stages = load_crop_stages()
    vt_stage_list = extract_vt_stage(crop_stages)
    sat_input = prepare_satellite_input(vt_stage_list)
    satellite_output = run_sat_client(sat_input)

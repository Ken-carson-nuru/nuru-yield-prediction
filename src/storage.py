# src/storage.py
import boto3
from botocore.exceptions import ClientError
import pandas as pd
from io import BytesIO
from datetime import datetime
import json
from loguru import logger
import numpy as np

from config.settings import get_settings


def make_json_safe(obj):
    """Convert numpy/pandas objects to native Python types recursively."""
    if isinstance(obj, (np.generic,)):
        return obj.item()  # numpy -> python

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_safe(x) for x in obj]

    return obj  # already safe


class DataStorage:
    """Handle data storage to S3/MinIO dynamically"""

    def __init__(self):
        self.settings = get_settings()
        creds = self.settings.s3_credentials

        if self.settings.USE_LOCAL_STORAGE:
            # Local MinIO
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=creds.get("endpoint_url"),
                aws_access_key_id=creds.get("aws_access_key_id"),
                aws_secret_access_key=creds.get("aws_secret_access_key"),
            )
        else:
            # AWS S3
            self.s3_client = boto3.client(
                "s3",
                region_name=creds.get("region_name"),
                aws_access_key_id=creds.get("aws_access_key_id"),
                aws_secret_access_key=creds.get("aws_secret_access_key"),
            )

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Create bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.settings.S3_BUCKET_NAME)
            logger.info(f"Bucket '{self.settings.S3_BUCKET_NAME}' exists")
        except ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.settings.S3_BUCKET_NAME)
                logger.info(f"Created bucket '{self.settings.S3_BUCKET_NAME}'")
            except ClientError as e:
                logger.error(f"Failed to create bucket: {e}")
                raise

    def save_raw_weather(self, df: pd.DataFrame, execution_date: datetime = None) -> str:
        execution_date = execution_date or datetime.utcnow()
        date_str = execution_date.strftime("%Y-%m-%d")
        key = f"{self.settings.S3_RAW_WEATHER_PREFIX}/{date_str}/weather_data.parquet"

        buf = BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        self.s3_client.put_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key=key,
            Body=buf.getvalue(),
        )

        metadata = {
            "execution_date": execution_date.isoformat(),
            "row_count": int(len(df)),
            "plot_count": int(df['plot_id'].nunique()),
            "date_range": {
                "start": df['date'].min(),
                "end": df['date'].max()
            },
            "columns": df.columns.tolist(),
            "data_quality": {
                "null_counts": df.isnull().sum().to_dict(),
                "duplicate_count": int(df.duplicated().sum())
            }
        }

        # Make JSON-safe
        metadata = make_json_safe(metadata)

        metadata_key = f"{self.settings.S3_RAW_WEATHER_PREFIX}/{date_str}/metadata.json"
        self.s3_client.put_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2)
        )

        s3_path = f"s3://{self.settings.S3_BUCKET_NAME}/{key}"
        logger.success(f"Saved raw weather data to {s3_path}")
        return s3_path

    def save_inferred_planting_dates(self, df: pd.DataFrame, execution_date: datetime = None) -> str:
        execution_date = execution_date or datetime.utcnow()
        date_str = execution_date.strftime("%Y-%m-%d")
        key = f"{self.settings.S3_PROCESSED_WEATHER_PREFIX}/{date_str}/planting_dates.parquet"

        buf = BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        self.s3_client.put_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key=key,
            Body=buf.getvalue(),
        )

        metadata = {
            "execution_date": execution_date.isoformat(),
            "plot_count": int(len(df)),
            "average_confidence": float(df['confidence_score'].mean()),
            "min_confidence": float(df['confidence_score'].min()),
            "max_confidence": float(df['confidence_score'].max()),
            "date_range": {
                "earliest_planting": df['inferred_planting_date'].min(),
                "latest_planting": df['inferred_planting_date'].max()
            }
        }

        # Make JSON-safe
        metadata = make_json_safe(metadata)

        metadata_key = f"{self.settings.S3_PROCESSED_WEATHER_PREFIX}/{date_str}/planting_metadata.json"
        self.s3_client.put_object(
            Bucket=self.settings.S3_BUCKET_NAME,
            Key=metadata_key,
            Body=json.dumps(metadata, indent=2)
        )

        s3_path = f"s3://{self.settings.S3_BUCKET_NAME}/{key}"
        logger.success(f"Saved planting dates to {s3_path}")
        return s3_path

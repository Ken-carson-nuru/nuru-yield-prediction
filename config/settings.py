# config/settings.py

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Global settings loaded from environment variables"""

    # API Keys
    VISUAL_CROSSING_API_KEY: str = Field(..., env="VISUAL_CROSSING_API_KEY")
    VISUAL_CROSSING_BASE_URL: str = Field(
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline",
        env="VISUAL_CROSSING_BASE_URL",
    )

    # Storage
    USE_LOCAL_STORAGE: bool = Field(True, env="USE_LOCAL_STORAGE")
    S3_BUCKET_NAME: str = Field(..., env="S3_BUCKET_NAME")
    S3_BASE_PREFIX: str = Field("nuru-yield", env="S3_BASE_PREFIX")
    S3_ENDPOINT_URL: str = Field(None, env="S3_ENDPOINT_URL")
    S3_ACCESS_KEY: str = Field(None, env="S3_ACCESS_KEY")
    S3_SECRET_KEY: str = Field(None, env="S3_SECRET_KEY")

    # AWS
    AWS_REGION: str = Field("eu-north-1", env="AWS_REGION")
    AWS_ACCESS_KEY_ID: str = Field(None, env="AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str = Field(None, env="AWS_SECRET_ACCESS_KEY")

    # Weather API settings
    REQUEST_TIMEOUT: int = 30
    MAX_WORKERS: int = Field(4, env="MAX_WORKERS")

    # Planting inference settings
    BASE_ALTITUDE_M: float = 1200.0
    ALTITUDE_DELAY_FACTOR: float = 100.0
    MIN_PLANTING_RAINFALL_MM: float = 5.0
    MIN_PLANTING_TEMP_C: float = 18.0

    # Labels/raw harvest inputs
    HARVEST_SHEET_CSV_URL: str | None = Field(None, env="HARVEST_SHEET_CSV_URL")
    RAW_LABELS_FILE_NAME: str = Field("harvest.csv", env="RAW_LABELS_FILE_NAME")

    @property
    def S3_RAW_WEATHER_PREFIX(self):
        return f"{self.S3_BASE_PREFIX}/raw/weather"

    @property
    def S3_PROCESSED_WEATHER_PREFIX(self):
        return f"{self.S3_BASE_PREFIX}/processed/weather"

    @property
    def S3_RAW_LABELS_PREFIX(self):
        return f"{self.S3_BASE_PREFIX}/raw/labels"

    @property
    def s3_credentials(self):
        """Return correct credentials based on storage mode"""
        if self.USE_LOCAL_STORAGE:
            return {
                "endpoint_url": self.S3_ENDPOINT_URL,
                "aws_access_key_id": self.S3_ACCESS_KEY,
                "aws_secret_access_key": self.S3_SECRET_KEY,
            }
        else:
            return {
                "region_name": self.AWS_REGION,
                "aws_access_key_id": self.AWS_ACCESS_KEY_ID,
                "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY,
            }


def get_settings() -> Settings:
    # Load from .env
    return Settings(_env_file=".env", _env_file_encoding="utf-8")

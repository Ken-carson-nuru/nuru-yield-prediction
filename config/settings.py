import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    S3_BUCKET = os.getenv("S3_BUCKET")
    AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
    AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    FEAST_REPO_PATH = os.getenv("FEAST_REPO_PATH", "feature_repo")

settings = Settings()

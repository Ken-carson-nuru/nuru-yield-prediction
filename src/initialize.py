# src/initialize.py
import os
import ee
from google.oauth2 import service_account
from loguru import logger

def initialize_earth_engine():
    """
    Safe Earth Engine initialization for Airflow / Docker / local dev.
    - Works in Airflow containers
    - Picks key file from env: GEE_SERVICE_ACCOUNT_JSON
    - Idempotent (won't re-initialize if already initialized)
    """

    # If already initialized → do nothing
    try:
        ee.Number(1).getInfo()
        logger.debug("Earth Engine already initialized")
        return
    except Exception:
        pass

    key_file = os.getenv("GEE_SERVICE_ACCOUNT_JSON", "serene-bastion-406504-9c938287dece.json")
    key_file = key_file.strip()

    if not os.path.exists(key_file):
        raise FileNotFoundError(
            f"Earth Engine service account JSON not found: {key_file}. "
            f"Mount it into Airflow and set GEE_SERVICE_ACCOUNT_JSON env var."
        )

    logger.info(f"Initializing Earth Engine using service account: {key_file}")

    credentials = service_account.Credentials.from_service_account_file(
        key_file,
        scopes=['https://www.googleapis.com/auth/earthengine']
    )

    ee.Initialize(credentials)
    logger.success("Google Earth Engine initialized successfully")

# src/inference/feature_serving.py
"""
Feature serving for production inference.
Retrieves features from Feast feature store or S3/MinIO.
"""
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
from io import BytesIO
from loguru import logger

from src.storage import DataStorage
from config.settings import get_settings

settings = get_settings()


class FeatureServer:
    """Serve features for inference from S3/MinIO or Feast."""
    
    def __init__(self):
        self.storage = DataStorage()
        self.use_feast = False  # Can be enabled when Feast online store is configured
        
    def get_plot_features(
        self, 
        plot_ids: List[int],
        execution_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get plot-level features for inference.
        
        Args:
            plot_ids: List of plot IDs to get features for
            execution_date: Date string (YYYY-MM-DD) or None for latest
            
        Returns:
            DataFrame with features for requested plots
        """
        if execution_date is None:
            execution_date = datetime.utcnow().strftime("%Y-%m-%d")
        
        # Try to load from S3
        key = f"{settings.S3_BASE_PREFIX}/features/{execution_date}/plot_features.parquet"
        
        try:
            response = self.storage.s3_client.get_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=key
            )
            df = pd.read_parquet(BytesIO(response["Body"].read()))
            
            # Filter to requested plot_ids
            if "plot_id" in df.columns:
                df = df[df["plot_id"].isin(plot_ids)].copy()
            
            logger.info(f"Loaded features for {len(df)} plots from {key}")
            return df
            
        except Exception as e:
            logger.warning(f"Failed to load features from {key}: {e}")
            # Try latest available
            return self._get_latest_features(plot_ids)
    
    def _get_latest_features(self, plot_ids: List[int]) -> pd.DataFrame:
        """Get latest available features."""
        prefix = f"{settings.S3_BASE_PREFIX}/features/"
        
        try:
            response = self.storage.s3_client.list_objects_v2(
                Bucket=settings.S3_BUCKET_NAME,
                Prefix=prefix
            )
            
            contents = response.get("Contents", [])
            feature_files = [
                obj["Key"] for obj in contents 
                if obj["Key"].endswith("/plot_features.parquet")
            ]
            
            if not feature_files:
                raise ValueError("No feature files found in S3")
            
            # Sort by date (from path) and get latest
            feature_files.sort(reverse=True)
            latest_key = feature_files[0]
            
            response = self.storage.s3_client.get_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=latest_key
            )
            df = pd.read_parquet(BytesIO(response["Body"].read()))
            
            if "plot_id" in df.columns:
                df = df[df["plot_id"].isin(plot_ids)].copy()
            
            logger.info(f"Loaded latest features from {latest_key} for {len(df)} plots")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load latest features: {e}")
            raise
    
    def get_features_for_inference(
        self,
        plot_id: int,
        latitude: float,
        longitude: float,
        planting_date: str,
        season: Optional[str] = None,
        altitude: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Get or construct features for a single plot inference.
        If features exist in S3, use them. Otherwise, construct minimal feature set.
        
        Args:
            plot_id: Plot identifier
            latitude: Plot latitude
            longitude: Plot longitude
            planting_date: Planting date (YYYY-MM-DD)
            season: Season (Short Rains / Long Rains)
            altitude: Altitude in meters
            
        Returns:
            DataFrame with features ready for model prediction
        """
        # Try to get existing features first
        try:
            df = self.get_plot_features([plot_id])
            if not df.empty:
                return df
        except Exception:
            pass
        
        # If no existing features, create minimal feature set
        # This would require weather/satellite data - for now return basic structure
        logger.warning(f"No existing features for plot {plot_id}, creating minimal feature set")
        
        # Create minimal feature DataFrame with required columns
        # In production, you'd want to fetch weather/satellite data here
        features = pd.DataFrame([{
            "plot_id": plot_id,
            "latitude": latitude,
            "longitude": longitude,
            "planting_date": planting_date,
            "season": season or "Short Rains",
            "altitude": altitude or 1200.0,
            # Add default/placeholder values for model features
            # These should ideally be fetched from weather/satellite APIs
            "precip_total": 0.0,
            "gdd_sum": 0.0,
            "gdd_peak": 0.0,
            "mean_ndvi": 0.0,
            "mean_evi": 0.0,
            "days_to_vt": 0,
        }])
        
        # Add date features
        planting_dt = pd.to_datetime(planting_date)
        features["plant_month"] = planting_dt.month
        features["plant_doy"] = planting_dt.dayofyear
        
        return features


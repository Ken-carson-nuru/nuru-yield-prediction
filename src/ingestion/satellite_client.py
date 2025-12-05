# src/ingestion/satellite_client.py
from __future__ import annotations
import ee
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import get_settings
from src.ingestion.vegetation_indices import VegetationIndicesCalculator
from src.storage import DataStorage
from src.initialize import initialize_earth_engine


settings = get_settings()


class SatelliteClient:
    """Fetch and aggregate Sentinel-2 indices for plots using Google Earth Engine."""

    def __init__(
        self,
        max_workers: Optional[int] = None,
        buffer_meters: Optional[int] = None,
        sample_step_days: Optional[int] = None,
        window_days: Optional[int] = None,
        max_cloud_cover: Optional[float] = None,
        default_growth_days: Optional[int] = None,
    ):
        self.max_workers = max_workers or settings.MAX_WORKERS
        self.buffer_meters = buffer_meters or getattr(settings, "SAT_BUFFER_M", 40)
        self.sample_step_days = sample_step_days or getattr(settings, "SAT_SAMPLE_STEP_DAYS", 5)
        self.window_days = window_days or getattr(settings, "SAT_WINDOW_DAYS", 10)
        self.max_cloud_cover = max_cloud_cover or getattr(settings, "SAT_MAX_CLOUD_PCT", 60.0)
        self.default_growth_days = default_growth_days or getattr(settings, "SAT_DEFAULT_GROWTH_DAYS", 60)

        # Initialize Earth Engine (idempotent)
        initialize_earth_engine()

        self.indices_calculator = VegetationIndicesCalculator()
        self.indices_list = self.indices_calculator.get_all_indices()
        self.s2_collection = getattr(settings, "S2_COLLECTION", "COPERNICUS/S2_SR_HARMONIZED")

        # Storage client used to persist results
        self.storage = DataStorage()

    def _format_date_for_gee(self, date: datetime) -> str:
        return date.strftime("%Y-%m-%d")

    def _create_plot_geometry(self, latitude: float, longitude: float) -> ee.Geometry:
        point = ee.Geometry.Point([float(longitude), float(latitude)])
        return point.buffer(int(self.buffer_meters))

    def _get_sentinel2_collection(self, geometry: ee.Geometry, start_date: str, end_date: str) -> ee.ImageCollection:
        """Return filtered Sentinel-2 collection (server-side)."""
        return (
            ee.ImageCollection(self.s2_collection)
            .filterBounds(geometry)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", float(self.max_cloud_cover)))
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ee.EEException, TimeoutError)),
        reraise=True
    )
    def _sample_date_indices(self, plot_id: int, geometry: ee.Geometry, sample_date: datetime) -> Optional[Dict]:
        """
        Sample vegetation indices for a specific date window.
        Uses server-side reducers to get mean/min/max/std where possible.
        """
        try:
            window_start = sample_date - timedelta(days=self.window_days)
            window_end = sample_date + timedelta(days=self.window_days)

            coll = self._get_sentinel2_collection(
                geometry, self._format_date_for_gee(window_start), self._format_date_for_gee(window_end)
            )

            # If no images, bail out quickly
            size = coll.size().getInfo()
            if size == 0:
                logger.debug(f"No Sentinel-2 images for plot {plot_id} around {sample_date.date()}")
                return None

            # Compute indices for the whole collection (server-side)
            indices_collection = self.indices_calculator.calculate_for_image_collection(coll)

            # Reduce the collection to statistics per band/index
            # We compute the median image and then per-band reducers over the median
            # (median reduces temporal noise and is cheap)
            median_image = indices_collection.median()

            # Use a combined reducer to get mean, stdDev, min, max in one call
            reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(), "", True) \
                                     .combine(ee.Reducer.min(), "", True) \
                                     .combine(ee.Reducer.max(), "", True)

            region_stats = median_image.reduceRegion(
                reducer=reducer,
                geometry=geometry,
                scale=10,
                maxPixels=1e9
            )

            # Pull server-side dict to client
            stats = region_stats.getInfo()

            if not stats:
                logger.debug(f"No stats returned for plot {plot_id} on {sample_date.date()}")
                return None

            # Format result
            result = {
                "plot_id": int(plot_id),
                "date": sample_date.strftime("%Y-%m-%d"),
                "sample_date_iso": sample_date.isoformat(),
            }

            # Map stats keys to our expected index fields:
            # VegetationIndicesCalculator should prepare band names matching returned stats keys
            for index in self.indices_list:
                key_mean = f"{index}_mean"
                key_std = f"{index}_stdDev"
                key_min = f"{index}_min"
                key_max = f"{index}_max"

                # Some calculators name bands differently — be defensive
                result[f"mean_{index.lower()}"] = _safe_numeric(stats.get(key_mean) or stats.get(index + "_mean"))
                result[f"std_{index.lower()}"] = _safe_numeric(stats.get(key_std) or stats.get(index + "_stdDev"))
                result[f"min_{index.lower()}"] = _safe_numeric(stats.get(key_min) or stats.get(index + "_min"))
                result[f"max_{index.lower()}"] = _safe_numeric(stats.get(key_max) or stats.get(index + "_max"))

            return result

        except ee.EEException as e:
            logger.error(f"EE error sampling indices for plot {plot_id} on {sample_date.date()}: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error sampling indices for plot {plot_id} on {sample_date.date()}: {e}")
            return None

    def process_single_plot(
        self,
        plot_id: int,
        latitude: float,
        longitude: float,
        planting_date: datetime,
        vt_date: Optional[datetime] = None,
    ) -> List[Dict]:
        """Process a single plot and return list of measurement dicts (JSON-safe)."""
        try:
            if pd.isna(planting_date):
                logger.warning(f"Skipping plot {plot_id}: missing planting date")
                return []

            planting_date = pd.to_datetime(planting_date).to_pydatetime()

            if vt_date is None or pd.isna(vt_date):
                vt_date = planting_date + timedelta(days=self.default_growth_days)
                logger.info(f"Plot {plot_id}: using default VT at {vt_date.date()}")

            vt_date = pd.to_datetime(vt_date).to_pydatetime()
            if vt_date < planting_date:
                logger.warning(f"Plot {plot_id}: vt_date < planting_date; adjusting vt_date")
                vt_date = planting_date + timedelta(days=self.default_growth_days)

            geometry = self._create_plot_geometry(latitude, longitude)

            sample_dates = pd.date_range(start=planting_date, end=vt_date, freq=f"{self.sample_step_days}D")
            if len(sample_dates) == 0:
                logger.warning(f"Plot {plot_id}: no sample dates generated")
                return []

            results: List[Dict] = []
            logger.info(f"Plot {plot_id}: processing {len(sample_dates)} sample dates")

            for dt in sample_dates:
                # convert to python datetime for safety
                res = self._sample_date_indices(plot_id, geometry, dt.to_pydatetime())
                if res:
                    results.append(res)

            if not results:
                logger.warning(f"Plot {plot_id}: no valid samples")
                return []

            df = pd.DataFrame(results).sort_values("date")

            # compute cumulative columns for each mean index
            for index in self.indices_list:
                mean_col = f"mean_{index.lower()}"
                cum_col = f"cumulative_{index.lower()}"
                if mean_col in df.columns:
                    # convert to numeric safely then cumsum (NaNs handled)
                    df[mean_col] = pd.to_numeric(df[mean_col], errors="coerce")
                    df[cum_col] = df[mean_col].fillna(0).cumsum()

            # convert numpy types to python native for JSON-safety
            df = _json_safe_dataframe(df)

            return df.to_dict("records")

        except Exception as e:
            logger.exception(f"Error processing plot {plot_id}: {e}")
            return []

    def batch_process_plots(self, plots_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process multiple plots (DataFrame with columns: plot_id, latitude, longitude, planting_date, vt_date).
        Returns a clean DataFrame ready to be saved.
        """
        if isinstance(plots_data, (list, tuple)):
            plots_data = pd.DataFrame(plots_data)

        required_columns = ["plot_id", "latitude", "longitude", "planting_date"]
        for c in required_columns:
            if c not in plots_data.columns:
                raise ValueError(f"Missing required column: {c}")

        plots_data["planting_date"] = pd.to_datetime(plots_data["planting_date"])
        if "vt_date" in plots_data.columns:
            plots_data["vt_date"] = pd.to_datetime(plots_data["vt_date"])
        else:
            plots_data["vt_date"] = pd.NaT

        all_results: List[Dict] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            for _, row in plots_data.iterrows():
                future = executor.submit(
                    self.process_single_plot,
                    int(row["plot_id"]),
                    float(row["latitude"]),
                    float(row["longitude"]),
                    row["planting_date"],
                    row.get("vt_date", None),
                )
                futures[future] = int(row["plot_id"])

            for future in as_completed(futures):
                pid = futures[future]
                try:
                    res = future.result()
                    if res:
                        all_results.extend(res)
                        logger.info(f"Completed plot {pid}: {len(res)} samples")
                    else:
                        logger.warning(f"No results for plot {pid}")
                except Exception as e:
                    logger.exception(f"Error on plot {pid}: {e}")

        if not all_results:
            raise ValueError("No satellite data processed for any plots")

        df = pd.DataFrame(all_results)

        # standardize column set
        base_cols = ["plot_id", "date", "sample_date_iso"]
        stat_cols = []
        for index in self.indices_list:
            idx = index.lower()
            stat_cols.extend([f"mean_{idx}", f"std_{idx}", f"min_{idx}", f"max_{idx}", f"cumulative_{idx}"])

        all_cols = base_cols + stat_cols
        for col in all_cols:
            if col not in df.columns:
                df[col] = None

        df = df[all_cols]

        logger.success(
            f"Satellite ingestion complete: plots={df['plot_id'].nunique()}, samples={len(df)}, "
            f"dates={df['date'].min()}..{df['date'].max()}"
        )

        # persist to storage (S3/MinIO)
        try:
            self.storage.save_satellite_indices(df)
            logger.info("Saved satellite indices to storage")
        except Exception:
            logger.exception("Failed to save satellite indices")

        return _json_safe_dataframe(df)


# -----------------------
# Helper functions
# -----------------------
def _safe_numeric(x):
    """Return float if numeric, else None. Handles ee server returned types."""
    try:
        if x is None:
            return None
        # ee may return nested dicts with 'value'
        if isinstance(x, dict) and "value" in x:
            return float(x["value"])
        return float(x)
    except Exception:
        return None


def _json_safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert numpy/pandas types to native python and datetimes to ISO strings."""
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%dT%H:%M:%S")
        elif pd.api.types.is_numeric_dtype(df[col]):
            # cast numpy ints/floats to native python types on serialization stage
            df[col] = df[col].apply(lambda v: (int(v) if (pd.notnull(v) and float(v).is_integer()) else float(v)) if pd.notnull(v) else None)
        else:
            # ensure no pandas dtype objects
            df[col] = df[col].apply(lambda v: v if v is None or isinstance(v, (int, float, str, bool)) else str(v))
    return df

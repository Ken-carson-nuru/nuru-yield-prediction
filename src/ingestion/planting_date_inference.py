# src/planting_date_inference.py
from datetime import datetime, timedelta
from typing import Optional, Dict
import pandas as pd
from loguru import logger

from config.settings import get_settings
from config.schemas import InferredPlantingDate


class PlantingDateInferenceEngine:
    """
    Infer optimal planting dates using agronomic rules:
      1. Daily rainfall ≥ MIN_PLANTING_RAINFALL_MM
      2. Daily average temperature ≥ MIN_PLANTING_TEMP_C
      3. Altitude adjustment: delay 1 day per ALTITUDE_DELAY_FACTOR meters above BASE_ALTITUDE_M
    """

    def __init__(self):
        self.settings = get_settings()

    # -----------------------------------
    # Altitude Delay Calculation
    # -----------------------------------
    def _calculate_altitude_delay(self, altitude: float) -> int:
        """Delay planting based on altitude above BASE_ALTITUDE_M"""
        if altitude <= self.settings.BASE_ALTITUDE_M:
            return 0
        delay_days = int(
            (altitude - self.settings.BASE_ALTITUDE_M) / self.settings.ALTITUDE_DELAY_FACTOR
        )
        logger.debug(f"Altitude {altitude}m → {delay_days} days delay")
        return delay_days

    # -----------------------------------
    # Check if weather meets planting thresholds
    # -----------------------------------
    def _check_planting_conditions(
        self,
        temp_max: float,
        temp_min: float,
        precip: float
    ) -> Dict[str, bool]:
        """Evaluate rainfall and temperature thresholds"""
        avg_temp = (temp_max + temp_min) / 2
        return {
            "rainfall_threshold": precip >= self.settings.MIN_PLANTING_RAINFALL_MM,
            "temperature_threshold": avg_temp >= self.settings.MIN_PLANTING_TEMP_C
        }

    # -----------------------------------
    # Single Plot Planting Date Inference
    # -----------------------------------
    def infer_planting_date(
        self,
        plot_id: int,
        weather_df: pd.DataFrame,
        altitude: float
    ) -> Optional[InferredPlantingDate]:
        """Infer the optimal planting date for one plot"""
        if weather_df.empty:
            logger.warning(f"No weather data for plot {plot_id}")
            return None

        weather_df = weather_df.sort_values("date").reset_index(drop=True)
        altitude_delay = self._calculate_altitude_delay(altitude)

        for _, row in weather_df.iterrows():
            criteria = self._check_planting_conditions(
                row["max_temp_c"],
                row["min_temp_c"],
                row["precip_mm"]
            )

            if all(criteria.values()):
                # Apply altitude delay
                base_date = row["date"]
                adjusted_date = base_date + timedelta(days=altitude_delay)

                confidence = self._calculate_confidence(row, criteria)

                inferred = InferredPlantingDate(
                    plot_id=plot_id,
                    inferred_planting_date=adjusted_date,
                    confidence_score=confidence,
                    method="rainfall_temperature_threshold",
                    criteria_met=criteria,
                    altitude_adjustment_days=altitude_delay
                )

                logger.success(
                    f"Plot {plot_id} → Planting date: {adjusted_date.date()} "
                    f"(confidence: {confidence:.2f})"
                )
                return inferred

        logger.warning(f"No suitable planting date found for plot {plot_id}")
        return None

    # -----------------------------------
    # Confidence Scoring
    # -----------------------------------
    def _calculate_confidence(self, row: pd.Series, criteria: Dict[str, bool]) -> float:
        """Compute confidence score (0–1) for planting date"""
        confidence = 0.0
        if all(criteria.values()):
            confidence = 0.7

        # Rainfall bonus
        rainfall_excess = row["precip_mm"] - self.settings.MIN_PLANTING_RAINFALL_MM
        if rainfall_excess > 10:
            confidence += 0.15
        elif rainfall_excess > 5:
            confidence += 0.10

        # Temperature bonus
        avg_temp = (row["max_temp_c"] + row["min_temp_c"]) / 2
        temp_excess = avg_temp - self.settings.MIN_PLANTING_TEMP_C
        if temp_excess > 5:
            confidence += 0.15
        elif temp_excess > 2:
            confidence += 0.10

        return min(confidence, 1.0)

    # -----------------------------------
    # Batch Inference
    # -----------------------------------
    def infer_batch(
        self,
        weather_df: pd.DataFrame,
        altitude_map: Dict[int, float]
    ) -> pd.DataFrame:
        """Infer planting dates for all plots in a batch"""
        results = []

        for plot_id in weather_df["plot_id"].unique():
            plot_weather = weather_df[weather_df["plot_id"] == plot_id].copy()
            altitude = altitude_map.get(plot_id, self.settings.BASE_ALTITUDE_M)
            inferred = self.infer_planting_date(plot_id, plot_weather, altitude)
            if inferred:
                results.append(inferred.model_dump())

        if not results:
            raise ValueError("No planting dates inferred for any plots")

        results_df = pd.DataFrame(results)
        logger.success(
            f"Inferred planting dates for {len(results_df)} plots. "
            f"Average confidence: {results_df['confidence_score'].mean():.2f}"
        )
        return results_df

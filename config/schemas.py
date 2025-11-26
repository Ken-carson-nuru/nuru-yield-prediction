# config/schemas.py
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Literal, Dict
from pydantic_settings import BaseSettings


# ----------------------------------
# 1. Input Plot Schema
# ----------------------------------
class PlotInput(BaseModel):
    """Incoming plot record before processing."""

    plot_id: int = Field(..., gt=0, description="Unique plot identifier")
    longitude: float = Field(..., ge=-180, le=180)
    latitude: float = Field(..., ge=-90, le=90)
    altitude: Optional[float] = Field(
        None,
        ge=0,
        le=10000,
        description="Altitude in meters above sea level"
    )
    season: Literal["Short Rains", "Long Rains"] = Field(..., description="Planting season")
    year: int = Field(..., ge=2000, le=2100)

    @field_validator("altitude", mode="before")
    def default_altitude(cls, v):
        """Default altitude = 1200m if missing."""
        return 1200.0 if v is None else v


# ----------------------------------
# 2. Weather Observation Schema
# ----------------------------------
class WeatherDataPoint(BaseModel):
    """Validated single-day weather observation after API fetch."""

    plot_id: int
    latitude: float
    longitude: float
    date: datetime

    # Visual Crossing field mappings
    max_temp_c: float = Field(..., alias="tempmax")
    min_temp_c: float = Field(..., alias="tempmin")
    mean_temp_c: float = Field(..., alias="temp")
    precip_mm: float = Field(0.0, alias="precip")
    humidity_pct: Optional[float] = Field(None, alias="humidity")
    solar_energy_mjm2: Optional[float] = Field(None, alias="solarenergy")
    wind_speed_kmh: Optional[float] = Field(None, alias="windspeed")
    cloud_cover_pct: Optional[float] = Field(None, alias="cloudcover")

    model_config = {
        "populate_by_name": True,
        "validate_assignment": True
    }

    @field_validator("precip_mm", mode="before")
    def default_precip(cls, v):
        """Replace None precipitation with 0.0 mm."""
        return 0.0 if v is None else v

    @field_validator("max_temp_c", "min_temp_c", "mean_temp_c")
    def temperature_range(cls, v):
        """Ensure temperatures fall within realistic global range."""
        if not -50 <= v <= 60:
            raise ValueError(f"Temperature {v}°C is outside acceptable range (-50 to 60°C)")
        return v


# ----------------------------------
# 3. Planting Date Inference Result
# ----------------------------------
class InferredPlantingDate(BaseModel):
    """Output of the planting date inference logic."""

    plot_id: int
    inferred_planting_date: datetime
    actual_planting_date: Optional[datetime] = None
    confidence_score: float = Field(..., ge=0, le=1)
    method: str = Field(..., description="Inference method used")
    criteria_met: Dict[str, bool] = Field(..., description="Which criteria passed")
    altitude_adjustment_days: int = Field(0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "plot_id": 1,
                "inferred_planting_date": "2021-10-07T00:00:00",
                "confidence_score": 0.95,
                "method": "rainfall_temperature_threshold",
                "criteria_met": {
                    "rainfall_threshold": True,
                    "temperature_threshold": True,
                    "altitude_adjusted": True
                },
                "altitude_adjustment_days": 6
            }
        }
    }

# config/schemas.py
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Literal, Dict
from pydantic_settings import BaseSettings
import pandas as pd


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

class CropStageInput(BaseModel):
    """Input schema for crop stage determination"""
    plot_id: int = Field(..., gt=0, description="Unique plot identifier")
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    altitude: float = Field(..., ge=0, le=5000, description="Altitude in meters")
    season: Literal["Short Rains", "Long Rains"]
    planting_date: datetime = Field(..., description="Inferred planting date")
    
    @field_validator("altitude", mode="before")
    def default_altitude(cls, v):
        """Default altitude = 1200m if missing"""
        return 1200.0 if v is None else v


class CropStageOutput(BaseModel):
    """Output schema for crop stage determination"""
    plot_id: int
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    altitude: float
    season: str
    planting_date: str  # YYYY-MM-DD
    VE: Optional[str] = Field(None, description="Emergence date")
    V2: Optional[str] = Field(None, description="2-leaf stage date")
    V6: Optional[str] = Field(None, description="6-leaf stage date")
    VT: Optional[str] = Field(None, description="Tasseling date (CRITICAL for satellite)")
    R1: Optional[str] = Field(None, description="Silking date")
    R4: Optional[str] = Field(None, description="Dough stage date")
    R6: Optional[str] = Field(None, description="Physiological maturity date")
    method: Literal["gdd", "estimated"] = "gdd"
    confidence: float = Field(0.0, ge=0, le=1, description="Confidence score 0-1")
    
    class Config:
        json_schema_extra = {
            "example": {
                "plot_id": 1,
                "latitude": -0.499127,
                "longitude": 37.612253,
                "altitude": 1252.08,
                "season": "Short Rains",
                "planting_date": "2021-10-07",
                "VE": "2021-10-16",
                "V2": "2021-10-23",
                "V6": "2021-11-12",
                "VT": "2021-12-15",
                "R1": "2021-12-25",
                "R4": None,
                "R6": None,
                "method": "gdd",
                "confidence": 0.85
            }
        }
    
    @field_validator("VE", "V2", "V6", "VT", "R1", "R4", "R6", mode="before")
    def validate_date_format(cls, v):
        """Ensure dates are in correct format or None"""
        if v is None or pd.isna(v):
            return None
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, str):
            # Try to parse and reformat
            try:
                dt = pd.to_datetime(v)
                return dt.strftime("%Y-%m-%d")
            except:
                raise ValueError(f"Invalid date format: {v}")
        return v


class VTStageOutput(BaseModel):
    """Schema for VT stage dates (used by satellite ingestion)"""
    plot_id: int
    vt_stage_date: str  # YYYY-MM-DD
    planting_date: str  # YYYY-MM-DD
    source: Literal["gdd_calculated", "r1_fallback", "estimated"] = "gdd_calculated"
    confidence: float = Field(0.0, ge=0, le=1)
    note: Optional[str] = None
    
    @field_validator("vt_stage_date", "planting_date", mode="before")
    def validate_dates(cls, v):
        """Ensure dates are in YYYY-MM-DD format"""
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%d")
        if isinstance(v, str):
            try:
                pd.to_datetime(v)  # Validate it can be parsed
                return v
            except:
                raise ValueError(f"Invalid date: {v}")
        return v


class CropStageMetadata(BaseModel):
    """Metadata for crop stage determination run"""
    execution_date: str
    plot_count: int
    vt_stage_found: int
    vt_percentage: float
    average_confidence: float
    source_distribution: Dict[str, int]
    date_range: Dict[str, str]
    
    class Config:
        json_schema_extra = {
            "example": {
                "execution_date": "2024-01-15T10:30:00Z",
                "plot_count": 428,
                "vt_stage_found": 380,
                "vt_percentage": 88.8,
                "average_confidence": 0.72,
                "source_distribution": {
                    "gdd_calculated": 350,
                    "r1_fallback": 20,
                    "estimated": 10
                },
                "date_range": {
                    "earliest_planting": "2021-10-07",
                    "latest_planting": "2021-11-15",
                    "earliest_vt": "2021-11-20",
                    "latest_vt": "2022-01-10"
                }
            }
        }

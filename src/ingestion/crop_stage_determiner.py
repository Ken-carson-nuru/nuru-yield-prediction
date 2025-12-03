# src/ingestion/crop_stage_determiner.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from loguru import logger


class GDDCalculator:
    """Calculate Growing Degree Days for maize"""
    
    def __init__(self, base_temp: float = 10.0):
        self.base_temp = base_temp
        
    def calculate_daily_gdd(self, max_temp: float, min_temp: float) -> float:
        """
        Calculate GDD for a single day.
        
        Formula: GDD = ((T_max + T_min)/2) - T_base
        If result < 0, return 0
        """
        try:
            max_temp = float(max_temp)
            min_temp = float(min_temp)
        except ValueError:
            raise ValueError(f"Invalid temperature value: max_temp={max_temp}, min_temp={min_temp}")
        
        adjusted_avg_temp = (max_temp + min_temp) / 2
        return max(adjusted_avg_temp - self.base_temp, 0)


class CropStageDeterminer:
    """
    Determine maize development stages using accumulated GDD.
    Based on Kenya maize varieties classified by altitude.
    Uses EXISTING weather data from storage (not new API calls).
    """
    
    # GDD thresholds by maize maturity class (altitude-based)
    GDD_THRESHOLDS = {
        "early": {  # < 1200m
            "VE": 80,    # Emergence
            "V2": 150,   # 2-leaf stage
            "V6": 350,   # 6-leaf stage
            "VT": 850,   # Tasseling (CRITICAL for satellite)
            "R1": 950,   # Silking
            "R4": 1150,  # Dough stage
            "R6": 1300   # Physiological maturity
        },
        "medium": {  # 1200-1600m
            "VE": 100,
            "V2": 180,
            "V6": 400,
            "VT": 1000,
            "R1": 1100,
            "R4": 1350,
            "R6": 1500
        },
        "late": {  # > 1600m
            "VE": 120,
            "V2": 210,
            "V6": 450,
            "VT": 1150,
            "R1": 1250,
            "R4": 1550,
            "R6": 1800
        }
    }
    
    def __init__(self, gdd_base_temp: float = 10.0):
        """
        Args:
            gdd_base_temp: Base temperature for GDD calculation (default 10°C for maize)
        """
        self.gdd_calculator = GDDCalculator(base_temp=gdd_base_temp)
        
    def get_maize_class(self, altitude: float) -> str:
        """Determine maize maturity class based on altitude"""
        if altitude < 1200:
            return "early"
        elif altitude <= 1600:
            return "medium"
        else:
            return "late"
    
    def _find_planting_date_column(self, df: pd.DataFrame) -> str:
        """Find the planting date column in the dataframe"""
        # List of possible column names for planting date
        possible_names = [
            'inferred_planting_date',
            'planting_date', 
            'extracted_planting_date',
            'date',
            'optimal_planting_date',
            'plantingDate'
        ]
        
        # Check for exact matches first
        for col_name in possible_names:
            if col_name in df.columns:
                logger.info(f"Found planting date column: '{col_name}'")
                return col_name
        
        # Check for case-insensitive matches
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['plant', 'date', 'infer', 'extract']):
                logger.info(f"Using '{col}' as planting date column (case-insensitive match)")
                return col
        
        # If still not found, check for any datetime column
        date_cols = []
        for col in df.columns:
            try:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_cols.append(col)
            except:
                pass
        
        if date_cols:
            logger.info(f"Using datetime column '{date_cols[0]}' as planting date column")
            return date_cols[0]
        
        raise ValueError(f"Could not find planting date column. Available columns: {list(df.columns)}")
    
    def calculate_stage_dates(
        self,
        weather_df: pd.DataFrame,
        altitude: float,
        planting_date: datetime
    ) -> Dict[str, Optional[str]]:
        """
        Calculate crop stage dates using cumulative GDD.
        
        Args:
            weather_df: Daily weather data (already fetched from Visual Crossing)
            altitude: Plot altitude in meters
            planting_date: Inferred planting date
        
        Returns:
            Dictionary with stage dates (YYYY-MM-DD format)
        """
        if weather_df.empty or pd.isna(planting_date):
            return {stage: None for stage in self.GDD_THRESHOLDS["early"].keys()}
        
        # Get maize class and thresholds
        maize_class = self.get_maize_class(altitude)
        thresholds = self.GDD_THRESHOLDS[maize_class]
        
        # Filter weather data from planting date
        planting_date = pd.to_datetime(planting_date)
        weather_df = weather_df[weather_df["date"] >= planting_date].copy()
        
        if weather_df.empty:
            return {stage: None for stage in thresholds.keys()}
        
        # Calculate daily and cumulative GDD
        weather_df["daily_gdd"] = weather_df.apply(
            lambda row: self.gdd_calculator.calculate_daily_gdd(
                row["max_temp_c"], row["min_temp_c"]
            ), axis=1
        )
        
        weather_df["cum_gdd"] = weather_df["daily_gdd"].cumsum()
        
        # Determine stage dates
        stage_dates = {}
        for stage, gdd_req in thresholds.items():
            stage_row = weather_df[weather_df["cum_gdd"] >= gdd_req].head(1)
            if not stage_row.empty:
                stage_dates[stage] = stage_row.iloc[0]["date"].strftime("%Y-%m-%d")
            else:
                stage_dates[stage] = None
        
        return stage_dates
    
    def process_single_plot(
        self,
        plot_id: int,
        weather_df: pd.DataFrame,
        altitude: float,
        season: str,
        planting_date: datetime
    ) -> Optional[Dict]:
        """Process a single plot using existing weather data"""
        if weather_df.empty:
            logger.warning(f"No weather data for plot {plot_id}")
            return None
        
        # Calculate stage dates
        stage_dates = self.calculate_stage_dates(weather_df, altitude, planting_date)
        
        if stage_dates:
            # Extract plot coordinates from weather data if available
            latitude = weather_df.iloc[0].get("latitude") if "latitude" in weather_df.columns else None
            longitude = weather_df.iloc[0].get("longitude") if "longitude" in weather_df.columns else None
            
            result = {
                "plot_id": plot_id,
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "season": season,
                "planting_date": planting_date.strftime("%Y-%m-%d"),
                **stage_dates,
                "method": "gdd",
                "confidence": self._calculate_confidence(stage_dates)
            }
            logger.info(f"Processed plot {plot_id}: VT={stage_dates.get('VT')}")
            return result
        
        logger.warning(f"No crop stages determined for plot {plot_id}")
        return None
    
    def _calculate_confidence(self, stage_dates: Dict) -> float:
        """Calculate confidence score based on completeness of stage data"""
        valid_stages = sum(1 for date in stage_dates.values() if date is not None)
        total_stages = len(stage_dates)
        
        # Base confidence on % of stages found
        completeness = valid_stages / total_stages
        
        # Bonus if VT stage is found (most important for satellite)
        if stage_dates.get("VT") is not None:
            completeness += 0.2
        
        return min(round(completeness, 2), 1.0)
    
    def batch_process_plots(
        self,
        weather_data: pd.DataFrame,
        planting_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Process multiple plots using existing weather data.
        
        Args:
            weather_data: DataFrame with weather data for all plots
                Required columns: ['plot_id', 'date', 'max_temp_c', 'min_temp_c']
                Optional columns: ['latitude', 'longitude']
            planting_data: DataFrame with planting dates and metadata
                Should have: ['plot_id', planting_date_column, 'altitude', 'season']
        
        Returns:
            DataFrame with crop stage dates
        """
        # Validate weather data columns
        required_weather_cols = ['plot_id', 'date', 'max_temp_c', 'min_temp_c']
        for col in required_weather_cols:
            if col not in weather_data.columns:
                raise ValueError(f"Weather data missing required column: {col}")
        
        # Validate planting data has plot_id
        if 'plot_id' not in planting_data.columns:
            raise ValueError(f"Planting data missing required column: plot_id")
        
        # Find planting date column
        planting_date_col = self._find_planting_date_column(planting_data)
        
        # Ensure 'altitude' and 'season' are in planting_data
        if 'altitude' not in planting_data.columns:
            logger.warning("'altitude' column not found in planting_data")
            planting_data['altitude'] = 1200.0  # Default
        
        if 'season' not in planting_data.columns:
            logger.warning("'season' column not found in planting_data")
            planting_data['season'] = 'Short Rains'  # Default
        
        # Convert date columns
        weather_data["date"] = pd.to_datetime(weather_data["date"])
        planting_data[planting_date_col] = pd.to_datetime(planting_data[planting_date_col])
        
        results = []
        processed_plots = set()
        
        # Process each plot
        for _, planting_row in planting_data.iterrows():
            plot_id = planting_row["plot_id"]
            
            if plot_id in processed_plots:
                continue
                
            planting_date = planting_row[planting_date_col]
            altitude = planting_row["altitude"]
            season = planting_row["season"]
            
            # Get weather data for this plot
            plot_weather = weather_data[weather_data["plot_id"] == plot_id].copy()
            
            if plot_weather.empty:
                logger.warning(f"No weather data for plot {plot_id}")
                continue
            
            # Process this plot
            result = self.process_single_plot(
                plot_id, plot_weather, altitude, season, planting_date
            )
            
            if result:
                results.append(result)
                processed_plots.add(plot_id)
        
        if not results:
            raise ValueError("No crop stages determined for any plots")
        
        # Create DataFrame
        stages_df = pd.DataFrame(results)
        
        # Define column order
        stage_cols = ["VE", "V2", "V6", "VT", "R1", "R4", "R6"]
        base_cols = ["plot_id", "latitude", "longitude", "altitude", "season", 
                     "planting_date", "method", "confidence"]
        all_cols = base_cols + stage_cols
        
        # Ensure all columns exist
        for col in all_cols:
            if col not in stages_df.columns:
                stages_df[col] = None
        
        stages_df = stages_df[all_cols]
        
        # Calculate summary statistics
        n_plots = len(stages_df)
        n_vt_stages = stages_df["VT"].notna().sum()
        vt_percentage = (n_vt_stages / n_plots * 100) if n_plots > 0 else 0
        avg_confidence = stages_df["confidence"].mean()
        
        logger.success(
            f"Crop stage determination complete:\n"
            f"  - Processed {n_plots} plots\n"
            f"  - VT stage found for {n_vt_stages} plots ({vt_percentage:.1f}%)\n"
            f"  - Average confidence: {avg_confidence:.2f}"
        )
        
        # Log stage distribution
        for stage in stage_cols:
            n_found = stages_df[stage].notna().sum()
            logger.info(f"  - {stage}: {n_found} plots")
        
        return stages_df
    
    def get_vt_stage_dates(self, stages_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract VT stage dates for satellite ingestion.
        
        Args:
            stages_df: DataFrame from batch_process_plots
        
        Returns:
            DataFrame with plot_id and vt_stage_date for satellite ingestion
        """
        vt_dates = []
        
        for _, row in stages_df.iterrows():
            plot_id = row["plot_id"]
            vt_date = row["VT"]
            planting_date = row["planting_date"]
            
            if pd.notna(vt_date):
                vt_dates.append({
                    "plot_id": plot_id,
                    "vt_stage_date": vt_date,
                    "planting_date": planting_date,
                    "source": "gdd_calculated",
                    "confidence": row.get("confidence", 0.0),
                    "note": None
                })
            else:
                # Fallback 1: Use R1 date
                r1_date = row["R1"]
                if pd.notna(r1_date):
                    vt_dates.append({
                        "plot_id": plot_id,
                        "vt_stage_date": r1_date,
                        "planting_date": planting_date,
                        "source": "r1_fallback",
                        "confidence": max(row.get("confidence", 0.0) - 0.2, 0.0),
                        "note": "VT_not_found_used_R1"
                    })
                else:
                    # Fallback 2: Estimate VT as 55 days after planting
                    try:
                        planting_dt = pd.to_datetime(planting_date)
                        estimated_vt = planting_dt + timedelta(days=55)
                        vt_dates.append({
                            "plot_id": plot_id,
                            "vt_stage_date": estimated_vt.strftime("%Y-%m-%d"),
                            "planting_date": planting_date,
                            "source": "estimated",
                            "confidence": max(row.get("confidence", 0.0) - 0.4, 0.0),
                            "note": "VT_not_found_estimated_55_days"
                        })
                    except Exception as e:
                        logger.warning(f"Could not estimate VT date for plot {plot_id}: {e}")
                        continue
        
        vt_df = pd.DataFrame(vt_dates)
        
        if not vt_df.empty:
            # Log source distribution
            source_counts = vt_df["source"].value_counts()
            logger.info("VT dates source distribution:")
            for source, count in source_counts.items():
                percentage = (count / len(vt_df) * 100)
                logger.info(f"  - {source}: {count} plots ({percentage:.1f}%)")
            
            # Calculate average confidence by source
            if "confidence" in vt_df.columns:
                for source in source_counts.index:
                    avg_conf = vt_df[vt_df["source"] == source]["confidence"].mean()
                    logger.info(f"  - {source} average confidence: {avg_conf:.2f}")
        
        logger.info(f"Extracted VT dates for {len(vt_df)} plots")
        return vt_df


def determine_crop_stages_from_existing_data(
    weather_data: pd.DataFrame,
    planting_data: pd.DataFrame,
    gdd_base_temp: float = 10.0
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main entry point for crop stage determination using existing data.
    
    Args:
        weather_data: DataFrame with already-fetched weather data
        planting_data: DataFrame with inferred planting dates
        gdd_base_temp: Base temperature for GDD calculation (default 10°C for maize)
    
    Returns:
        Tuple of (stages_df, vt_dates_df)
    """
    logger.info("Starting crop stage determination...")
    logger.info(f"Weather data shape: {weather_data.shape}")
    logger.info(f"Planting data shape: {planting_data.shape}")
    logger.info(f"Planting data columns: {list(planting_data.columns)}")
    
    determiner = CropStageDeterminer(gdd_base_temp=gdd_base_temp)
    
    # Process all plots
    stages_df = determiner.batch_process_plots(weather_data, planting_data)
    
    # Extract VT dates for satellite ingestion
    vt_dates_df = determiner.get_vt_stage_dates(stages_df)
    
    return stages_df, vt_dates_df
# src/utils/crop_stage_utils.py
import pandas as pd
import json
from datetime import datetime
from loguru import logger
from typing import Dict

from src.ingestion.crop_stage_determiner import determine_crop_stages_from_existing_data
from src.storage import DataStorage


def run_crop_stage_determination(
    weather_data_path: str,
    planting_data_path: str,
    output_dir: str = "./output"
) -> None:
    """
    Run crop stage determination from local files (for testing).
    
    Args:
        weather_data_path: Path to weather data CSV/Parquet
        planting_data_path: Path to planting data CSV/Parquet
        output_dir: Directory to save outputs
    """
    # Load data
    logger.info(f"Loading weather data from {weather_data_path}")
    if weather_data_path.endswith('.parquet'):
        weather_df = pd.read_parquet(weather_data_path)
    else:
        weather_df = pd.read_csv(weather_data_path)
    
    logger.info(f"Loading planting data from {planting_data_path}")
    if planting_data_path.endswith('.parquet'):
        planting_df = pd.read_parquet(planting_data_path)
    else:
        planting_df = pd.read_csv(planting_data_path)
    
    # Run crop stage determination
    stages_df, vt_dates_df = determine_crop_stages_from_existing_data(
        weather_df, planting_df
    )
    
    # Save outputs
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    stages_path = f"{output_dir}/crop_stages_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    vt_path = f"{output_dir}/vt_stage_dates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    stages_df.to_csv(stages_path, index=False)
    vt_dates_df.to_csv(vt_path, index=False)
    
    logger.success(f"Crop stages saved to {stages_path}")
    logger.success(f"VT stage dates saved to {vt_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("CROP STAGE DETERMINATION SUMMARY")
    print("="*50)
    print(f"Total plots processed: {len(stages_df)}")
    print(f"VT stage found: {stages_df['VT'].notna().sum()}")
    print(f"Average confidence: {stages_df['confidence'].mean():.2f}")
    
    if not vt_dates_df.empty:
        source_counts = vt_dates_df['source'].value_counts()
        print("\nVT Dates Source Distribution:")
        for source, count in source_counts.items():
            print(f"  {source}: {count} plots")
    
    return stages_df, vt_dates_df


def save_crop_stages_to_storage(
    stages_df: pd.DataFrame,
    vt_dates_df: pd.DataFrame,
    execution_date: datetime = None
) -> Dict[str, str]:
    """
    Save crop stage data to storage (S3/MinIO).
    
    Returns:
        Dictionary with storage paths
    """
    storage = DataStorage()
    execution_date = execution_date or datetime.utcnow()
    
    # Save crop stages
    stages_path = storage.save_crop_stages(stages_df, execution_date)
    
    # Save VT dates
    vt_path = storage.save_vt_stage_dates(vt_dates_df, execution_date)
    
    return {
        "crop_stages_path": stages_path,
        "vt_stage_dates_path": vt_path
    }
# airflow/dags/weather_ingestion_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import pandas as pd
from loguru import logger
import sys

# Add src to path
sys.path.insert(0, '/opt/airflow/dags/repo')

from src.ingestion.weather_client import WeatherAPIClient
from src.ingestion.planting_date_inference import PlantingDateInferenceEngine
from src.storage import DataStorage
from config.schemas import PlotInput
from config.settings import get_settings

settings = get_settings()

# DAG default arguments
default_args = {
    'owner': 'nuru-mlops',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


def load_input_plots(**context):
    """Load input plots (demo/hardcoded or from DB/API)"""
    plots_data = [
        {"plot_id": 1, "longitude": 37.612253, "latitude": -0.499127, "altitude": 1252.08, "season": "Short Rains", "year": 2021},
        {"plot_id": 2, "longitude": 37.640289, "latitude": -0.475451, "altitude": 1234.50, "season": "Short Rains", "year": 2021},
        {"plot_id": 3, "longitude": 37.608650, "latitude": -0.535121, "altitude": 1226.40, "season": "Short Rains", "year": 2021},
    ]
    
    plots = [PlotInput(**plot) for plot in plots_data]
    
    context['task_instance'].xcom_push(
        key='input_plots',
        value=[plot.model_dump() for plot in plots]
    )
    
    logger.info(f"Loaded {len(plots)} plots for processing")


def fetch_weather_data(**context):
    """Fetch weather data from Visual Crossing API"""
    plots_data = context['task_instance'].xcom_pull(task_ids='load_input_plots', key='input_plots')
    plots = [PlotInput(**plot) for plot in plots_data]

    client = WeatherAPIClient()
    weather_df = client.fetch_weather_batch(plots)

    # 🔥 Convert ALL Timestamp objects to ISO strings before pushing to XCom
    weather_df = weather_df.applymap(
        lambda x: x.isoformat() if hasattr(x, "isoformat") else x
    )

    context['task_instance'].xcom_push(
        key='weather_data',
        value=weather_df.to_dict('records')
    )

    logger.info(f"Fetched {len(weather_df)} weather records")


def validate_weather_data(**context):
    """Validate weather data quality"""
    weather_data = context['task_instance'].xcom_pull(task_ids='fetch_weather_data', key='weather_data')
    df = pd.DataFrame(weather_data)

    issues = []

    # Check for nulls
    critical_cols = ['max_temp_c', 'min_temp_c', 'precip_mm']
    null_counts = df[critical_cols].isnull().sum()
    if null_counts.any():
        issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")

    # Check temperature ranges
    if (df['max_temp_c'] < -50).any() or (df['max_temp_c'] > 60).any():
        issues.append("Temperature values outside valid range (-50 to 60°C)")

    # Check duplicates
    duplicates = df.duplicated(subset=['plot_id', 'date']).sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate records")

    if issues:
        error_msg = "Data validation failed:\n" + "\n".join(issues)
        logger.error(error_msg)
        raise ValueError(error_msg)

    # 🔥 FIX: Convert timestamps before pushing XCom
    df = df.applymap(lambda x: x.isoformat() if hasattr(x, "isoformat") else x)

    context['task_instance'].xcom_push(
        key='validated_weather',
        value=df.to_dict('records')
    )

    logger.success("Weather data validation passed")



def infer_planting_dates(**context):
    """Infer planting dates using weather data"""
    weather_data = context['task_instance'].xcom_pull(task_ids='validate_weather_data', key='validated_weather')
    plots_data = context['task_instance'].xcom_pull(task_ids='load_input_plots', key='input_plots')

    weather_df = pd.DataFrame(weather_data)
    weather_df['date'] = pd.to_datetime(weather_df['date'])

    altitude_map = {plot['plot_id']: plot['altitude'] for plot in plots_data}
    engine = PlantingDateInferenceEngine()
    planting_df = engine.infer_batch(weather_df, altitude_map)

    planting_df = planting_df.applymap(
    lambda x: x.isoformat() if hasattr(x, "isoformat") else x
    )

    context['task_instance'].xcom_push(
        key='planting_dates',
        value=planting_df.to_dict('records')
    )
    logger.info(f"Inferred planting dates for {len(planting_df)} plots")


def save_to_storage(**context):
    """Save all data to S3/MinIO dynamically based on settings"""
    weather_data = context['task_instance'].xcom_pull(task_ids='validate_weather_data', key='validated_weather')
    planting_data = context['task_instance'].xcom_pull(task_ids='infer_planting_dates', key='planting_dates')

    weather_df = pd.DataFrame(weather_data)
    planting_df = pd.DataFrame(planting_data)

    weather_df['date'] = pd.to_datetime(weather_df['date'])
    planting_df['inferred_planting_date'] = pd.to_datetime(planting_df['inferred_planting_date'])

    storage = DataStorage()

    # Save using dynamic paths from settings
    weather_path = storage.save_raw_weather(weather_df)
    planting_path = storage.save_inferred_planting_dates(planting_df)

    context['task_instance'].xcom_push(
        key='storage_paths',
        value={'weather_path': weather_path, 'planting_path': planting_path}
    )

    logger.info(f"Weather data saved to {weather_path}")
    logger.info(f"Planting dates saved to {planting_path}")


# DAG definition
with DAG(
    'weather_ingestion_and_planting_inference',
    default_args=default_args,
    description='Fetch weather data and infer planting dates',
    schedule_interval='0 2 * * *',  # Daily 2 AM UTC
    start_date=days_ago(1),
    catchup=False,
    tags=['weather', 'ingestion', 'data-pipeline'],
) as dag:

    load_plots = PythonOperator(
        task_id='load_input_plots',
        python_callable=load_input_plots,
        provide_context=True,
    )

    fetch_weather = PythonOperator(
        task_id='fetch_weather_data',
        python_callable=fetch_weather_data,
        provide_context=True,
    )

    validate_weather = PythonOperator(
        task_id='validate_weather_data',
        python_callable=validate_weather_data,
        provide_context=True,
    )

    infer_planting = PythonOperator(
        task_id='infer_planting_dates',
        python_callable=infer_planting_dates,
        provide_context=True,
    )

    save_data = PythonOperator(
        task_id='save_to_storage',
        python_callable=save_to_storage,
        provide_context=True,
    )

    # Task dependencies
    load_plots >> fetch_weather >> validate_weather >> infer_planting >> save_data

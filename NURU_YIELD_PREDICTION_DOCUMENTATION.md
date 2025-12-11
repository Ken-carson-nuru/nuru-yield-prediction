# Nuru Yield Prediction – Full MLOps System Documentation

**Version:** 1.0  
**Date:** December 2025  
**Project:** Agricultural Yield Prediction using Satellite & Weather Data

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Architecture Overview](#architecture-overview)
3. [Technology Stack](#technology-stack)
4. [Data Pipeline Architecture](#data-pipeline-architecture)
5. [Detailed Pipeline Workflows](#detailed-pipeline-workflows)
6. [Data Storage Structure](#data-storage-structure)
7. [Key Features & Capabilities](#key-features--capabilities)
8. [Deployment Architecture](#deployment-architecture)
9. [Configuration & Environment](#configuration--environment)
10. [Data Flow Diagrams](#data-flow-diagrams)
11. [API Integrations](#api-integrations)
12. [Model Training Details](#model-training-details)
13. [Monitoring & Drift Detection](#monitoring--drift-detection)

---

## Executive Summary

The **Nuru Yield Prediction System** is a comprehensive MLOps platform designed to predict agricultural crop yields using a combination of:

- **Satellite Imagery** (Sentinel-2 via Google Earth Engine)
- **Weather Data** (Visual Crossing API)
- **Crop Metadata** (Plot locations, seasons, planting dates)
- **Ensemble Machine Learning Models** (XGBoost, LightGBM, CatBoost, RandomForest, Ensemble Voting)

The system implements a complete end-to-end pipeline from raw data ingestion to model deployment, with automated feature engineering, model training, and production monitoring.

### Key Capabilities

- Automated weather data ingestion and validation
- Intelligent planting date inference using agronomic rules
- Crop phenological stage determination (GDD-based)
- Satellite vegetation index extraction (NDVI, EVI, NDRE, SAVI, NDWI, NDMI)
- Feature store integration (Feast)
- Automated model training with hyperparameter tuning
- Production drift detection and monitoring
- Full containerized deployment with Docker

---

## Architecture Overview

The system follows a **5-stage MLOps pipeline** orchestrated by Apache Airflow:

```
┌─────────────────────────────────────────────────────────────┐
│                    AIRFLOW ORCHESTRATION                     │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   Weather     │   │   Satellite   │   │   Feature    │
│  Ingestion    │   │   Ingestion   │   │Materialization│
└──────────────┘   └──────────────┘   └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │   Model      │
                    │   Training   │
                    └──────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │    Drift     │
                    │  Detection   │
                    └──────────────┘
```

### System Components

1. **Data Ingestion Layer**
   - Weather API client (Visual Crossing)
   - Satellite client (Google Earth Engine)
   - Planting date inference engine
   - Crop stage determiner

2. **Feature Engineering Layer**
   - Daily feature computation (GDD, cumulative indices)
   - Plot-level aggregations
   - Stage labeling

3. **Storage Layer**
   - MinIO/S3 object storage
   - PostgreSQL (metadata)
   - Feast feature store

4. **ML Layer**
   - Model training (XGBoost, LightGBM, CatBoost, RandomForest, Ensemble VotingRegressor)
   - MLflow tracking and registry
   - Hyperparameter tuning (GridSearchCV)

5. **Monitoring Layer**
   - Drift detection (PSI)
   - Feature monitoring
   - Alert system

---

## Technology Stack

### Core Technologies

| Component | Technology | Version/Purpose |
|-----------|-----------|-----------------|
| **Orchestration** | Apache Airflow | 2.9.2 |
| **Feature Store** | Feast | Latest |
| **ML Tracking** | MLflow | Latest |
| **Storage** | MinIO / AWS S3 | S3-compatible |
| **Database** | PostgreSQL | 14 |
| **Containerization** | Docker & Docker Compose | Latest |
| **Satellite Data** | Google Earth Engine | earthengine-api |
| **Weather Data** | Visual Crossing API | REST API |

### ML Libraries

- **XGBoost** - Gradient boosting regressor
- **LightGBM** - Fast gradient boosting
- **CatBoost** - Gradient boosting with categorical support
- **RandomForest** (scikit-learn) - Bagging ensemble baseline
- **VotingRegressor** (scikit-learn) - Ensemble over all four models
- **scikit-learn** - Preprocessing, metrics, GridSearchCV
- **pandas** - Data manipulation
- **numpy** - Numerical computations

### Python Packages

- **Pydantic** - Data validation and settings management
- **loguru** - Structured logging
- **tenacity** - Retry logic with exponential backoff
- **boto3** - AWS S3/MinIO client
- **requests** - HTTP client for APIs

---

## Data Pipeline Architecture

### Pipeline Stages

The system consists of **5 main Airflow DAGs**:

1. **Weather Ingestion & Crop Stages** (`weather_ingestion_planting_and_crop_stages`)
   - Schedule: Daily at 2:00 AM
   - Purpose: Fetch weather, infer planting dates, calculate crop stages

2. **Satellite Ingestion** (`satellite_ingestion`)
   - Schedule: Manual trigger (depends on crop stages)
   - Purpose: Extract vegetation indices from Sentinel-2 imagery

3. **Feature Materialization** (`feature_materialization`)
   - Schedule: Manual trigger
   - Purpose: Join weather + satellite data, create training features

4. **Model Training** (`model_training_dag_weekly`)
   - Schedule: Weekly on Mondays at 3:00 AM
   - Purpose: Train ensemble models with GridSearchCV

5. **Drift Detection** (`drift_detection_dag`)
   - Schedule: Daily at 4:00 AM
   - Purpose: Monitor feature drift using PSI

---

## Detailed Pipeline Workflows

### 1. Weather Ingestion & Crop Stages DAG

**DAG ID:** `weather_ingestion_planting_and_crop_stages`  
**Schedule:** `0 2 * * *` (Daily at 2:00 AM)  
**Tags:** `weather`, `ingestion`, `crop_stages`

#### Task Flow

```
load_input_plots
    ↓
fetch_weather_data
    ↓
validate_weather_data
    ↓
infer_planting_dates
    ↓
determine_crop_stages
    ↓
log_pipeline_completion
```

#### Task Details

##### Task 1: Load Input Plots
- **Input:** `data/processed/selected_farm_data.csv`
- **Process:**
  - Reads plot metadata with flexible column naming
  - Handles: `plot_id`, `plot_no`, `Plot_No`
  - Extracts: `latitude`, `longitude`, `altitude`, `season`, `year`
  - Derives season from planting date if missing
  - Validates using Pydantic `PlotInput` schema
- **Output:** List of validated plots pushed to XCom

##### Task 2: Fetch Weather Data
- **Client:** `WeatherAPIClient`
- **Process:**
  - Batch processing with ThreadPoolExecutor (4 workers default)
  - Calls Visual Crossing API for each plot
  - Season-based date ranges:
    - **Short Rains:** Sep 1 - Dec 31
    - **Long Rains:** Feb 1 - May 31
  - Retry logic: 5 attempts with exponential backoff
  - Rate limiting handling (429 responses)
  - Validates responses using `WeatherDataPoint` schema
- **Output:** Raw weather DataFrame saved to S3

##### Task 3: Validate Weather Data
- **Quality Checks:**
  - Null checks on critical columns (`max_temp_c`, `min_temp_c`, `precip_mm`)
  - Temperature range validation (-50°C to 60°C)
  - Duplicate detection (plot_id + date)
- **Output:** Validated weather data saved to S3

##### Task 4: Infer Planting Dates
- **Engine:** `PlantingDateInferenceEngine`
- **Agronomic Rules:**
  1. **Rainfall Threshold:** ≥ 5mm daily precipitation
  2. **Temperature Threshold:** ≥ 18°C average temperature
  3. **Altitude Adjustment:** +1 day per 100m above 1200m baseline
- **Process:**
  - Iterates through weather data chronologically
  - Finds first day meeting both thresholds
  - Applies altitude-based delay
  - Calculates confidence score
- **Output:** Planting dates with confidence scores saved to S3

##### Task 5: Determine Crop Stages
- **Method:** Growing Degree Days (GDD) with base temperature 10°C
- **Stages Calculated:**
  - **VE** - Emergence
  - **V2** - 2-leaf stage
  - **V6** - 6-leaf stage
  - **VT** - Tasseling (CRITICAL for satellite timing)
  - **R1** - Silking
  - **R4** - Dough stage
  - **R6** - Physiological maturity
- **Process:**
  - Uses weather data + planting dates
  - Calculates cumulative GDD for each stage
  - Applies stage-specific GDD thresholds
- **Output:** Crop stages DataFrame saved to S3

---

### 2. Satellite Ingestion DAG

**DAG ID:** `satellite_ingestion`  
**Schedule:** Manual trigger (depends on crop stages availability)  
**Tags:** `satellite`, `ingestion`

#### Task Flow

```
load_crop_stages
    ↓
extract_vt_stage
    ↓
prepare_satellite_input
    ↓
run_sat_client
```

#### Task Details

##### Task 1: Load Crop Stages
- **Input:** S3 path `crop_stages/crop_stages_{YYYY-MM-DD}.parquet`
- **Process:**
  - Falls back to latest available if execution date not found
  - Validates using `CropStageOutput` Pydantic schema
- **Output:** Validated crop stages list

##### Task 2: Extract VT Stage
- **Purpose:** Extract tasseling (VT) dates for satellite timing
- **Process:**
  - Filters crop stages for non-null VT dates
  - Creates `VTStageOutput` records
- **Output:** VT stage dates list

##### Task 3: Prepare Satellite Input
- **Process:**
  - Merges crop stages with plot coordinates
  - Attaches VT dates if available
  - Falls back to planting_date + 60 days if VT missing
  - Ensures JSON-serializable date formats
- **Output:** Input records for satellite processing

##### Task 4: Run Satellite Client
- **Client:** `SatelliteClient` with Google Earth Engine
- **Process:**
  - **Collection:** Sentinel-2 Harmonized (`COPERNICUS/S2_SR_HARMONIZED`)
  - **Sampling:** Every 5 days from planting to VT
  - **Window:** ±10 days around each sample date
  - **Cloud Filter:** < 60% cloud cover
  - **Buffer:** 40m radius around plot coordinates
  - **Indices Calculated:**
    - NDVI (Normalized Difference Vegetation Index)
    - EVI (Enhanced Vegetation Index)
    - NDRE (Normalized Difference Red Edge)
    - SAVI (Soil-Adjusted Vegetation Index)
    - NDWI (Normalized Difference Water Index)
    - NDMI (Normalized Difference Moisture Index)
  - **Statistics:** Mean, StdDev, Min, Max per index
  - **Cumulative:** Running sum of mean indices
  - Batch processing with ThreadPoolExecutor
- **Output:** Satellite indices DataFrame saved to S3

---

### 3. Feature Materialization DAG

**DAG ID:** `feature_materialization`  
**Schedule:** Manual trigger  
**Tags:** `features`, `materialization`

#### Task Flow

```
load_weather_path ──┐
load_planting_dates_path ──┐
load_crop_stages_path ──┐
load_satellite_indices_path ──┐
                              │
                              ▼
                    build_daily_features
                              │
                              ▼
                    aggregate_plot_features
                              │
resolve_raw_labels_source ──┐
                              │
                              ▼
                    generate_labels_from_raw
```

#### Task Details

##### Task 1-4: Load Data Sources
- **Weather Path:** Validated weather data (falls back to raw)
- **Planting Dates Path:** Inferred planting dates
- **Crop Stages Path:** Full crop stage data
- **Satellite Path:** All vegetation indices
- **Fallback Logic:** Finds latest available if execution date not found

##### Task 5: Build Daily Features
- **Process:**
  1. **Merge Planting Metadata:** Join planting dates to weather
  2. **Compute GDD:**
     - Daily GDD: `((Tmax + Tmin)/2) - 10°C` (clipped at 0)
     - Cumulative GDD: Sum from planting_date onwards
  3. **Join Satellite Data:** Left join on `plot_id` + `date`
  4. **Label Crop Stages:** Determine current stage per day
     - Stages: VE, V2, V6, VT, R1, R4, R6
     - Labels: `pre_planting`, `post_planting_pre_VE`, stage name
  5. **Compute Days Since Planting:** `date - planting_date`
- **Output Columns:**
  - Core: `plot_id`, `date`, `latitude`, `longitude`
  - Weather: `max_temp_c`, `min_temp_c`, `precip_mm`
  - Derived: `daily_gdd`, `gdd_cumulative`, `days_since_planting`, `current_stage`
  - Satellite: `mean_{index}`, `cumulative_{index}` for each index
- **Output:** Daily features parquet saved to S3

##### Task 6: Aggregate Plot Features
- **Process:**
  - Groups daily features by `plot_id`
  - **Aggregations:**
    - `precip_mm` → `precip_total` (sum)
    - `daily_gdd` → `gdd_sum` (sum)
    - `gdd_cumulative` → `gdd_peak` (max)
    - Satellite indices → mean values
  - **Derived Features:**
    - `days_to_vt`: `VT_date - planting_date`
  - Joins crop stage metadata (season, altitude, confidence)
- **Output:** Plot-level features parquet saved to S3

##### Task 7-8: Labels Processing
- **Two Paths:**
  1. **Local Labels Export:** Finds local parquet/CSV files
  2. **Raw Labels Generation:** Processes raw harvest data
     - Sources: S3, Google Sheets URL, or local file
     - Computes yield labels:
       - `dry_harvest_kg/ha` = `((dry_box1 + dry_box2) / total_area_m2) * 10000`
       - `wet_harvest_kg/ha` = `((wet_box1 + wet_box2) / total_area_m2) * 10000`
     - Box area: 40 m² × 2 boxes = 80 m² total
- **Output:** Labels parquet saved to S3

---

### 4. Model Training DAG

**DAG ID:** `model_training_dag_weekly`  
**Schedule:** `0 3 * * 1` (Weekly on Mondays at 3:00 AM)  
**Tags:** `model`, `training`, `gridsearch`

#### Task Flow

```
load_labels_path ──┐
load_plot_features_path ──┐
                          │
                          ▼
            combine_features_and_labels
                          │
                          ▼
                    train_models
                          │
                          ▼
                    log_completion
```

#### Task Details

##### Task 1-2: Load Data
- **Labels Path:** Yield labels from S3
- **Plot Features Path:** Aggregated plot features from S3
- **Fallback:** Latest available if execution date not found

##### Task 3: Combine Features and Labels
- **Merge Strategy:**
  1. **Primary:** `plot_id` + `planting_date`
  2. **Fallback 1:** `plot_id` only
  3. **Fallback 2:** Coordinate-based (rounded lat/lon ± planting_date)
- **Process:**
  - Normalizes column names (plot_no → plot_id)
  - Handles timezone normalization
  - Detects missing features and attempts fallback merges
  - Validates merge quality
- **Output:** Combined dataset ready for training

##### Task 4: Train Models
- **Preprocessing:**
  - Converts `Planting_Date` to `plant_month`, `plant_doy`
  - Label encodes `crop_type` → `crop_type_enc`
  - Converts all to numeric (coerce errors)
- **Train/Test Split:** 80/20 with random_state=42
- **Target:** `dry_harvest_kg/ha`

##### Model 1: XGBoost
- **Base Parameters:**
  - `n_estimators`: 600
  - `learning_rate`: 0.05
  - `max_depth`: 8
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
- **Grid Search:**
  - `n_estimators`: [600, 1200]
  - `learning_rate`: [0.03, 0.05]
  - `max_depth`: [6, 8, 10]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.8, 1.0]
- **CV:** 3-fold
- **Scoring:** Negative RMSE

##### Model 2: LightGBM
- **Base Parameters:**
  - `n_estimators`: 600
  - `learning_rate`: 0.05
  - `subsample`: 0.8
  - `colsample_bytree`: 0.8
- **Grid Search:**
  - `n_estimators`: [600, 1200]
  - `learning_rate`: [0.03, 0.05]
  - `num_leaves`: [31, 63, 127]
  - `max_depth`: [-1, 8, 12]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.8, 1.0]
- **CV:** 3-fold

##### Model 3: CatBoost
- **Base Parameters:**
  - `loss_function`: RMSE
  - `eval_metric`: RMSE
  - `verbose`: False
- **Grid Search:**
  - `depth`: [6, 8, 10]
  - `learning_rate`: [0.03, 0.05]
  - `iterations`: [1000, 2000]
- **Categorical Features:** `crop_type_enc` (if present)
- **CV:** 3-fold

##### Model 4: Random Forest
- **Base Parameters:**
  - `n_estimators`: 100
  - `max_depth`: 10
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
- **Grid Search:**
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [8, 10, 12, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - `max_features`: [`sqrt`, `log2`, 0.5]
- **CV:** 3-fold
- **Scoring:** Negative RMSE

##### Model 5: Ensemble (VotingRegressor)
- **Type:** VotingRegressor (equal weights)
- **Base Models:** XGBoost, LightGBM, CatBoost, RandomForest
- **Logged Metrics:** RMSE, MAE, R2
- **Artifacts:** Ensemble model logged to MLflow and saved to S3/MinIO

##### MLflow Logging
- **Experiment:** `YieldPrediction_MultiModel`
- **Logged:**
  - Hyperparameters (best from GridSearch) for all four base models
  - Metrics: RMSE, MAE, R2 for each model and the ensemble
  - Model artifacts (saved to S3/MinIO) for each model and the ensemble
- **Output:** Training metrics dictionary + S3 paths for metrics, predictions, comparison reports

---

### 5. Drift Detection DAG

**DAG ID:** `drift_detection_dag`  
**Schedule:** `0 4 * * *` (Daily at 4:00 AM)  
**Tags:** `monitoring`, `drift`

#### Task Flow

```
detect_drift
    ↓
finalize
```

#### Task Details

##### Task 1: Detect Drift
- **Method:** Population Stability Index (PSI)
- **Process:**
  1. **Load Current Features:** Latest daily features
  2. **Load Reference:** Last 30 days of historical features
  3. **Calculate PSI:**
     - **Numeric Features:** Binned PSI (10 bins)
     - **Categorical Features:** Category-based PSI
  4. **Classify Drift Levels:**
     - **High:** PSI ≥ 0.25
     - **Medium:** PSI ≥ 0.1
     - **Low:** PSI < 0.1
  5. **Track Missing Values:** Compare missing rates
- **Output:**
  - Drift report DataFrame (top 20 features)
  - Summary metrics (n_high, n_medium, n_features)
  - Saved to S3 and MLflow

##### MLflow Logging
- **Experiment:** `DriftMonitoring`
- **Logged:**
  - Parameters: execution date, reference days, feature count
  - Metrics: n_high, n_medium drift features
  - Artifacts: CSV report, JSON summary

##### Alert System
- **High Drift Alert:** Triggered if `n_high > 0`
- **Status:** `ok` or `alert: high_drift`

---

## Data Storage Structure

### S3/MinIO Bucket Organization

```
nuru-yield/
├── raw/
│   ├── weather/
│   │   └── {YYYY-MM-DD}/
│   │       ├── weather_data.parquet
│   │       └── metadata.json
│   └── labels/
│       └── {YYYY-MM-DD}/
│           └── harvest.csv
│
├── validated/
│   └── weather/
│       └── {YYYY-MM-DD}/
│           └── weather_data.parquet
│
├── processed/
│   ├── weather/
│   │   └── {YYYY-MM-DD}/
│   │       ├── planting_dates.parquet
│   │       └── planting_metadata.json
│   └── satellite/
│       └── {YYYY-MM-DD}/
│           ├── all_indices.parquet
│           └── metadata.json
│
├── features/
│   └── {YYYY-MM-DD}/
│       ├── daily_features.parquet
│       ├── daily_features_metadata.json
│       ├── plot_features.parquet
│       └── plot_features_metadata.json
│
├── labels/
│   └── {YYYY-MM-DD}/
│       ├── yield.parquet
│       └── metadata.json
│
├── crop_stages/
│   └── crop_stages_{YYYY-MM-DD}.parquet
│
├── vt_stages/
│   └── vt_stages_{YYYY-MM-DD}.parquet
│
├── models/
│   └── {YYYY-MM-DD}/
│       ├── training_metrics.json
│       ├── predictions_comparison.parquet
│       ├── model_comparison.csv
│       └── training_summary.json
│
├── metadata/
│   └── plots/
│       └── plots_{YYYY-MM-DD}.parquet
│
└── monitoring/
    └── drift/
        └── {YYYY-MM-DD}/
            ├── report.parquet
            └── summary.json
```

### Metadata Structure

Each data file includes JSON metadata with:
- Execution date
- Row/plot counts
- Date ranges
- Column lists
- Data quality metrics (null counts, duplicates)

---

## Key Features & Capabilities

### 1. Robust Data Validation

- **Pydantic Schemas:** Type-safe data validation throughout
  - `PlotInput` - Plot metadata
  - `WeatherDataPoint` - Weather observations
  - `InferredPlantingDate` - Planting date results
  - `CropStageOutput` - Crop stage data
  - `VTStageOutput` - VT stage dates

- **Quality Checks:**
  - Temperature range validation (-50°C to 60°C)
  - Null checks on critical columns
  - Duplicate detection
  - Coordinate validation

### 2. Agronomic Intelligence

- **Planting Date Inference:**
  - Rainfall threshold: ≥ 5mm
  - Temperature threshold: ≥ 18°C average
  - Altitude adjustment: +1 day per 100m above 1200m
  - Confidence scoring

- **Crop Stage Determination:**
  - GDD-based calculation (base temp: 10°C)
  - Stage-specific thresholds
  - VT stage identification (critical for satellite timing)

### 3. Feature Engineering

- **Daily Features:**
  - Growing Degree Days (daily & cumulative)
  - Days since planting
  - Current crop stage label
  - Satellite index means and cumulative values

- **Plot-Level Aggregations:**
  - Total precipitation
  - GDD sum and peak
  - Mean satellite indices
  - Days to VT stage

### 4. Model Training

- **Ensemble Approach:**
  - XGBoost, LightGBM, CatBoost, RandomForest + VotingRegressor ensemble
  - GridSearchCV for hyperparameter tuning (3-fold)
  - Ensemble metrics logged; artifacts saved to MLflow and S3/MinIO

- **MLflow Integration:**
  - Experiment tracking
  - Model versioning
  - Artifact storage (S3/MinIO) for all base models and ensemble

### 5. Monitoring

- **Drift Detection:**
  - PSI calculation (numeric & categorical)
  - Daily monitoring
  - Alert system for high drift

- **Feature Monitoring:**
  - Missing value tracking
  - Distribution changes
  - Historical comparison

---

## Deployment Architecture

### Docker Compose Services

```yaml
Services:
  1. postgres (PostgreSQL 14)
     - Port: 5432
     - Databases: airflow, mlflow
     - Volumes: postgres_data

  2. minio (MinIO Server)
     - Ports: 9004 (API), 9001 (WebUI)
     - Credentials: minioadmin/minioadmin
     - Volumes: minio_data

  3. minio-init (Bucket Initialization)
     - Creates buckets on startup
     - Sets permissions

  4. airflow-init (Airflow Database Setup)
     - Initializes Airflow database
     - Creates admin user

  5. airflow-webserver
     - Port: 8083
     - Web UI for DAG management

  6. airflow-scheduler
     - Executes scheduled tasks
     - Manages task dependencies

  7. mlflow-db (PostgreSQL for MLflow)
     - Database: mlflow
     - Volumes: mlflow_pg_data

  8. mlflow-server
     - Port: 5000
     - Backend: PostgreSQL
     - Artifacts: S3/MinIO
```

### Volume Mounts

- **Airflow Home:** `/opt/airflow` (named volume)
- **DAGs:** `./airflow/dags` → `/opt/airflow/dags`
- **Logs:** `./airflow_logs` → `/opt/airflow/logs`
- **Source Code:** `./src` → `/opt/airflow/dags/repo/src`
- **Config:** `./config` → `/opt/airflow/dags/repo/config`
- **Data:** `./data` → `/opt/airflow/dags/repo/data`

### Environment Variables

Key environment variables (from `.env` file):

```bash
# API Keys
VISUAL_CROSSING_API_KEY=<your_key>

# Storage
USE_LOCAL_STORAGE=true
S3_BUCKET_NAME=nuru-yield
S3_ENDPOINT_URL=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# MLflow
MLFLOW_DB_PASSWORD=<password>
MLFLOW_TRACKING_URI=http://mlflow-server:5000

# Google Earth Engine
GEE_SERVICE_ACCOUNT_JSON=/opt/airflow/dags/repo/src/serene-bastion-406504-9c938287dece.json

# Airflow
AIRFLOW__CORE__EXECUTOR=LocalExecutor
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://airflow:airflow@postgres/airflow
```

---

## Configuration & Environment

### Settings File (`config/settings.py`)

The system uses Pydantic Settings for configuration:

- **API Configuration:**
  - Visual Crossing API key and base URL
  - Request timeout (30s)
  - Max workers (4)

- **Storage Configuration:**
  - Local vs AWS S3 mode
  - Bucket name and prefixes
  - Credentials (endpoint URL for MinIO)

- **Agronomic Settings:**
  - Base altitude: 1200m
  - Altitude delay factor: 100m per day
  - Min planting rainfall: 5mm
  - Min planting temperature: 18°C

- **Labels Configuration:**
  - Harvest sheet CSV URL (optional)
  - Raw labels file name: `harvest.csv`

### Schema Definitions (`config/schemas.py`)

Pydantic models for data validation:

- `PlotInput` - Input plot metadata
- `WeatherDataPoint` - Weather observation
- `InferredPlantingDate` - Planting date result
- `CropStageInput` - Crop stage input
- `CropStageOutput` - Crop stage result
- `VTStageOutput` - VT stage dates

---

## Data Flow Diagrams

### Complete Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA                               │
│  - Plot CSV (selected_farm_data.csv)                       │
│  - Raw Harvest Data (harvest.csv)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              WEATHER INGESTION DAG                          │
│  1. Load Plots                                              │
│  2. Fetch Weather (Visual Crossing API)                    │
│  3. Validate Weather                                        │
│  4. Infer Planting Dates                                    │
│  5. Determine Crop Stages (GDD)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              SATELLITE INGESTION DAG                         │
│  1. Load Crop Stages                                        │
│  2. Extract VT Dates                                        │
│  3. Prepare Input                                           │
│  4. Process via Google Earth Engine                         │
│     - Sentinel-2 Collection                                │
│     - Vegetation Indices (NDVI, EVI, etc.)                 │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│           FEATURE MATERIALIZATION DAG                        │
│  1. Load All Data Sources                                   │
│  2. Build Daily Features                                    │
│     - Join weather + satellite                              │
│     - Compute GDD                                           │
│     - Label stages                                          │
│  3. Aggregate Plot Features                                │
│  4. Process Labels (yield calculation)                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              MODEL TRAINING DAG                              │
│  1. Load Features & Labels                                   │
│  2. Combine Dataset                                         │
│  3. Train Models (XGBoost, LightGBM, CatBoost)            │
│     - GridSearchCV                                          │
│     - MLflow Logging                                         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            DRIFT DETECTION DAG                                │
│  1. Load Current Features                                   │
│  2. Compare with Historical (PSI)                           │
│  3. Generate Alerts                                         │
└─────────────────────────────────────────────────────────────┘
```

### Feature Engineering Flow

```
Weather Data          Satellite Data         Crop Stages
    │                      │                      │
    └──────────┬───────────┴──────────┬──────────┘
               │                      │
               ▼                      ▼
        Daily Features        Plot Aggregations
               │                      │
               └──────────┬───────────┘
                          │
                          ▼
                   Training Dataset
```

---

## API Integrations

### 1. Visual Crossing Weather API

- **Endpoint:** `https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline`
- **Method:** GET
- **Parameters:**
  - `lat,lon` - Coordinates
  - `start_date/end_date` - Date range
  - `unitGroup` - metric
  - `key` - API key
- **Response Fields:**
  - `tempmax`, `tempmin`, `temp` - Temperatures
  - `precip` - Precipitation (mm)
  - `humidity`, `solarenergy`, `windspeed`, `cloudcover`
- **Features:**
  - Retry logic (5 attempts)
  - Exponential backoff
  - Rate limiting handling (429)
  - Batch processing (ThreadPoolExecutor)

### 2. Google Earth Engine

- **Collection:** `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2)
- **Authentication:** Service account JSON
- **Operations:**
  - Image collection filtering (bounds, date, cloud cover)
  - Vegetation index calculation (server-side)
  - Region statistics (mean, std, min, max)
- **Features:**
  - Server-side processing
  - Retry logic for EE exceptions
  - Batch processing (ThreadPoolExecutor)

---

## Model Training Details

### Preprocessing Steps

1. **Date Features:**
   - `Planting_Date` → `plant_month`, `plant_doy`
   - Drop original date column

2. **Categorical Encoding:**
   - `crop_type` → `crop_type_enc` (LabelEncoder)

3. **Type Conversion:**
   - All columns to numeric (coerce errors)
   - Drop unnamed index columns

### Feature Set

**Plot-Level Features:**
- `precip_total` - Total precipitation
- `gdd_sum` - Sum of daily GDD
- `gdd_peak` - Peak cumulative GDD
- `mean_ndvi`, `mean_evi`, `mean_ndre`, `mean_savi`, `mean_ndwi`, `mean_ndmi`
- `days_to_vt` - Days from planting to VT stage
- `season` - Short Rains / Long Rains
- `altitude` - Plot altitude
- `plant_month`, `plant_doy` - Planting date features
- `crop_type_enc` - Encoded crop type

**Target:**
- `dry_harvest_kg/ha` - Dry yield in kg per hectare

### Model Evaluation

**Metrics:**
- **RMSE** - Root Mean Squared Error
- **MAE** - Mean Absolute Error
- **R2** - Coefficient of Determination

**Validation:**
- Train/Test Split: 80/20
- Cross-Validation: 3-fold (for GridSearchCV)

### Hyperparameter Tuning

**GridSearchCV Parameters:**

| Model | Parameters | Values |
|-------|-----------|--------|
| XGBoost | n_estimators | [600, 1200] |
| | learning_rate | [0.03, 0.05] |
| | max_depth | [6, 8, 10] |
| | subsample | [0.8, 1.0] |
| | colsample_bytree | [0.8, 1.0] |
| LightGBM | n_estimators | [600, 1200] |
| | learning_rate | [0.03, 0.05] |
| | num_leaves | [31, 63, 127] |
| | max_depth | [-1, 8, 12] |
| | subsample | [0.8, 1.0] |
| | colsample_bytree | [0.8, 1.0] |
| CatBoost | depth | [6, 8, 10] |
| | learning_rate | [0.03, 0.05] |
| | iterations | [1000, 2000] |

---

## Monitoring & Drift Detection

### Population Stability Index (PSI)

**Formula:**
```
PSI = Σ((Actual% - Expected%) × ln(Actual% / Expected%))
```

**Calculation:**
- **Numeric Features:** Binned into 10 quantile-based bins
- **Categorical Features:** Category-based proportions

**Thresholds:**
- **High Drift:** PSI ≥ 0.25
- **Medium Drift:** PSI ≥ 0.1
- **Low Drift:** PSI < 0.1

### Monitoring Process

1. **Load Current Features:** Latest daily features
2. **Load Reference:** Last 30 days of historical features
3. **Calculate PSI:** For each feature (numeric and categorical)
4. **Track Missing Values:** Compare missing rates
5. **Generate Report:** Top 20 features by PSI
6. **Log to MLflow:** Metrics and artifacts
7. **Save to S3:** Parquet report and JSON summary

### Alert System

- **High Drift Alert:** Triggered when `n_high > 0`
- **Status Values:**
  - `ok` - No high drift detected
  - `alert: high_drift` - High drift detected
  - `no_data` - No current data available
  - `no_reference` - No historical data available

---

## File Structure

```
nuru-yield-prediction/
├── airflow/
│   ├── dags/
│   │   ├── weather_ingestion_dag.py
│   │   ├── satellite_ingestion_dag.py
│   │   ├── feature_materialization_dag.py
│   │   ├── model_training_dag.py
│   │   └── drift_detection_dag.py
│   └── plugins/
├── config/
│   ├── settings.py
│   └── schemas.py
├── data/
│   ├── raw/
│   │   └── harvest.csv
│   ├── processed/
│   │   └── selected_farm_data.csv
│   └── features/
├── src/
│   ├── ingestion/
│   │   ├── weather_client.py
│   │   ├── satellite_client.py
│   │   ├── planting_date_inference.py
│   │   ├── crop_stage_determiner.py
│   │   └── vegetation_indices.py
│   ├── storage.py
│   ├── labels.py
│   ├── initialize.py
│   └── feast_repo/
│       ├── entities.py
│       ├── feature_views.py
│       └── sources.py
├── docker-compose.yml
├── Dockerfile
├── mlflow.Dockerfile
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

1. **Docker & Docker Compose** installed
2. **Google Earth Engine** service account JSON file
3. **Visual Crossing API** key
4. **Input Data:**
   - Plot CSV: `data/processed/selected_farm_data.csv`
   - Harvest data: `data/raw/harvest.csv` (optional)

### Setup Steps

1. **Clone Repository:**
   ```bash
   git clone <repository-url>
   cd nuru-yield-prediction
   ```

2. **Create `.env` File:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Place Service Account JSON:**
   ```bash
   # Copy GEE service account JSON to:
   src/serene-bastion-406504-9c938287dece.json
   ```

4. **Prepare Input Data:**
   - Place plot CSV at: `data/processed/selected_farm_data.csv`
   - Place harvest CSV at: `data/raw/harvest.csv` (optional)

5. **Start Services:**
   ```bash
   docker-compose up -d
   ```

6. **Access Services:**
   - Airflow UI: http://localhost:8083 (admin/admin)
   - MLflow UI: http://localhost:5000
   - MinIO UI: http://localhost:9001 (minioadmin/minioadmin)

7. **Trigger DAGs:**
   - Weather Ingestion: Runs daily at 2:00 AM (or trigger manually)
   - Satellite Ingestion: Trigger manually after crop stages available
   - Feature Materialization: Trigger manually after satellite data available
   - Model Training: Runs weekly on Mondays at 3:00 AM (or trigger manually)
   - Drift Detection: Runs daily at 4:00 AM

---

## Troubleshooting

### Common Issues

1. **Airflow DAGs Not Appearing:**
   - Check DAGs folder is mounted correctly
   - Verify Python imports are working
   - Check Airflow logs: `docker-compose logs airflow-scheduler`

2. **Weather API Failures:**
   - Verify API key in `.env`
   - Check rate limits (429 errors)
   - Review retry logic in logs

3. **Google Earth Engine Errors:**
   - Verify service account JSON path
   - Check GEE authentication: `earthengine authenticate`
   - Review quota limits

4. **S3/MinIO Connection Issues:**
   - Verify endpoint URL: `http://minio:9000`
   - Check credentials in `.env`
   - Ensure bucket exists (created by minio-init)

5. **Model Training Failures:**
   - Verify features and labels are available in S3
   - Check merge logic (plot_id matching)
   - Review MLflow connection

---

## Performance Considerations

### Optimization Strategies

1. **Batch Processing:**
   - Weather API: ThreadPoolExecutor (4 workers)
   - Satellite: ThreadPoolExecutor (4 workers)
   - Adjust `MAX_WORKERS` in settings

2. **Data Storage:**
   - Parquet format for efficient storage
   - Partitioned by date for easy access
   - Metadata files for quick inspection

3. **Model Training:**
   - GridSearchCV with `n_jobs=-1` for parallel execution
   - 3-fold CV (balance between speed and accuracy)

4. **Drift Detection:**
   - Limited to last 30 days for reference
   - Top 20 features only in reports

---

## Future Enhancements

### Potential Improvements

1. **Feature Store Integration:**
   - Full Feast integration for online serving
   - Real-time feature retrieval

2. **Model Serving:**
   - FastAPI inference service
   - Model versioning and A/B testing

3. **Advanced Monitoring:**
   - Data quality metrics
   - Model performance tracking
   - Automated retraining triggers

4. **Additional Data Sources:**
   - Soil data integration
   - Irrigation data
   - Pest/disease indicators

5. **Model Improvements:**
   - Deep learning models (LSTM, Transformer)
   - Time series forecasting
   - Multi-task learning

---

## Support & Contact

For issues, questions, or contributions:

- **Project Repository:** [GitHub URL]
- **Documentation:** This file
- **Logs:** Check Airflow UI or `./airflow_logs/`

---

## Appendix

### A. Vegetation Indices Formulas

- **NDVI:** `(NIR - Red) / (NIR + Red)`
- **EVI:** `2.5 × (NIR - Red) / (NIR + 6×Red - 7.5×Blue + 1)`
- **NDRE:** `(NIR - RedEdge) / (NIR + RedEdge)`
- **SAVI:** `(NIR - Red) / (NIR + Red + L) × (1 + L)` where L=0.5
- **NDWI:** `(Green - NIR) / (Green + NIR)`
- **NDMI:** `(NIR - SWIR) / (NIR + SWIR)`

### B. Crop Stage GDD Thresholds

Typical thresholds (may vary by crop):
- **VE:** ~50 GDD
- **V2:** ~150 GDD
- **V6:** ~400 GDD
- **VT:** ~800-1000 GDD
- **R1:** ~1200 GDD
- **R4:** ~1800 GDD
- **R6:** ~2500 GDD

### C. Data Schema Examples

**Plot Input:**
```json
{
  "plot_id": 1,
  "latitude": -0.499127,
  "longitude": 37.612253,
  "altitude": 1252.08,
  "season": "Short Rains",
  "year": 2021
}
```

**Weather Data Point:**
```json
{
  "plot_id": 1,
  "date": "2021-10-07",
  "max_temp_c": 28.5,
  "min_temp_c": 18.2,
  "mean_temp_c": 23.35,
  "precip_mm": 12.5
}
```

**Crop Stage Output:**
```json
{
  "plot_id": 1,
  "planting_date": "2021-10-07",
  "VE": "2021-10-16",
  "VT": "2021-12-15",
  "R1": "2021-12-25",
  "method": "gdd",
  "confidence": 0.85
}
```

---

**End of Documentation**

*Last Updated: December 2025*


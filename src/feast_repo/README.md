# Feast Feature Store (nuru_yield)

This repo defines a basic Feast setup mapping the materialized features to feature views, guided by `yield_features.md`:

- Entity: `plot` keyed by `plot_id`
- Feature Views:
  - `daily_features`: weather + satellite indices per day, with phenology labels and GDD metrics. Event timestamp: `date`.
  - `plot_features`: plot-level aggregates (precipitation totals, GDD sums, mean vegetation indices), phenology summaries (`planting_date`, `VT`, `days_to_vt`), and metadata (`season`, `altitude`, `confidence`). Event timestamp: `planting_date`.

Data sources point to S3/MinIO Parquet files written by `DataStorage.save_features_daily` and `DataStorage.save_features_plot` using s3fs.

## Usage

1. Ensure environment variables are set for S3/MinIO (see `config/settings.py`).
2. After running the Airflow DAGs to generate features, use Feast to build datasets:

```bash
feast apply -c src/feast_repo/feature_store.yaml
feast registry dump -c src/feast_repo/feature_store.yaml
```

You can create training datasets by joining the `plot_features` with your labels constructed per `yield_features.md` (`dry_harvest_kg/ha`, `wet_harvest_kg/ha`, `season`, etc.).


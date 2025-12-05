from feast import FileSource
from feast.data_format import ParquetFormat
from feast import Timestamp
from config.settings import get_settings


def _bucket_and_prefix():
    settings = get_settings()
    return settings.S3_BUCKET_NAME, settings.S3_BASE_PREFIX


# Daily features source (event timestamp: `date`)
def daily_features_source() -> FileSource:
    bucket, base_prefix = _bucket_and_prefix()
    path = f"s3://{bucket}/{base_prefix}/features/*/daily_features.parquet"
    return FileSource(
        name="daily_features_source",
        path=path,
        timestamp_field="date",
        file_format=ParquetFormat(),
    )


# Plot-level aggregated features source (event timestamp: `planting_date`)
def plot_features_source() -> FileSource:
    bucket, base_prefix = _bucket_and_prefix()
    path = f"s3://{bucket}/{base_prefix}/features/*/plot_features.parquet"
    return FileSource(
        name="plot_features_source",
        path=path,
        timestamp_field="planting_date",
        file_format=ParquetFormat(),
    )


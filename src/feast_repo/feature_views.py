from feast import FeatureView, Field
from feast.types import Int64, Float64, String

from .entities import plot
from .sources import daily_features_source, plot_features_source


# Daily features: weather + satellite indices + phenology labels
daily_features = FeatureView(
    name="daily_features",
    entities=[plot],
    ttl=None,
    schema=[
        Field(name="latitude", dtype=Float64),
        Field(name="longitude", dtype=Float64),
        Field(name="max_temp_c", dtype=Float64),
        Field(name="min_temp_c", dtype=Float64),
        Field(name="precip_mm", dtype=Float64),
        Field(name="daily_gdd", dtype=Float64),
        Field(name="gdd_cumulative", dtype=Float64),
        Field(name="days_since_planting", dtype=Int64),
        Field(name="current_stage", dtype=String),
        # Satellite indices (conditionally present; Feast tolerates missing columns)
        Field(name="mean_ndvi", dtype=Float64),
        Field(name="mean_evi", dtype=Float64),
        Field(name="mean_ndre", dtype=Float64),
        Field(name="mean_savi", dtype=Float64),
        Field(name="mean_ndwi", dtype=Float64),
        Field(name="mean_ndmi", dtype=Float64),
        Field(name="cumulative_ndvi", dtype=Float64),
        Field(name="cumulative_evi", dtype=Float64),
        Field(name="cumulative_ndre", dtype=Float64),
        Field(name="cumulative_savi", dtype=Float64),
        Field(name="cumulative_ndwi", dtype=Float64),
        Field(name="cumulative_ndmi", dtype=Float64),
    ],
    source=daily_features_source(),
)


# Plot-level aggregates: align with yield_features.md guidance
plot_features = FeatureView(
    name="plot_features",
    entities=[plot],
    ttl=None,
    schema=[
        Field(name="precip_total", dtype=Float64),
        Field(name="gdd_sum", dtype=Float64),
        Field(name="gdd_peak", dtype=Float64),
        Field(name="latitude", dtype=Float64),
        Field(name="longitude", dtype=Float64),
        Field(name="mean_ndvi", dtype=Float64),
        Field(name="mean_evi", dtype=Float64),
        Field(name="mean_ndre", dtype=Float64),
        Field(name="mean_savi", dtype=Float64),
        Field(name="mean_ndwi", dtype=Float64),
        Field(name="mean_ndmi", dtype=Float64),
        Field(name="planting_date", dtype=String),  # timestamp_field, exposed as feature for convenience
        Field(name="VT", dtype=String),
        Field(name="days_to_vt", dtype=Int64),
        Field(name="season", dtype=String),
        Field(name="altitude", dtype=Float64),
        Field(name="confidence", dtype=Float64),
    ],
    source=plot_features_source(),
)


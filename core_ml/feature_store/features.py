from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Array, Float32, Int32

# Offline Store Source (El batch de datos preprocesados periódicamente)
sensor_window_source = FileSource(
    name="engine_sensor_windows",
    path="s3://predictive-maintenance-mlops-artifacts-040175285118/feast/data/engine_features.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Entidad principal
engine = Entity(
    name="engine_id",
    join_keys=["engine_id"],
    value_type=ValueType.STRING,
)

# Feature View
engine_window_fv = FeatureView(
    name="engine_sensor_window_features",
    entities=[engine],
    ttl=timedelta(days=1),  # Retención en el Online Store
    schema=[
        Field(name="windowed_features", dtype=Array(Float32)),  # Array aplanado 30x24
        Field(name="failure_type", dtype=Int32),
    ],
    online=True,
    source=sensor_window_source,
    tags={"team": "mlops", "model": "predictive_maintenance"},
)

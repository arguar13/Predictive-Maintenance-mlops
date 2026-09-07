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
#
# online=False: sin streaming (core_ml/streaming/ fue eliminado - este
# proyecto no sirve inferencia en tiempo real), no hay nada que materialice
# ni consuma un Online Store. Mantenerlo en True sin nada que lo escriba ni
# lo lea es exactamente el tipo de capacidad fingida que este repo evita en
# otros lados (ver el docstring que tenia monitoring/evidently_service.py
# sobre concept drift, ya eliminado junto con todo lo demas de streaming).
# get_historical_features (train.py) es la unica ruta de lectura de este
# FeatureView, y usa el Offline Store exclusivamente.
engine_window_fv = FeatureView(
    name="engine_sensor_window_features",
    entities=[engine],
    ttl=timedelta(days=1),
    schema=[
        Field(name="windowed_features", dtype=Array(Float32)),  # Array aplanado 30x24
        Field(name="failure_type", dtype=Int32),
    ],
    online=False,
    source=sensor_window_source,
    tags={"team": "mlops", "model": "predictive_maintenance"},
)

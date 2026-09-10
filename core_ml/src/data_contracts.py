"""Contratos de datos (fail fast) para el pipeline de mantenimiento predictivo.

Cada función valida un DataFrame en un punto concreto del pipeline
(data_processing -> prepare_training_data -> train) y lanza
`pandera.errors.SchemaError` inmediatamente si el contrato se rompe, antes
de invertir cómputo (entrenamiento, llamadas a MLflow) en datos inválidos.
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaError, SchemaErrors

__all__ = [
    "SchemaError",
    "SchemaErrors",
    "validate_raw_telemetry",
    "validate_labeled_telemetry",
    "validate_windowed_features",
    "validate_training_batch",
]

_SENSOR_COLUMN_PATTERN = r"^sensor_\d+$"
_OP_SETTING_COLUMN_PATTERN = r"^op_setting_[1-3]$"

# ---------------------------------------------------------------------------
# 1. Telemetría cruda combinada (salida de load_and_combine_data)
# ---------------------------------------------------------------------------
RAW_TELEMETRY_SCHEMA = pa.DataFrameSchema(
    columns={
        "unit_number": pa.Column(int, checks=pa.Check.gt(0), nullable=False),
        "time_in_cycles": pa.Column(int, checks=pa.Check.gt(0), nullable=False),
        _OP_SETTING_COLUMN_PATTERN: pa.Column(
            float, checks=pa.Check(lambda s: s.abs() < 1e6), nullable=False, regex=True
        ),
        # Algunos sensores del dataset C-MAPSS (p.ej. sensor_17/sensor_18) solo
        # toman valores enteros en los ficheros originales, por lo que pandas
        # los infiere como int64; se admite float o int y se convierte (coerce).
        _SENSOR_COLUMN_PATTERN: pa.Column(
            float, checks=pa.Check(lambda s: s.abs() < 1e6), nullable=False, regex=True
        ),
        "dataset_id": pa.Column(str, nullable=False),
        "global_unit": pa.Column(str, nullable=False),
    },
    strict=False,
    coerce=True,
)

# ---------------------------------------------------------------------------
# 2. Telemetría etiquetada (salida de build_multiclass_target)
# ---------------------------------------------------------------------------
LABELED_TELEMETRY_SCHEMA = RAW_TELEMETRY_SCHEMA.add_columns(
    {
        "RUL": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
        "failure_type": pa.Column(int, checks=pa.Check.isin([0, 1, 2]), nullable=False),
    }
)

# ---------------------------------------------------------------------------
# 3. Ventanas listas para entrenamiento (salida de prepare_training_data.py)
# ---------------------------------------------------------------------------
WINDOWED_FEATURE_SCHEMA = pa.DataFrameSchema(
    columns={
        "engine_id": pa.Column(str, checks=pa.Check.str_length(min_value=1), nullable=False),
        "event_timestamp": pa.Column("datetime64[ns]", nullable=False),
        "created_timestamp": pa.Column("datetime64[ns]", nullable=False),
        "windowed_features": pa.Column(object, nullable=False),
        "failure_type": pa.Column(int, checks=pa.Check.isin([0, 1, 2]), nullable=False),
    },
    strict=False,
    coerce=False,
)

# ---------------------------------------------------------------------------
# 4. Batch ya cargado, listo para entrenar (salida de _load_training_data en
#    train.py) -- deliberadamente MAS LAXO que WINDOWED_FEATURE_SCHEMA en el
#    tipo exacto de "event_timestamp": no es una feature del modelo, asi que
#    alcanza con exigir "datetime de cualquier variante" en vez de fijar un
#    dtype exacto.
_TRAINING_BATCH_SCHEMA = pa.DataFrameSchema(
    columns={
        "engine_id": pa.Column(str, checks=pa.Check.str_length(min_value=1), nullable=False),
        # dtype=None (sin anotar) a proposito: pandera compara el dtype
        # anotado por IGUALDAD EXACTA. El Check de abajo verifica que sea
        # datetime de cualquier variante, sin pinnear una sola.
        "event_timestamp": pa.Column(
            dtype=None,
            checks=pa.Check(
                lambda s: pd.api.types.is_datetime64_any_dtype(s),
                element_wise=False,
                error="event_timestamp debe ser datetime64 (naive o tz-aware)",
            ),
            nullable=False,
        ),
        "windowed_features": pa.Column(object, nullable=False),
        "failure_type": pa.Column(int, checks=pa.Check.isin([0, 1, 2]), nullable=False),
    },
    strict=False,
    coerce=False,
)


def validate_raw_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Falla rápido si la telemetría cruda combinada no cumple el contrato."""
    return RAW_TELEMETRY_SCHEMA.validate(df, lazy=True)


def validate_labeled_telemetry(df: pd.DataFrame) -> pd.DataFrame:
    """Falla rápido si la telemetría etiquetada (RUL/failure_type) es inválida."""
    return LABELED_TELEMETRY_SCHEMA.validate(df, lazy=True)


def validate_windowed_features(df: pd.DataFrame, expected_length: int) -> pd.DataFrame:
    """Falla rápido si las ventanas aplanadas no tienen el contrato esperado.

    `expected_length` es `window_size * num_features` (ver config.yaml);
    se valida por separado porque depende de configuración, no es estático.
    """
    validated = WINDOWED_FEATURE_SCHEMA.validate(df, lazy=True)

    bad_length = validated["windowed_features"].map(len) != expected_length
    if bad_length.any():
        raise SchemaError(
            schema=WINDOWED_FEATURE_SCHEMA,
            data=validated,
            message=(
                f"{int(bad_length.sum())} fila(s) tienen 'windowed_features' con "
                f"longitud distinta de la esperada ({expected_length} = "
                "window_size * num_features)."
            ),
        )
    return validated


def validate_training_batch(df: pd.DataFrame, expected_length: int) -> pd.DataFrame:
    """Falla rápido si el batch ya cargado (engine_features.parquet) no es entrenable.

    Usar en train.py, DESPUÉS de `_load_training_data`. Para el DataFrame
    recién escrito por prepare_training_data.py, usar
    `validate_windowed_features`.
    """
    validated = _TRAINING_BATCH_SCHEMA.validate(df, lazy=True)

    bad_length = validated["windowed_features"].map(len) != expected_length
    if bad_length.any():
        raise SchemaError(
            schema=_TRAINING_BATCH_SCHEMA,
            data=validated,
            message=(
                f"{int(bad_length.sum())} fila(s) tienen 'windowed_features' con "
                f"longitud distinta de la esperada ({expected_length} = "
                "window_size * num_features)."
            ),
        )
    return validated

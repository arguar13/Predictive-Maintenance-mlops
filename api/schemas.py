"""Contratos de datos (Pydantic) para la API de inferencia.

FAIL FAST: una petición que no cumpla el contrato (matriz vacía o no
rectangular, valores no finitos, engine_id vacío) es rechazada por FastAPI
con un 422 antes de tocar el scaler o el modelo.

La forma EXACTA (window_size x num_features) no se valida aquí sino en el
endpoint, contra el modelo que está cargado en ese momento: el modelo
"champion" puede cambiar (o cargarse después de arrancar) sin reiniciar la
API, y un esquema construido una sola vez al importar el módulo quedaría
fijado a una forma obsoleta.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, field_validator, model_validator


class SensorWindowRequest(BaseModel):
    engine_id: str = Field(..., min_length=1, max_length=64)
    # Una fila por timestep; cada fila con las features en el orden de
    # `feature_names` (ver GET /health).
    readings: list[list[float]] = Field(..., min_length=1)

    @field_validator("engine_id")
    @classmethod
    def engine_id_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("engine_id no puede estar vacío o ser solo espacios.")
        return value

    @model_validator(mode="after")
    def validate_matrix(self) -> SensorWindowRequest:
        n_columns = len(self.readings[0])
        if n_columns == 0:
            raise ValueError("Las filas de 'readings' no pueden estar vacías.")
        for row_index, row in enumerate(self.readings):
            if len(row) != n_columns:
                raise ValueError(
                    f"Fila {row_index}: tiene {len(row)} valores y la fila 0 tiene "
                    f"{n_columns}; 'readings' debe ser una matriz rectangular."
                )
            if not all(math.isfinite(value) for value in row):
                raise ValueError(
                    f"Fila {row_index}: valores de sensor no finitos (NaN/Inf) no están permitidos."
                )
        return self


class PredictionResponse(BaseModel):
    engine_id: str
    prediction: str
    # Probabilidad por clase: permite al consumidor aplicar su propio umbral
    # (p.ej. alertar si P(Critical) > 0.3) en vez de depender solo del argmax.
    probabilities: dict[str, float]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    window_size: int
    num_features: int
    # Orden exacto de columnas que espera cada fila de /predict (sale del
    # scaler registrado junto al modelo). None si el scaler no lo guardó.
    feature_names: list[str] | None

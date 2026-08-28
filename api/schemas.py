"""Contratos de datos (Pydantic) para la API de inferencia.

FAIL FAST: una petición que no cumpla el contrato (forma incorrecta,
valores no finitos, engine_id vacío) es rechazada por FastAPI con un 422
antes de tocar el scaler o el modelo, evitando gastar cómputo de inferencia
en inputs inválidos.
"""

from __future__ import annotations

import math
from functools import lru_cache

from pydantic import BaseModel, Field, field_validator, model_validator


def build_sensor_window_model(window_size: int, num_features: int) -> type[BaseModel]:
    """Crea un modelo Pydantic parametrizado por la forma esperada de la ventana.

    La forma (window_size x num_features) depende del modelo actualmente
    servido (ver config.yaml -> model), por lo que no puede ser una
    constante fija en el módulo: se construye una vez al arrancar la app.
    """

    class SensorWindowRequest(BaseModel):
        engine_id: str = Field(..., min_length=1, max_length=64)
        readings: list[list[float]] = Field(..., min_length=1)

        @field_validator("engine_id")
        @classmethod
        def engine_id_must_not_be_blank(cls, value: str) -> str:
            if not value.strip():
                raise ValueError("engine_id no puede estar vacío o ser solo espacios.")
            return value

        @model_validator(mode="after")
        def validate_window_shape_and_values(self) -> SensorWindowRequest:
            n_rows = len(self.readings)
            if n_rows != window_size:
                raise ValueError(
                    f"Se esperaban {window_size} timesteps (window_size), se recibieron {n_rows}."
                )
            for row_index, row in enumerate(self.readings):
                if len(row) != num_features:
                    raise ValueError(
                        f"Fila {row_index}: se esperaban {num_features} sensores "
                        f"(num_features), se recibieron {len(row)}."
                    )
                for value in row:
                    if not math.isfinite(value):
                        raise ValueError(
                            f"Fila {row_index}: valores de sensor no finitos (NaN/Inf) "
                            "no están permitidos."
                        )
            return self

    SensorWindowRequest.__name__ = "SensorWindowRequest"
    return SensorWindowRequest


@lru_cache(maxsize=8)
def get_sensor_window_model(window_size: int, num_features: int) -> type[BaseModel]:
    """Variante cacheada de `build_sensor_window_model` para reuso en la app."""
    return build_sensor_window_model(window_size, num_features)

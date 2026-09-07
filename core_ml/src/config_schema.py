"""Contrato de datos (Pydantic) para config.yaml.

FAIL FAST: si el YAML de configuración falta una clave, tiene un tipo
incorrecto o un valor fuera de rango, la aplicación debe fallar al arrancar
con un error claro, no minutos/horas después con un KeyError críptico en
medio de un batch de entrenamiento o de una petición de inferencia.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)


class ModelConfig(BaseModel):
    window_size: int = Field(gt=0)
    num_features: int = Field(gt=0)
    mlflow_tracking_uri: str = Field(min_length=1)
    model_name: str = Field(min_length=1)


class MonitoringConfig(BaseModel):
    # Quality Gate: un modelo recién entrenado solo se promueve en el Model
    # Registry de MLflow (alias "champion") si supera AMBOS umbrales -
    # ver train.py::_evaluate_quality_gate para el porqué (accuracy plano
    # es ciego al costo asimétrico de un falso negativo en "Critical").
    f2_weighted_threshold: float = Field(ge=0, le=1)
    critical_recall_threshold: float = Field(ge=0, le=1)


class AppConfig(BaseModel):
    project: ProjectConfig
    model: ModelConfig
    monitoring: MonitoringConfig

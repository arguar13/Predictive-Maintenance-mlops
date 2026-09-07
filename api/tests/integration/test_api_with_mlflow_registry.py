"""Integración End-to-End: api.main sirve de verdad desde un MLflow Model
Registry real respaldado por Postgres (Testcontainers) — sin sqlite, sin
mocks del cliente de MLflow.

Fase 3: valida que api/ (serving) y el contrato que produce
core_ml/src/train.py (modelo + scaler co-ubicados en el mismo run, alias
"champion") "se entienden" localmente antes de tocar RDS/EKS reales.
Requiere Docker.

Nota: mlflow/torch/sklearn/testcontainers se importan de forma perezosa
(dentro de fixtures, no a nivel de módulo) para que la recolección de
pytest siga siendo rápida en `make test` cuando estos tests se deseleccionan
por marker.
"""

import sys

import pytest

pytestmark = pytest.mark.integration

# Deben coincidir con config/config.yaml (model.*) para que api.main use
# el mismo contrato de forma/nombre que este test registra en MLflow.
MODEL_NAME = "Turbofan_FCN"
WINDOW_SIZE = 30
NUM_FEATURES = 14
TEST_API_KEY = "integration-test-api-key"


@pytest.fixture
def postgres_container():
    # Scope de funcion (no de modulo): cada test necesita un registro de
    # MLflow AISLADO -- en particular, el test de "sin champion" no debe ver
    # el modelo que registra el otro test en el mismo contenedor/DB.
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as container:
        yield container


def _register_champion_model(tracking_uri: str, artifact_dir) -> None:
    """Registra un modelo+scaler reales y los promueve a 'champion',
    replicando exactamente lo que hace core_ml/src/train.py, sin depender
    del paquete core_ml (api/ y core_ml/ son proyectos Poetry separados)."""
    import joblib
    import mlflow
    import mlflow.pytorch
    import numpy as np
    import torch
    import torch.nn as nn
    from mlflow import MlflowClient
    from sklearn.preprocessing import StandardScaler

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment("api_integration_test", artifact_location=f"file:///{artifact_dir}")
    mlflow.set_experiment("api_integration_test")

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(NUM_FEATURES, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x.mean(dim=1))

    model = TinyModel()
    scaler = StandardScaler().fit(np.random.randn(50, NUM_FEATURES))

    with mlflow.start_run():
        example_input = torch.randn(1, WINDOW_SIZE, NUM_FEATURES)
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=MODEL_NAME,
            input_example=example_input.numpy(),
            serialization_format="pickle",
        )
        scaler_path = artifact_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

    client = MlflowClient()
    client.set_registered_model_alias(
        MODEL_NAME, "champion", str(model_info.registered_model_version)
    )


def _import_fresh_main():
    """api.main ejecuta la carga del modelo como efecto de import; se limpia
    la cache de sys.modules para forzar una recarga contra el MLFLOW_TRACKING_URI
    actual en cada test (en vez de reusar el módulo ya importado)."""
    for module_name in ("main", "config_loader", "schemas"):
        sys.modules.pop(module_name, None)
    import main

    return main


def test_api_serves_predictions_from_a_real_mlflow_registry(
    postgres_container, tmp_path_factory, monkeypatch
):
    from fastapi.testclient import TestClient

    tracking_uri = postgres_container.get_connection_url()
    artifact_dir = tmp_path_factory.mktemp("mlflow_artifacts")
    _register_champion_model(tracking_uri, artifact_dir)

    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    main = _import_fresh_main()

    client = TestClient(main.app)

    health = client.get("/health")
    assert health.status_code == 200
    assert str(health.json()["model_version"]) == "1"

    readings = [[0.0] * NUM_FEATURES for _ in range(WINDOW_SIZE)]
    payload = {"engine_id": "ENG_001", "readings": readings}

    # Sin X-API-Key: /predict debe rechazar, no servir. Es el comportamiento
    # que este cambio existe para garantizar - probarlo aqui, contra la app
    # real (no un mock de require_api_key), es lo unico que confirma que la
    # dependencia realmente esta enganchada al endpoint y no solo definida.
    unauthenticated = client.post("/predict", json=payload)
    assert unauthenticated.status_code == 401

    prediction = client.post("/predict", json=payload, headers={"X-API-Key": TEST_API_KEY})
    assert prediction.status_code == 200
    body = prediction.json()
    assert body["engine_id"] == "ENG_001"
    assert body["prediction"] in {"Healthy", "Alert", "Critical"}
    assert str(body["model_version"]) == "1"


def test_api_reports_unavailable_when_no_champion_is_registered(postgres_container, monkeypatch):
    from fastapi.testclient import TestClient

    # Un tracking store Postgres real, pero sin ningún modelo registrado
    # todavía (p.ej. un despliegue nuevo antes del primer entrenamiento que
    # supera el quality gate): la API debe arrancar sin caerse, y /health
    # debe reflejar honestamente que no hay modelo disponible.
    tracking_uri = postgres_container.get_connection_url()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("API_KEY", TEST_API_KEY)
    main = _import_fresh_main()

    client = TestClient(main.app)
    health = client.get("/health")
    assert health.status_code == 503

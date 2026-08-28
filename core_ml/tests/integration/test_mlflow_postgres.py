"""Integración End-to-End: train_pipeline() contra un MLflow real respaldado
por un Postgres efímero (Testcontainers), no un sqlite de usar-y-tirar.

Fase 3: "garantizar que los componentes se entienden" (core_ml + MLflow +
Postgres) localmente, antes de tocar AWS/RDS reales. Requiere Docker.

Nota: mlflow/torch/testcontainers se importan de forma perezosa (dentro de
los fixtures, no a nivel de módulo) para que la recolección de pytest siga
siendo rápida en `make test` cuando estos tests se deseleccionan por marker.
"""

import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def postgres_container():
    from testcontainers.community.postgres import PostgresContainer

    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="module")
def monkeypatch_module():
    # pytest's built-in `monkeypatch` fixture es function-scoped; el tracking
    # URI debe persistir para todos los tests del módulo (mismo contenedor).
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def mlflow_tracking_uri(postgres_container, tmp_path_factory, monkeypatch_module):
    import mlflow

    tracking_uri = postgres_container.get_connection_url()
    monkeypatch_module.setenv("MLFLOW_TRACKING_URI", tracking_uri)

    mlflow.set_tracking_uri(tracking_uri)
    artifact_dir = tmp_path_factory.mktemp("mlflow_artifacts")
    mlflow.create_experiment(
        "Predictive_Maintenance_FCN", artifact_location=f"file:///{artifact_dir}"
    )

    return tracking_uri


def test_train_pipeline_persists_lineage_and_promotes_champion_in_real_postgres(
    mlflow_tracking_uri,
):
    import joblib
    import mlflow
    from mlflow import MlflowClient

    from train import train_pipeline

    run_id = train_pipeline(
        data_dir="data_toy",
        data_source="parquet",
        epochs=1,
        enforce_quality_gate=False,
    )

    client = MlflowClient()
    run = client.get_run(run_id)

    # La tupla de trazabilidad quedó persistida de verdad en Postgres, no en memoria.
    assert run.data.tags["dvc_data_dir"] == "data_toy"
    assert run.data.tags["data_source"] == "parquet"
    assert run.data.tags["git_commit_sha"]
    assert "val_accuracy" in run.data.metrics
    assert run.data.params["window_size"] == "30"

    # El modelo quedó en el Model Registry (no en un directorio suelto).
    model_name = "Turbofan_FCN"
    versions = client.search_model_versions(f"name='{model_name}'")
    assert any(v.run_id == run_id for v in versions)

    # El scaler es un artefacto recuperable del MISMO run.
    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="preprocessing/scaler.joblib"
    )
    scaler = joblib.load(scaler_path)
    assert hasattr(scaler, "transform")


def test_quality_gate_promotion_matches_the_persisted_gate_tag(mlflow_tracking_uri):
    from mlflow import MlflowClient

    from train import train_pipeline

    run_id = train_pipeline(
        data_dir="data_toy",
        data_source="parquet",
        epochs=1,
        enforce_quality_gate=False,
    )

    client = MlflowClient()
    run = client.get_run(run_id)
    model_name = "Turbofan_FCN"
    version = next(
        v for v in client.search_model_versions(f"name='{model_name}'") if v.run_id == run_id
    )

    quality_gate_passed = run.data.tags["quality_gate_passed"] == "True"
    champion = client.get_model_version_by_alias(model_name, "champion")
    if quality_gate_passed:
        assert champion.version == version.version
    else:
        assert champion.version != version.version

import pytest

from config_loader import load_config


def _valid_config_text(
    broker: str = "kafka:9092", mlflow_uri: str = "http://localhost:5000"
) -> str:
    return f"""
project:
  name: "predictive_maintenance_turbofan"
  version: "2.0.0"
kafka:
  broker: "{broker}"
  telemetry_topic: "engine_telemetry"
  alert_topic: "engine_alerts"
model:
  window_size: 30
  num_features: 14
  mlflow_tracking_uri: "{mlflow_uri}"
  model_name: "Turbofan_FCN"
monitoring:
  prometheus_port: 8000
  drift_threshold: 0.05
  accuracy_threshold: 0.85
"""


def test_load_config_resolves_env_var(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(broker="${TEST_BROKER}"))
    monkeypatch.setenv("TEST_BROKER", "kafka:9092")

    config = load_config(config_path=str(config_file))

    assert config["kafka"]["broker"] == "kafka:9092"


def test_load_config_uses_local_default_for_known_env_var(tmp_path, monkeypatch):
    """Una variable CONOCIDA sin definir cae a su default de desarrollo."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(broker="${KAFKA_BROKER}"))
    monkeypatch.delenv("KAFKA_BROKER", raising=False)

    config = load_config(config_path=str(config_file))

    assert config["kafka"]["broker"] == "localhost:9092"


def test_load_config_default_mlflow_uri_has_a_scheme(tmp_path, monkeypatch):
    """Regresion: el default de MLFLOW_TRACKING_URI debe ser una URI valida.

    El fallback anterior era `f"localhost:{'9092' if 'KAFKA' in env_var else '5000'}"`,
    que producia "localhost:5000" -- SIN esquema. MLflow lo rechaza con
    UnsupportedModelRegistryStoreURIException, de modo que `make smoke-test`
    y `make train-toy` fallaban siempre salvo que la variable estuviese
    exportada a mano, pese a documentarse como comandos autonomos.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(mlflow_uri="${MLFLOW_TRACKING_URI}"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    config = load_config(config_path=str(config_file))

    assert config["model"]["mlflow_tracking_uri"] == "http://localhost:5000"
    assert "://" in config["model"]["mlflow_tracking_uri"]


def test_load_config_fails_fast_on_unknown_env_var(tmp_path, monkeypatch):
    """FAIL FAST: una variable DESCONOCIDA no se inventa, se reporta.

    Antes, cualquier placeholder mal escrito se resolvia silenciosamente a
    "localhost:5000" y el error aparecia mucho despues, sin relacion
    aparente con su causa.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(broker="${MISSING_KAFKA_VAR}"))
    monkeypatch.delenv("MISSING_KAFKA_VAR", raising=False)

    with pytest.raises(ValueError, match="MISSING_KAFKA_VAR"):
        load_config(config_path=str(config_file))


def test_load_config_fails_fast_on_missing_section(tmp_path):
    """FAIL FAST: falta la sección 'monitoring' completa."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
project:
  name: "predictive_maintenance_turbofan"
  version: "2.0.0"
kafka:
  broker: "localhost:9092"
  telemetry_topic: "engine_telemetry"
  alert_topic: "engine_alerts"
model:
  window_size: 30
  num_features: 14
  mlflow_tracking_uri: "{mlflow_uri}"
  model_name: "Turbofan_FCN"
"""
    )

    with pytest.raises(ValueError, match="config.yaml inválido"):
        load_config(config_path=str(config_file))

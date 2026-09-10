import pytest
from pydantic import ValidationError

from config_loader import load_config


def _valid_config_text(mlflow_uri: str = "http://localhost:5000") -> str:
    return f"""
project:
  name: "predictive_maintenance_turbofan"
  version: "2.0.0"
model:
  window_size: 30
  num_features: 14
  mlflow_tracking_uri: "{mlflow_uri}"
  model_name: "Turbofan_FCN"
monitoring:
  f2_weighted_threshold: 0.75
  critical_recall_threshold: 0.75
"""


def test_load_config_resolves_env_var(tmp_path, monkeypatch):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(mlflow_uri="${TEST_MLFLOW_URI}"))
    monkeypatch.setenv("TEST_MLFLOW_URI", "http://mlflow-test:5000")

    config = load_config(config_path=str(config_file))

    assert config["model"]["mlflow_tracking_uri"] == "http://mlflow-test:5000"


def test_load_config_default_mlflow_uri_has_a_scheme(tmp_path, monkeypatch):
    """El default de MLFLOW_TRACKING_URI debe ser una URI valida, CON esquema.

    MLflow rechaza un host:puerto sin esquema con
    UnsupportedModelRegistryStoreURIException, de modo que `make smoke-test`
    y `make train-toy` fallarian siempre salvo que la variable estuviese
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

    Un placeholder mal escrito en config.yaml debe fallar de forma
    explícita en el momento, no resolverse silenciosamente a un valor por
    defecto arbitrario que oculte la causa real del error.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text(mlflow_uri="${MISSING_MLFLOW_VAR}"))
    monkeypatch.delenv("MISSING_MLFLOW_VAR", raising=False)

    with pytest.raises(ValueError, match="MISSING_MLFLOW_VAR"):
        load_config(config_path=str(config_file))


def test_load_config_leaves_plain_values_untouched(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(_valid_config_text())

    config = load_config(config_path=str(config_file))

    assert config["project"]["name"] == "predictive_maintenance_turbofan"
    assert config["model"]["window_size"] == 30


def test_load_config_fails_fast_on_missing_section(tmp_path):
    """FAIL FAST: falta la sección 'model' completa."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        """
project:
  name: "predictive_maintenance_turbofan"
  version: "2.0.0"
monitoring:
  f2_weighted_threshold: 0.75
  critical_recall_threshold: 0.75
"""
    )

    with pytest.raises(ValueError, match="config.yaml inválido"):
        load_config(config_path=str(config_file))


def test_load_config_fails_fast_on_out_of_range_threshold(tmp_path):
    """FAIL FAST: f2_weighted_threshold fuera de [0, 1] no es un umbral válido."""
    config_file = tmp_path / "config.yaml"
    bad_value = "f2_weighted_threshold: 1.5"
    invalid_text = _valid_config_text().replace("f2_weighted_threshold: 0.75", bad_value)
    config_file.write_text(invalid_text)

    with pytest.raises(ValueError, match="config.yaml inválido"):
        load_config(config_path=str(config_file))


def test_load_config_fails_fast_on_wrong_type(tmp_path):
    """FAIL FAST: window_size debe ser un entero, no una cadena."""
    config_file = tmp_path / "config.yaml"
    invalid_text = _valid_config_text().replace("window_size: 30", 'window_size: "treinta"')
    config_file.write_text(invalid_text)

    with pytest.raises((ValueError, ValidationError)):
        load_config(config_path=str(config_file))

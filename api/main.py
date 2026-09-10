import hmac
import os

import joblib
import mlflow
import mlflow.artifacts
import mlflow.pytorch
import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from mlflow import MlflowClient

from config_loader import load_config
from logging_config import configure_logging
from schemas import get_sensor_window_model

log = configure_logging("api")

# Alias del Model Registry que sirve esta API: solo se mueve a este alias un
# modelo que superó el Quality Gate de train.py (ver core_ml/src/train.py).
# Nunca se sirve "latest" a ciegas ni se lee un .joblib suelto del disco.
CHAMPION_ALIAS = "champion"

app = FastAPI(title="Predictive Maintenance API", version="2.0")
config = load_config()
mlflow.set_tracking_uri(config["model"]["mlflow_tracking_uri"])

# La API se servia sin ninguna autenticacion detras de un Service type:
# LoadBalancer expuesto a internet - cualquiera con la URL podia consultar
# /predict. API_KEY llega via envFrom -> secretRef: mlops-secrets
# (kubernetes/base/secret.yaml, un Secret plano de Kubernetes), nunca
# hardcodeada ni en config.yaml (que no es secreto y se commitea). FAIL
# FAST: arrancar sin ella y caer de vuelta a "sin auth" reintroduciria en
# silencio exactamente el hueco que este cambio cierra.
_API_KEY_HEADER = "X-API-Key"
_api_key_header = APIKeyHeader(name=_API_KEY_HEADER, auto_error=False)
_raw_api_key = os.environ.get("API_KEY")
if not _raw_api_key:
    raise RuntimeError(
        "La variable de entorno API_KEY no esta definida. La API se niega a "
        "arrancar sin autenticacion configurada (ver kubernetes/base/"
        "secret.yaml, clave API_KEY)."
    )
_API_KEY: str = _raw_api_key


def _is_valid_api_key(provided: str | None, expected: str) -> bool:
    """Comparacion en tiempo constante: evita una fuga de timing que dejaria
    adivinar la API key caracter a caracter contra un `==` normal."""
    if provided is None:
        return False
    return hmac.compare_digest(provided, expected)


def require_api_key(provided: str | None = Depends(_api_key_header)) -> None:
    if not _is_valid_api_key(provided, _API_KEY):
        raise HTTPException(status_code=401, detail="API key inválida o ausente.")


def _infer_window_shape(model_uri: str, scaler, fallback: tuple[int, int]) -> tuple[int, int]:
    """Deduce (window_size, num_features) del modelo servido, no de config.yaml.

    POR QUE: `num_features` NO es una constante del proyecto, es una
    propiedad del dataset. `create_sliding_windows` (core_ml/src/
    data_processing.py) descarta los sensores invariantes calculandolos sobre
    los datos concretos, de modo que el dataset toy produce 18 features y el
    completo puede producir otro numero. config.yaml declaraba 14 a mano.

    Con el valor de config, `api/schemas.py` validaba peticiones de 30x14 y
    acto seguido `scaler.transform()` -- ajustado sobre 18 columnas --
    reventaba con ValueError. Es decir: /predict no podia devolver una
    prediccion correcta NUNCA, y el desajuste solo se manifestaba en runtime,
    peticion a peticion.

    La fuente de verdad es el artefacto: la signature que train.py registra
    en MLflow (`mlflow.models.infer_signature`) y, como respaldo,
    `scaler.n_features_in_`. Asi el contrato de la API sigue automaticamente
    a cualquier modelo que se promueva a "champion", sin editar config.yaml.
    """
    try:
        from mlflow.models import get_model_info

        shape = get_model_info(model_uri).signature.inputs.inputs[0].shape
        # (batch, window_size, num_features) con batch = -1
        if len(shape) == 3 and shape[1] > 0 and shape[2] > 0:
            return int(shape[1]), int(shape[2])
        log.warning("model_signature_unexpected_shape", shape=list(shape))
    except Exception:
        log.warning("model_signature_unavailable", exc_info=True)

    num_features = getattr(scaler, "n_features_in_", None)
    if num_features:
        return fallback[0], int(num_features)
    return fallback


def _load_champion_model_and_scaler(model_name: str):
    """Carga el modelo y su scaler desde el MLflow Model Registry.

    Ambos son referencias inmutables al mismo run de MLflow: nunca un
    model.pkl/scaler.joblib suelto en un directorio local. Si aún no existe
    ningún modelo con el alias "champion" (p.ej. despliegue nuevo antes del
    primer entrenamiento que supere el quality gate), la API arranca sin
    modelo y /health lo refleja como no disponible.
    """
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)

    model_uri = f"models:/{model_name}@{CHAMPION_ALIAS}"
    loaded_model = mlflow.pytorch.load_model(model_uri)
    loaded_model.eval()

    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id, artifact_path="preprocessing/scaler.joblib"
    )
    loaded_scaler = joblib.load(scaler_path)
    return loaded_model, loaded_scaler, model_version.version, model_uri


model = None
scaler = None
model_version = None

# Forma declarada en config.yaml: solo se usa mientras no haya un modelo
# "champion" cargado (la API arranca igual y /health devuelve 503).
window_size = config["model"]["window_size"]
num_features = config["model"]["num_features"]

try:
    model, scaler, model_version, _model_uri = _load_champion_model_and_scaler(
        config["model"]["model_name"]
    )
    window_size, num_features = _infer_window_shape(
        _model_uri, scaler, fallback=(window_size, num_features)
    )
    log.info(
        "model_loaded",
        model_name=config["model"]["model_name"],
        model_version=model_version,
        alias=CHAMPION_ALIAS,
        window_size=window_size,
        num_features=num_features,
        config_num_features=config["model"]["num_features"],
    )
except Exception:
    log.exception(
        "model_load_failed",
        model_name=config["model"]["model_name"],
        alias=CHAMPION_ALIAS,
    )

# El contrato de /predict se construye DESPUES de cargar el modelo, con la
# forma real que este espera.
SensorWindowRequest = get_sensor_window_model(
    window_size=window_size,
    num_features=num_features,
)


@app.get("/health")
# Deliberadamente SIN require_api_key: el readinessProbe/livenessProbe de
# kubernetes/base/api.yaml lo consultan via httpGet sin headers custom, y no
# expone nada mas sensible que "hay un modelo cargado y que forma espera" -
# el mismo criterio que separa endpoints de salud de endpoints de negocio en
# cualquier API publica.
def health_check():
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return {
        "status": "ok",
        "service": "online",
        "model_version": model_version,
        # Expuesto para que un cliente sepa que forma enviar sin tener que
        # provocar un 422 primero.
        "window_size": window_size,
        "num_features": num_features,
    }


@app.post("/predict", dependencies=[Depends(require_api_key)])
# SensorWindowRequest se construye en tiempo de ejecución con la forma
# (window_size x num_features) del modelo servido, por lo que mypy no puede
# verificar estáticamente sus atributos: es un factory de Pydantic, no un
# tipo estático (ver api/schemas.py::get_sensor_window_model).
def predict_manual(data: SensorWindowRequest):  # type: ignore[valid-type]
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")

    scaled_data = scaler.transform(data.readings)  # type: ignore[attr-defined]
    input_tensor = torch.tensor(np.array([scaled_data]), dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = int(torch.argmax(outputs, dim=1).item())

    classes = ["Healthy", "Alert", "Critical"]
    engine_id = data.engine_id  # type: ignore[attr-defined]
    result = classes[prediction]
    log.info(
        "prediction_served", engine_id=engine_id, prediction=result, model_version=model_version
    )
    return {
        "engine_id": engine_id,
        "prediction": result,
        "model_version": model_version,
    }

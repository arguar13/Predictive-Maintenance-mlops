import hmac
import math
import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Annotated, Any

import joblib
import mlflow
import mlflow.artifacts
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import APIKeyHeader
from mlflow import MlflowClient

from config_loader import load_config
from logging_config import configure_logging
from schemas import HealthResponse, PredictionResponse, SensorWindowRequest

log = configure_logging("api")

# Alias del Model Registry que sirve esta API: solo se mueve a este alias un
# modelo que superó el Quality Gate de train.py (ver core_ml/src/train.py).
# Nunca se sirve "latest" a ciegas ni se lee un .joblib suelto del disco.
CHAMPION_ALIAS = "champion"
# 0=Healthy, 1=Alert, 2=Critical (ver core_ml/src/data_processing.py).
CLASSES = ("Healthy", "Alert", "Critical")
# Tras un intento fallido de carga no se reintenta en cada request: cada
# intento pega contra MLflow, y /health lo consulta el readinessProbe.
RELOAD_COOLDOWN_SECONDS = float(os.environ.get("MODEL_RELOAD_COOLDOWN_SECONDS", "30"))

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


# ---------------------------------------------------------------------------
# Modelo servido
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedModel:
    """Modelo + scaler de un MISMO run de MLflow, y la forma que esperan."""

    model: Any
    scaler: Any
    version: str
    window_size: int
    num_features: int
    feature_names: list[str] | None

    def predict_proba(self, readings: np.ndarray) -> np.ndarray:
        """readings: (window_size, num_features) crudas -> probabilidad por clase."""
        # Con los nombres de columna con los que se ajusto el scaler (evita el
        # UserWarning de sklearn en cada request y documenta el orden).
        features = (
            pd.DataFrame(readings, columns=self.feature_names)
            if self.feature_names is not None
            else readings
        )
        scaled = np.asarray(self.scaler.transform(features), dtype=np.float32)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(scaled[np.newaxis]))
            return torch.softmax(logits, dim=1)[0].numpy()


def _infer_window_shape(model_uri: str, scaler: Any, default_window_size: int) -> tuple[int, int]:
    """Deduce (window_size, num_features) de los ARTEFACTOS servidos, no de config.yaml.

    `num_features` depende de los datos (los sensores invariantes se descartan
    sobre el dataset concreto), asi que su fuente de verdad es el scaler
    registrado en el mismo run que el modelo: es exactamente lo que
    `scaler.transform()` va a aceptar. La signature del modelo aporta el
    window_size y sirve de chequeo cruzado: si modelo y scaler no coinciden,
    el run esta roto y es mejor no servirlo que fallar en cada peticion.
    """
    num_features = int(scaler.n_features_in_)
    window_size = default_window_size
    try:
        from mlflow.models import get_model_info

        shape = get_model_info(model_uri).signature.inputs.inputs[0].shape
        # (batch, window_size, num_features) con batch = -1
        if len(shape) == 3 and shape[1] > 0 and shape[2] > 0:
            window_size = int(shape[1])
            if int(shape[2]) != num_features:
                raise ValueError(
                    f"La signature del modelo espera {shape[2]} features y el scaler "
                    f"del mismo run {num_features}: artefactos inconsistentes."
                )
        else:
            log.warning("model_signature_unexpected_shape", shape=list(shape))
    except ValueError:
        raise
    except Exception:
        log.warning("model_signature_unavailable", exc_info=True)
    return window_size, num_features


def load_champion(model_name: str, default_window_size: int) -> LoadedModel:
    """Carga el modelo "champion" y su scaler desde el MLflow Model Registry.

    Ambos son referencias inmutables al mismo run de MLflow: nunca un
    model.pkl/scaler.joblib suelto en un directorio local.
    """
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)

    # URI por VERSION concreta (no por alias): modelo y scaler salen del mismo
    # run aunque el alias se mueva entre esta llamada y la anterior.
    model_uri = f"models:/{model_name}/{model_version.version}"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()

    scaler_path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id, artifact_path="preprocessing/scaler.joblib"
    )
    scaler = joblib.load(scaler_path)
    window_size, num_features = _infer_window_shape(model_uri, scaler, default_window_size)
    feature_names = getattr(scaler, "feature_names_in_", None)
    return LoadedModel(
        model=model,
        scaler=scaler,
        version=str(model_version.version),
        window_size=window_size,
        num_features=num_features,
        feature_names=[str(name) for name in feature_names] if feature_names is not None else None,
    )


class ModelService:
    """Mantiene el modelo servido y reintenta cargarlo mientras no exista.

    Antes el modelo se cargaba UNA vez al importar el modulo: si MLflow aun no
    respondia o todavia no habia champion (despliegue nuevo), la API quedaba
    en 503 para siempre hasta reiniciar el pod. Ahora cada consulta a /health
    o /predict reintenta la carga (con un cooldown entre intentos), asi que el
    readinessProbe se pone en verde solo en cuanto aparece el champion.

    Promover un champion NUEVO sigue requiriendo reiniciar la API (rollout
    restart): un modelo ya cargado no se sustituye en caliente.
    """

    def __init__(self, loader: Callable[[], LoadedModel], cooldown_seconds: float) -> None:
        self._loader = loader
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._current: LoadedModel | None = None
        self._last_attempt = -math.inf

    def get(self) -> LoadedModel | None:
        if self._current is not None:
            return self._current
        # No bloqueante: si otro hilo ya esta cargando (puede tardar, MLflow
        # reintenta), esta peticion responde 503 en vez de quedarse colgada.
        if not self._lock.acquire(blocking=False):
            return None
        try:
            now = time.monotonic()
            if self._current is None and now - self._last_attempt >= self._cooldown_seconds:
                self._last_attempt = now
                try:
                    self._current = self._loader()
                    log.info(
                        "model_loaded",
                        model_version=self._current.version,
                        window_size=self._current.window_size,
                        num_features=self._current.num_features,
                    )
                except Exception:
                    log.exception("model_load_failed", alias=CHAMPION_ALIAS)
            return self._current
        finally:
            self._lock.release()


model_service = ModelService(
    loader=lambda: load_champion(config["model"]["model_name"], config["model"]["window_size"]),
    cooldown_seconds=RELOAD_COOLDOWN_SECONDS,
)


def get_model_service() -> ModelService:
    return model_service


ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]


def _require_model(service: ModelService) -> LoadedModel:
    loaded = service.get()
    if loaded is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return loaded


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Primer intento de carga en segundo plano: el servidor empieza a aceptar
    # conexiones (livenessProbe TCP en verde) aunque MLflow tarde en responder.
    threading.Thread(target=model_service.get, name="model-loader", daemon=True).start()
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    version=config["project"]["version"],
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
# Deliberadamente SIN require_api_key: el readinessProbe de
# kubernetes/base/api.yaml lo consulta via httpGet sin headers custom, y no
# expone nada mas sensible que "hay un modelo cargado y que forma espera".
def health_check(service: ModelServiceDep) -> HealthResponse:
    loaded = _require_model(service)
    return HealthResponse(
        status="ok",
        model_version=loaded.version,
        window_size=loaded.window_size,
        num_features=loaded.num_features,
        feature_names=loaded.feature_names,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    dependencies=[Depends(require_api_key)],
)
def predict(
    request: SensorWindowRequest,
    service: ModelServiceDep,
) -> PredictionResponse:
    loaded = _require_model(service)

    readings = np.asarray(request.readings, dtype=np.float64)
    expected_shape = (loaded.window_size, loaded.num_features)
    if readings.shape != expected_shape:
        raise HTTPException(
            status_code=422,
            detail=(
                f"'readings' debe tener forma {expected_shape} (window_size x num_features), "
                f"se recibio {readings.shape}. Ver GET /health para la forma y el orden de "
                "columnas esperados."
            ),
        )

    probabilities = loaded.predict_proba(readings)
    prediction = CLASSES[int(np.argmax(probabilities))]
    log.info(
        "prediction_served",
        engine_id=request.engine_id,
        prediction=prediction,
        model_version=loaded.version,
    )
    return PredictionResponse(
        engine_id=request.engine_id,
        prediction=prediction,
        probabilities={
            name: round(float(p), 6) for name, p in zip(CLASSES, probabilities, strict=True)
        },
        model_version=loaded.version,
    )

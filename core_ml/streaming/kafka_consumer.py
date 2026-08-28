"""Consumidor de streaming: infiere sobre telemetria de Kafka en tiempo real.

Fase 4 (Observabilidad y Resiliencia):
  * Logging JSON estructurado (logging_config.configure_logging) en vez de
    print() -- apto para CloudWatch Logs Insights / Elasticsearch.
  * Reintentos con backoff exponencial ACOTADO (tenacity) para fallos
    transitorios de Feast/MLflow -- nunca un reintento infinito.
  * Circuit breaker (pybreaker) alrededor del pipeline de inferencia: si una
    dependencia externa (Feast/MLflow) falla de forma sostenida, el breaker
    se abre y deja de golpearla durante `reset_timeout`, en vez de entrar en
    un loop de fallos que satura CPU/red y genera ruido en los logs.
  * Un mensaje individual que falla NUNCA tumba el proceso: se loguea y se
    continua con el siguiente (antes, cualquier excepcion no capturada
    mataba el consumidor entero -> crashloop en Kubernetes).
  * Solo sirve el modelo con el alias "champion" (el que superó el Quality
    Gate de train.py), igual que api/main.py -- antes usaba "latest" a
    ciegas.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import mlflow.pytorch
import numpy as np
import pybreaker
import torch
from kafka import KafkaConsumer, KafkaProducer
from kafka_security import kafka_client_kwargs
from mlflow import MlflowClient
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config_loader import load_config
from logging_config import configure_logging

if TYPE_CHECKING:
    # Import perezoso en runtime (solo para tipos): feast es una dependencia
    # pesada que _process_message/_fetch_online_features no necesitan de
    # verdad importar para ser testeadas unitariamente (mockeando el store).
    from feast import FeatureStore

log = configure_logging("streaming-consumer")

CHAMPION_ALIAS = "champion"
WINDOW_SIZE = 30
CLASSES = ["Healthy", "Alert", "Critical"]

# Tras 5 fallos consecutivos del pipeline de inferencia (Feast/MLflow/Kafka),
# el breaker se abre 60s: deja de intentar en cada mensaje y falla rapido,
# en vez de reintentar indefinidamente contra una dependencia caida.
inference_breaker = pybreaker.CircuitBreaker(fail_max=5, reset_timeout=60)

_TRANSIENT_ERRORS = (ConnectionError, TimeoutError, OSError)

# Reintentos con backoff exponencial ACOTADO (maximo 3 intentos, nunca
# indefinido) para fallos transitorios de red hacia Feast/Kafka.
_retry_transient = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    reraise=True,
)


def _load_champion_model(model_name: str):
    client = MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    model = mlflow.pytorch.load_model(f"models:/{model_name}@{CHAMPION_ALIAS}")
    model.eval()
    log.info(
        "model_loaded", model_name=model_name, alias=CHAMPION_ALIAS, version=model_version.version
    )
    return model, model_version.version


@_retry_transient
def _fetch_online_features(store: FeatureStore, engine_id: str) -> np.ndarray | None:
    feature_vector = store.get_online_features(
        features=["engine_sensor_window_features:windowed_features"],
        entity_rows=[{"engine_id": engine_id}],
    ).to_dict()

    raw = feature_vector["windowed_features"][0]
    if raw is None:
        return None
    return np.array(raw)


@_retry_transient
def _publish_alert(producer: KafkaProducer, alert_topic: str, result: dict) -> None:
    producer.send(alert_topic, result).get(timeout=10)


@inference_breaker
def _process_message(
    engine_id: str,
    store: FeatureStore,
    model,
    model_version: str,
    producer: KafkaProducer,
    alert_topic: str,
) -> None:
    raw_features = _fetch_online_features(store, engine_id)
    if raw_features is None:
        log.warning("no_online_features", engine_id=engine_id)
        return

    if raw_features.size % WINDOW_SIZE != 0:
        log.warning(
            "unexpected_feature_size",
            engine_id=engine_id,
            size=int(raw_features.size),
            window_size=WINDOW_SIZE,
        )
        return

    num_features = raw_features.size // WINDOW_SIZE
    window_matrix = raw_features.reshape(WINDOW_SIZE, num_features)
    input_tensor = torch.tensor(np.array([window_matrix]), dtype=torch.float32)

    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = int(torch.argmax(outputs, dim=1).item())

    result = {
        "engine_id": engine_id,
        "status": CLASSES[prediction],
        "model_version": model_version,
    }
    _publish_alert(producer, alert_topic, result)
    log.info("prediction_published", **result)


def run_streaming_inference() -> None:
    from feast import FeatureStore

    config = load_config()
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", config["model"]["mlflow_tracking_uri"])
    )

    model, model_version = _load_champion_model(config["model"]["model_name"])

    feature_store_path = os.path.join(os.path.dirname(__file__), "../feature_store")
    store = FeatureStore(repo_path=feature_store_path)

    kafka_broker = config["kafka"]["broker"]
    telemetry_topic = config["kafka"]["telemetry_topic"]
    alert_topic = config["kafka"]["alert_topic"]

    # security_protocol se inyecta por entorno: PLAINTEXT en el Kafka local de
    # docker-compose, SSL contra MSK (client_broker = TLS). Ver
    # streaming/kafka_security.py.
    transport = kafka_client_kwargs()

    consumer = KafkaConsumer(
        telemetry_topic,
        bootstrap_servers=[kafka_broker],
        auto_offset_reset="latest",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
        **transport,
    )
    producer = KafkaProducer(
        bootstrap_servers=[kafka_broker],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        **transport,
    )

    log.info(
        "consumer_started",
        kafka_broker=kafka_broker,
        telemetry_topic=telemetry_topic,
        alert_topic=alert_topic,
        security_protocol=transport["security_protocol"],
        model_version=model_version,
    )

    for message in consumer:
        engine_id = message.value.get("engine_id")
        if not engine_id:
            log.warning("message_missing_engine_id")
            continue

        try:
            _process_message(engine_id, store, model, model_version, producer, alert_topic)
        except pybreaker.CircuitBreakerError:
            # El breaker esta abierto: la dependencia externa viene fallando
            # de forma sostenida. No se reintenta mensaje a mensaje -- se
            # deja pasar el mensaje y se espera a que el breaker se cierre.
            log.error("circuit_breaker_open", engine_id=engine_id)
        except Exception:
            # Un mensaje individual jamas debe tumbar el proceso completo.
            log.exception("message_processing_failed", engine_id=engine_id)


if __name__ == "__main__":
    run_streaming_inference()

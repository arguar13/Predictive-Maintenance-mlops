"""Servicio de deteccion de data drift: consume telemetria de Kafka, compara
contra una distribucion de referencia con Evidently y expone metricas en
Prometheus (scrapeadas por kubernetes/base/prometheus.yml).

Reescrito desde cero (version anterior no funcional, ver historial de git):
  * API de Evidently: el proyecto fija `evidently>=0.4.30` pero instala la
    serie 0.7.x, que reemplazo por completo `evidently.report.Report` /
    `evidently.metric_preset` (0.4.x) por `evidently.Report` +
    `evidently.presets.DataDriftPreset`, con un `Snapshot.dict()` de forma
    distinta. El modulo anterior no importaba siquiera.
  * `reference_data = pd.read_csv("data/reference_data_clean.csv")` a nivel
    de modulo: ese archivo no existe en ningun lugar del repositorio -> el
    proceso moria en el import, antes de poder arrancar nada.
  * `run_monitoring_service()` creaba dos `KafkaConsumer` y terminaba sin
    iterarlos: no habia ningun bucle de consumo real. `detect_drift()` y
    `detect_concept_drift()` estaban definidas pero jamas se llamaban desde
    ningun lado.
  * `trigger_github_actions_retraining()` apuntaba a `TU_USUARIO` (placeholder
    sin rellenar) y a GitHub Actions -- vestigio de una arquitectura anterior
    a la migracion a GitLab CI (ver .gitlab-ci.yml, kubernetes/base/
    externalsecret.yaml -> GIT_REPO_TOKEN). Se reemplaza por un trigger real
    de pipeline de GitLab.
  * Monitoreo de concept drift (accuracy en produccion) requeriria un topic
    de "ground truth" con las etiquetas reales de cada motor; ese contrato
    no existe en ningun otro punto del proyecto (ni productor ni consumidor
    lo implementan). Implementarlo aqui de forma aislada seria inventar una
    funcionalidad sin dueno real en el resto del sistema, asi que se deja
    fuera de alcance explicitamente en vez de simularlo a medias.

Nota sobre los datos: `core_ml/streaming/producer_sim.py` (el simulador de
telemetria, ver su docstring) genera lecturas ALEATORIAS -- no hay un dataset
de referencia "real" con el que compararlas de forma significativa. La
distribucion de referencia se genera aqui con la misma forma y semilla fija,
para que el pipeline (consumo -> lote -> reporte -> metricas -> trigger) sea
funcional y verificable de punta a punta; el *contenido* del drift detectado
sobre telemetria sintetica no tiene relevancia de negocio real, igual que el
resto de la simulacion.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from evidently import Report
from evidently.presets import DataDriftPreset
from kafka import KafkaConsumer
from prometheus_client import Gauge, start_http_server

sys.path.append(str(Path(__file__).resolve().parent.parent / "core_ml" / "src"))
from config_loader import load_config  # noqa: E402
from logging_config import configure_logging  # noqa: E402

log = configure_logging("evidently-drift-monitor")

DATA_DRIFT_SHARE = Gauge(
    "evidently_data_drift_share",
    "Fraccion de features con drift detectado en el ultimo lote",
)
DATA_DRIFT_DETECTED = Gauge(
    "evidently_data_drift_detected",
    "1 si el ultimo lote supero el umbral de drift, 0 si no",
)
BATCHES_PROCESSED = Gauge(
    "evidently_batches_processed_total",
    "Lotes de telemetria evaluados desde el arranque",
)

# Tamano del lote antes de correr un reporte de drift. No forma parte de
# config/config.yaml (monitoring.drift_threshold/accuracy_threshold son
# umbrales de negocio, esto es un parametro operativo del propio servicio).
_BATCH_SIZE = int(os.environ.get("EVIDENTLY_BATCH_SIZE", "20"))
_REFERENCE_SEED = int(os.environ.get("EVIDENTLY_REFERENCE_SEED", "42"))

_GITLAB_HOST = os.environ.get("GITLAB_HOST", "gitlab.com")
_GITLAB_PROJECT_PATH = os.environ.get(
    "GITLAB_PROJECT_PATH", "personal-group7745334/proyecto611_verdadero"
)
_GITLAB_TRIGGER_REF = os.environ.get("GITLAB_TRIGGER_REF", "dev")


def _feature_columns(num_features: int) -> list[str]:
    return [f"feature_{i}" for i in range(num_features)]


def _build_reference_data(
    window_size: int, num_features: int, num_rows: int = 200
) -> pd.DataFrame:
    """Linea base fija (semilla constante) con la misma forma que la
    telemetria simulada: `producer_sim.py` no lee de ningun dataset real
    (ver docstring del modulo), asi que la referencia se genera igual.
    """
    rng = np.random.default_rng(_REFERENCE_SEED)
    windows = rng.random((num_rows, window_size, num_features))
    pooled = windows.mean(axis=1)  # una fila por ventana: promedio temporal
    return pd.DataFrame(pooled, columns=_feature_columns(num_features))


def _window_to_row(sensor_readings: list, num_features: int) -> np.ndarray | None:
    matrix = np.asarray(sensor_readings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != num_features:
        return None
    return matrix.mean(axis=0)


def _run_drift_report(
    reference_data: pd.DataFrame, current_batch: pd.DataFrame
) -> float:
    report = Report(metrics=[DataDriftPreset()])
    snapshot = report.run(current_data=current_batch, reference_data=reference_data)
    result = snapshot.dict()

    drift_count_metric = next(
        m
        for m in result["metrics"]
        if m["metric_name"].startswith("DriftedColumnsCount")
    )
    drift_share = float(drift_count_metric["value"]["share"])
    return drift_share


def _trigger_gitlab_retraining() -> None:
    # Reutiliza GIT_REPO_TOKEN (PAT ya provisionado por terraform/secrets.tf
    # -> AWS Secrets Manager -> ExternalSecret, ver kubernetes/base/
    # externalsecret.yaml) contra el endpoint estandar de la API en vez de
    # crear un Pipeline Trigger Token aparte: un secreto menos que rotar y
    # sincronizar, a costa de un scope algo mas amplio que el estrictamente
    # necesario (aceptable: el token ya existe para el job "release").
    token = os.environ.get("GIT_REPO_TOKEN")
    if not token:
        log.error("gitlab_token_missing", env_var="GIT_REPO_TOKEN")
        return

    url = f"https://{_GITLAB_HOST}/api/v4/projects/{quote(_GITLAB_PROJECT_PATH, safe='')}/pipeline"
    try:
        response = requests.post(
            url,
            headers={"PRIVATE-TOKEN": token},
            params={"ref": _GITLAB_TRIGGER_REF},
            timeout=10,
        )
        response.raise_for_status()
        log.info(
            "gitlab_pipeline_triggered",
            project_path=_GITLAB_PROJECT_PATH,
            ref=_GITLAB_TRIGGER_REF,
        )
    except requests.RequestException:
        log.exception("gitlab_trigger_failed", project_path=_GITLAB_PROJECT_PATH)


def run_monitoring_service() -> None:
    config = load_config()
    window_size = config["model"]["window_size"]
    num_features = config["model"]["num_features"]
    drift_threshold = config["monitoring"]["drift_threshold"]
    prometheus_port = config["monitoring"]["prometheus_port"]
    telemetry_topic = config["kafka"]["telemetry_topic"]
    kafka_broker = config["kafka"]["broker"]

    reference_data = _build_reference_data(window_size, num_features)

    start_http_server(prometheus_port)
    log.info(
        "monitor_started",
        prometheus_port=prometheus_port,
        kafka_broker=kafka_broker,
        telemetry_topic=telemetry_topic,
        batch_size=_BATCH_SIZE,
        drift_threshold=drift_threshold,
    )

    consumer = KafkaConsumer(
        telemetry_topic,
        # kafka_broker puede traer varios brokers separados por coma (MSK):
        # ver la nota equivalente en core_ml/streaming/kafka_consumer.py.
        bootstrap_servers=kafka_broker,
        auto_offset_reset="latest",
        value_deserializer=lambda x: json.loads(x.decode("utf-8")),
    )

    batch_rows: list[np.ndarray] = []
    for message in consumer:
        sensor_readings = message.value.get("sensor_readings")
        if sensor_readings is None:
            log.warning("message_missing_sensor_readings")
            continue

        row = _window_to_row(sensor_readings, num_features)
        if row is None:
            log.warning("unexpected_window_shape", expected_num_features=num_features)
            continue

        batch_rows.append(row)
        if len(batch_rows) < _BATCH_SIZE:
            continue

        current_batch = pd.DataFrame(batch_rows, columns=_feature_columns(num_features))
        batch_rows = []

        try:
            drift_share = _run_drift_report(reference_data, current_batch)
        except Exception:
            log.exception("drift_report_failed")
            continue

        drift_detected = drift_share >= drift_threshold
        DATA_DRIFT_SHARE.set(drift_share)
        DATA_DRIFT_DETECTED.set(1 if drift_detected else 0)
        BATCHES_PROCESSED.inc()

        log.info(
            "drift_batch_evaluated",
            drift_share=drift_share,
            drift_detected=drift_detected,
        )

        if drift_detected:
            log.warning(
                "data_drift_detected",
                drift_share=drift_share,
                threshold=drift_threshold,
            )
            _trigger_gitlab_retraining()


if __name__ == "__main__":
    run_monitoring_service()

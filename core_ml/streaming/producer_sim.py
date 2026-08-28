"""Simulador de telemetria de sensores hacia el topic `engine_telemetry`.

Correcciones respecto a la version anterior:
  * El broker ya NO esta hardcodeado a "localhost:9092": se lee de
    config/config.yaml (`kafka.broker`), que a su vez resuelve la variable
    de entorno KAFKA_BROKER inyectada por kubernetes/base/configmap.yaml.
    Tal como estaba, este simulador solo podia funcionar en un portatil:
    dentro del cluster nunca hubiera encontrado un broker.
  * Los topics tambien salen de config.yaml, en vez de una constante local
    que podia divergir de la que usa el consumidor.
  * `num_features` se toma de `model.num_features` (14), no de un 24 fijo
    que no correspondia a ninguna parte del pipeline.
  * Logging JSON estructurado (structlog) en vez de print(), igual que el
    resto de componentes -- consultable en CloudWatch Logs Insights.
  * Transporte TLS configurable por entorno (ver streaming/kafka_security.py),
    necesario contra Amazon MSK con client_broker = TLS.
"""

from __future__ import annotations

import json
import time

import numpy as np
from kafka import KafkaProducer
from kafka_security import kafka_client_kwargs

from config_loader import load_config
from logging_config import configure_logging

log = configure_logging("producer-sim")

SEND_INTERVAL_SECONDS = 5


def run_simulation() -> None:
    config = load_config()

    broker = config["kafka"]["broker"]
    topic = config["kafka"]["telemetry_topic"]
    window_size = config["model"]["window_size"]
    num_features = config["model"]["num_features"]

    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        **kafka_client_kwargs(),
    )

    log.info(
        "producer_started",
        kafka_broker=broker,
        telemetry_topic=topic,
        window_size=window_size,
        num_features=num_features,
    )

    engine_id_counter = 1
    try:
        while True:
            readings = np.random.rand(window_size, num_features).tolist()
            data = {
                "engine_id": f"ENG_{engine_id_counter:03d}",
                "sensor_readings": readings,
            }
            producer.send(topic, data).get(timeout=10)
            log.info("telemetry_sent", engine_id=data["engine_id"], topic=topic)
            engine_id_counter += 1
            time.sleep(SEND_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        log.info("simulation_stopped")
    finally:
        producer.close()


if __name__ == "__main__":
    run_simulation()

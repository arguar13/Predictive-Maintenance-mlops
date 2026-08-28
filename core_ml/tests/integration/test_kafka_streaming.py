"""Integración End-to-End: el contrato de mensajería Kafka usado por
streaming/producer_sim.py y streaming/kafka_consumer.py, contra un broker
Kafka real y efímero (Testcontainers), no mockeado.

Fase 3: valida que productor y consumidor "se entienden" localmente antes
de tocar Amazon MSK. Requiere Docker.

Nota: testcontainers/kafka se importan de forma perezosa (dentro de los
fixtures, no a nivel de módulo) para que la recolección de pytest siga
siendo rápida en `make test` cuando estos tests se deseleccionan por marker.
"""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.integration

TELEMETRY_TOPIC = "engine_telemetry"
ALERT_TOPIC = "engine_alerts"


@pytest.fixture(scope="module")
def kafka_container():
    from testcontainers.community.kafka import KafkaContainer

    with KafkaContainer() as container:
        yield container


def _make_producer(broker: str):
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )


def _make_consumer(broker: str, topic: str):
    from kafka import KafkaConsumer

    # auto_offset_reset="earliest" (en vez de "latest", como en producción)
    # para que el test sea determinista sin depender de ganar una carrera
    # contra la suscripción del consumidor.
    return KafkaConsumer(
        topic,
        bootstrap_servers=[broker],
        auto_offset_reset="earliest",
        consumer_timeout_ms=20000,
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")),
    )


def test_producer_sim_payload_is_understood_by_the_consumer_contract(kafka_container):
    """Reproduce el payload exacto de producer_sim.py y confirma que se
    deserializa tal como lo espera kafka_consumer.py (message.value['engine_id'])."""
    broker = kafka_container.get_bootstrap_server()

    producer = _make_producer(broker)
    window_size, num_features = 30, 24
    payload = {
        "engine_id": "ENG_001",
        "sensor_readings": np.random.rand(window_size, num_features).tolist(),
    }
    producer.send(TELEMETRY_TOPIC, payload).get(timeout=10)
    producer.flush()
    producer.close()

    consumer = _make_consumer(broker, TELEMETRY_TOPIC)
    try:
        received = next(iter(consumer)).value
    finally:
        consumer.close()

    assert received["engine_id"] == "ENG_001"
    assert len(received["sensor_readings"]) == window_size
    assert len(received["sensor_readings"][0]) == num_features


def test_alert_roundtrip_matches_kafka_consumer_output_shape(kafka_container):
    """Reproduce el payload de salida que kafka_consumer.py publica en
    engine_alerts tras una inferencia, y confirma que es consumible."""
    broker = kafka_container.get_bootstrap_server()

    producer = _make_producer(broker)
    alert = {"engine_id": "ENG_042", "status": "Critical"}
    producer.send(ALERT_TOPIC, alert).get(timeout=10)
    producer.flush()
    producer.close()

    consumer = _make_consumer(broker, ALERT_TOPIC)
    try:
        received = next(iter(consumer)).value
    finally:
        consumer.close()

    assert received == alert
    assert received["status"] in {"Healthy", "Alert", "Critical"}

"""Unit tests (sin Docker/Kafka/Feast reales) del control de reintentos y
del circuit breaker de streaming/kafka_consumer.py -- Fase 4."""

from unittest.mock import MagicMock

import pybreaker
import pytest
from kafka_consumer import CLASSES, _fetch_online_features, _process_message, inference_breaker


@pytest.fixture(autouse=True)
def reset_circuit_breaker():
    """El breaker es un singleton a nivel de modulo: se resetea entre tests
    para que el estado de uno no contamine al siguiente."""
    inference_breaker.close()
    yield
    inference_breaker.close()


def test_fetch_online_features_retries_transient_errors_then_succeeds():
    store = MagicMock()
    store.get_online_features.side_effect = [
        ConnectionError("feast unreachable"),
        ConnectionError("feast unreachable"),
        MagicMock(to_dict=lambda: {"windowed_features": [[0.1, 0.2, 0.3]]}),
    ]

    result = _fetch_online_features(store, "ENG_001")

    assert result.tolist() == [0.1, 0.2, 0.3]
    assert store.get_online_features.call_count == 3


def test_fetch_online_features_gives_up_after_max_attempts():
    store = MagicMock()
    store.get_online_features.side_effect = ConnectionError("feast unreachable")

    with pytest.raises(ConnectionError):
        _fetch_online_features(store, "ENG_001")

    # stop_after_attempt(3): nunca reintenta indefinidamente.
    assert store.get_online_features.call_count == 3


def test_fetch_online_features_does_not_retry_non_transient_errors():
    store = MagicMock()
    store.get_online_features.side_effect = ValueError("payload malformado")

    with pytest.raises(ValueError):
        _fetch_online_features(store, "ENG_001")

    assert store.get_online_features.call_count == 1


def test_process_message_returns_none_when_no_online_features():
    store = MagicMock()
    store.get_online_features.return_value.to_dict.return_value = {"windowed_features": [None]}
    producer = MagicMock()

    _process_message(
        "ENG_001",
        store,
        model=MagicMock(),
        model_version="1",
        producer=producer,
        alert_topic="engine_alerts",
    )

    producer.send.assert_not_called()


def test_process_message_publishes_prediction(monkeypatch):
    import numpy as np
    import torch

    store = MagicMock()
    flat = np.zeros(30 * 2).tolist()
    store.get_online_features.return_value.to_dict.return_value = {"windowed_features": [flat]}

    model = MagicMock(return_value=torch.tensor([[0.1, 0.2, 5.0]]))
    producer = MagicMock()
    producer.send.return_value.get.return_value = None

    _process_message(
        "ENG_001",
        store,
        model=model,
        model_version="3",
        producer=producer,
        alert_topic="engine_alerts",
    )

    producer.send.assert_called_once()
    topic, payload = producer.send.call_args[0]
    assert topic == "engine_alerts"
    assert payload["engine_id"] == "ENG_001"
    assert payload["status"] == CLASSES[2]  # logits favorecen "Critical"
    assert payload["model_version"] == "3"


def test_circuit_breaker_opens_after_repeated_failures():
    store = MagicMock()
    store.get_online_features.side_effect = ConnectionError("feast down")
    producer = MagicMock()

    def call_once():
        _process_message(
            "ENG_001",
            store,
            model=MagicMock(),
            model_version="1",
            producer=producer,
            alert_topic="t",
        )

    # fail_max=5: las primeras (fail_max - 1) llamadas fallan "normalmente"
    # (propagan el error real de la dependencia). La llamada que cruza el
    # umbral (la 5a) es la que pybreaker reporta como CircuitBreakerError
    # -- a partir de ahi, el breaker esta abierto.
    for _ in range(inference_breaker.fail_max - 1):
        with pytest.raises(ConnectionError):
            call_once()
    assert inference_breaker.current_state == "closed"

    with pytest.raises(pybreaker.CircuitBreakerError):
        call_once()
    assert inference_breaker.current_state == "open"

    # Con el breaker abierto, ni siquiera se llama a la dependencia real:
    # falla rapido en vez de seguir golpeando un servicio caido.
    calls_before = store.get_online_features.call_count
    with pytest.raises(pybreaker.CircuitBreakerError):
        call_once()
    assert store.get_online_features.call_count == calls_before

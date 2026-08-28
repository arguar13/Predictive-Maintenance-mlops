import pytest
from pydantic import ValidationError

from schemas import get_sensor_window_model


@pytest.fixture
def model():
    return get_sensor_window_model(window_size=3, num_features=2)


def test_valid_window_is_accepted(model):
    instance = model(engine_id="ENG_001", readings=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    assert instance.engine_id == "ENG_001"
    assert len(instance.readings) == 3


def test_rejects_wrong_window_size(model):
    with pytest.raises(ValidationError, match="window_size"):
        model(engine_id="ENG_001", readings=[[1.0, 2.0], [3.0, 4.0]])


def test_rejects_wrong_num_features(model):
    with pytest.raises(ValidationError, match="num_features"):
        model(engine_id="ENG_001", readings=[[1.0], [3.0, 4.0], [5.0, 6.0]])


def test_rejects_nan_values(model):
    with pytest.raises(ValidationError, match="finitos"):
        model(engine_id="ENG_001", readings=[[1.0, float("nan")], [3.0, 4.0], [5.0, 6.0]])


def test_rejects_infinite_values(model):
    with pytest.raises(ValidationError, match="finitos"):
        model(engine_id="ENG_001", readings=[[1.0, float("inf")], [3.0, 4.0], [5.0, 6.0]])


def test_rejects_blank_engine_id(model):
    with pytest.raises(ValidationError):
        model(engine_id="   ", readings=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])


def test_rejects_empty_readings(model):
    with pytest.raises(ValidationError):
        model(engine_id="ENG_001", readings=[])


def test_model_is_cached_per_shape():
    model_a = get_sensor_window_model(window_size=30, num_features=14)
    model_b = get_sensor_window_model(window_size=30, num_features=14)
    assert model_a is model_b

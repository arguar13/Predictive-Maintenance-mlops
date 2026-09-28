from types import SimpleNamespace

import numpy as np
from mlflow.exceptions import MlflowException

from train import (
    _checkpoint_score,
    _evaluate_quality_gate,
    _should_replace_champion,
    _split_by_engine,
    _training_data_fingerprint,
)


def _fake_windows(
    engine_window_counts: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X/y/groups sinteticos: `n` ventanas por motor. El contenido de cada
    ventana codifica de que motor vino (un valor constante por motor), asi
    que a partir de un X_train/X_val devuelto por _split_by_engine se puede
    recuperar que motores fueron a cada lado sin depender de `groups`."""
    X, y, groups = [], [], []
    for engine_index, (engine, n) in enumerate(engine_window_counts.items()):
        for i in range(n):
            X.append(np.full((3, 2), fill_value=float(engine_index), dtype=float))
            y.append(i % 3)
            groups.append(engine)
    return np.array(X), np.array(y), np.array(groups)


def _engine_indices_present(X_side: np.ndarray) -> set[float]:
    return set(np.unique(X_side))


def test_split_by_engine_never_splits_a_single_engine_across_train_and_val():
    # 20 motores con 10 ventanas solapadas cada uno (200 ventanas en total) -
    # suficientes motores para que un split accidental fila-a-fila (en vez
    # de por grupo) sea estadisticamente casi seguro de detectar.
    engines = {f"ENG_{i}": 10 for i in range(20)}
    X, y, groups = _fake_windows(engines)

    X_train, X_val, _y_train, _y_val = _split_by_engine(X, y, groups, val_split=0.25, seed=42)

    train_engine_indices = _engine_indices_present(X_train)
    val_engine_indices = _engine_indices_present(X_val)

    assert train_engine_indices.isdisjoint(val_engine_indices)
    assert len(train_engine_indices) + len(val_engine_indices) == len(engines)


def test_split_by_engine_keeps_every_window_of_the_same_engine_together():
    engines = {"ENG_A": 8, "ENG_B": 8, "ENG_C": 8}
    X, y, groups = _fake_windows(engines)

    X_train, X_val, _y_train, _y_val = _split_by_engine(X, y, groups, val_split=0.34, seed=7)

    # Cada motor aporta exactamente 8 ventanas: si alguna hubiera cruzado al
    # otro lado, el conteo total ya no cuadraria con multiplos de 8.
    assert len(X_train) % 8 == 0
    assert len(X_val) % 8 == 0
    assert len(X_train) + len(X_val) == len(X)


def test_split_by_engine_falls_back_to_using_everything_below_two_engines():
    X, y, groups = _fake_windows({"ENG_ONLY": 12})

    X_train, X_val, y_train, y_val = _split_by_engine(X, y, groups, val_split=0.2, seed=42)

    # Con un solo motor no hay forma de reservar val sin partirlo: el
    # fallback documentado en _split_by_engine usa todo el dataset como
    # train y como val en vez de lanzar el ValueError que GroupShuffleSplit
    # tiraria ("el train set quedaria vacio").
    assert len(X_train) == len(X_val) == len(X)
    np.testing.assert_array_equal(y_train, y_val)


def test_quality_gate_requires_both_thresholds_to_pass():
    # Buen F2 global, mal recall en Critical: el gate no debe pasar aunque
    # F2 solo sí lo haría - es exactamente el caso que critical_recall_threshold
    # existe para atrapar (un modelo que compensa fallando en la clase que
    # mas importa).
    assert not _evaluate_quality_gate(
        f2_weighted=0.90, critical_recall=0.40, f2_threshold=0.75, critical_recall_threshold=0.75
    )


def test_quality_gate_rejects_good_critical_recall_with_bad_f2():
    assert not _evaluate_quality_gate(
        f2_weighted=0.50, critical_recall=0.95, f2_threshold=0.75, critical_recall_threshold=0.75
    )


def test_quality_gate_passes_when_both_thresholds_are_met():
    assert _evaluate_quality_gate(
        f2_weighted=0.80, critical_recall=0.80, f2_threshold=0.75, critical_recall_threshold=0.75
    )


def test_training_data_fingerprint_is_deterministic_and_split_sensitive():
    X, y, groups = _fake_windows({"ENG_A": 4, "ENG_B": 4})

    base = _training_data_fingerprint(X, y, groups, val_split=0.2, seed=42)

    assert base == _training_data_fingerprint(X, y, groups, val_split=0.2, seed=42)
    assert base != _training_data_fingerprint(X, y, groups, val_split=0.3, seed=42)
    assert base != _training_data_fingerprint(X, y, groups, val_split=0.2, seed=7)
    assert base != _training_data_fingerprint(X * 2, y, groups, val_split=0.2, seed=42)


class _FakeClient:
    """Imita lo minimo de MlflowClient que usa _should_replace_champion."""

    def __init__(self, fingerprint: str | None = None, f2: float | None = None) -> None:
        self._has_champion = fingerprint is not None
        self._fingerprint = fingerprint
        self._f2 = f2

    def get_model_version_by_alias(self, name, alias):
        if not self._has_champion:
            raise MlflowException(f"Registered model alias {alias} not found.")
        return SimpleNamespace(version="3", run_id="run-champion")

    def get_run(self, run_id):
        metrics = {} if self._f2 is None else {"best_val_f2_weighted": self._f2}
        return SimpleNamespace(
            data=SimpleNamespace(
                tags={"training_data_fingerprint": self._fingerprint}, metrics=metrics
            )
        )


def test_champion_is_replaced_when_there_is_none_yet():
    assert _should_replace_champion(_FakeClient(), "m", "fp", challenger_f2=0.1)


def test_champion_is_kept_when_challenger_is_worse_on_the_same_data():
    client = _FakeClient(fingerprint="fp", f2=0.90)
    assert not _should_replace_champion(client, "m", "fp", challenger_f2=0.80)


def test_champion_is_replaced_when_challenger_is_better_on_the_same_data():
    client = _FakeClient(fingerprint="fp", f2=0.80)
    assert _should_replace_champion(client, "m", "fp", challenger_f2=0.85)


def test_gate_alone_decides_when_runs_are_not_comparable():
    # Mismo F2 "peor", pero otra huella de datos: comparar no tiene sentido.
    client = _FakeClient(fingerprint="other-data", f2=0.99)
    assert _should_replace_champion(client, "m", "fp", challenger_f2=0.50)


def test_checkpoint_that_passes_the_gate_beats_one_with_higher_f2_that_fails():
    # Caso real observado: mejor F2 pero recall Critical bajo el piso.
    thresholds = {"f2_threshold": 0.75, "critical_recall_threshold": 0.75}
    fails_gate = _checkpoint_score(0.826, 0.61, **thresholds)
    passes_gate = _checkpoint_score(0.807, 0.86, **thresholds)

    assert passes_gate > fails_gate


def test_checkpoint_score_falls_back_to_f2_when_nothing_passes_the_gate():
    thresholds = {"f2_threshold": 0.99, "critical_recall_threshold": 0.99}
    assert _checkpoint_score(0.80, 0.5, **thresholds) > _checkpoint_score(0.70, 0.9, **thresholds)

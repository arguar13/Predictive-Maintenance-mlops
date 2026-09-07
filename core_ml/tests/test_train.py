import numpy as np

from train import _split_by_engine


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

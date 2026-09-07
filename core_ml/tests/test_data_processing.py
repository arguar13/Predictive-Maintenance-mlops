import pandas as pd

from data_processing import build_multiclass_target, clean_and_prepare, create_sliding_windows


def _sample_unit_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "unit_number": [1, 1, 1, 1],
            "time_in_cycles": [1, 2, 3, 100],
            "global_unit": ["FD001_1", "FD001_1", "FD001_1", "FD001_1"],
            "dataset_id": ["FD001", "FD001", "FD001", "FD001"],
            "op_setting_1": [0.0, 0.0, 0.0, 0.0],
            "op_setting_2": [0.0, 0.0, 0.0, 0.0],
            "op_setting_3": [100.0, 100.0, 100.0, 100.0],
            "sensor_1": [0.1, 0.2, 0.3, 0.4],
            "sensor_2": [1.0, 1.0, 1.0, 1.0],  # sensor sin varianza (invariante)
        }
    )


def test_build_multiclass_target_labels_by_remaining_useful_life():
    df = _sample_unit_df()

    result = build_multiclass_target(df)

    # RUL = max_cycle (100) - time_in_cycles; con RUL > 60 el ciclo es Healthy
    assert result["RUL"].tolist() == [99, 98, 97, 0]
    assert result["failure_type"].tolist() == [0, 0, 0, 2]


def test_build_multiclass_target_flags_critical_near_end_of_life():
    df = _sample_unit_df().iloc[:2].copy()
    df["time_in_cycles"] = [1, 100]  # RUL: 99, 0 -> Healthy, Critical

    result = build_multiclass_target(df)

    assert result["failure_type"].tolist() == [0, 2]


def test_clean_and_prepare_drops_invariant_sensors_and_helper_columns():
    df = build_multiclass_target(_sample_unit_df())

    cleaned = clean_and_prepare(df)

    assert "sensor_2" not in cleaned.columns  # sin varianza -> eliminado
    assert "sensor_1" in cleaned.columns
    assert "RUL" not in cleaned.columns
    assert "dataset_id" not in cleaned.columns


def test_clean_and_prepare_drops_duplicate_rows():
    df = build_multiclass_target(_sample_unit_df())
    df_with_dupe = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    cleaned = clean_and_prepare(df_with_dupe)

    assert len(cleaned) == len(df)


def _two_unit_df(cycles_per_unit: int = 5) -> pd.DataFrame:
    """Dos motores con `cycles_per_unit` filas cada uno - suficiente para
    generar mas de una ventana por motor con window_size=3, que es lo que
    hace falta para probar que create_sliding_windows no mezcla motores."""
    rows = []
    for unit_idx, unit in enumerate(["FD001_1", "FD001_2"]):
        for cycle in range(1, cycles_per_unit + 1):
            rows.append(
                {
                    "unit_number": unit_idx + 1,
                    "time_in_cycles": cycle,
                    "global_unit": unit,
                    # Valor distinto por motor: permite reconocer, a partir
                    # del propio contenido de la ventana, de que motor vino.
                    "sensor_1": float(unit_idx * 1000 + cycle),
                    "failure_type": 0,
                }
            )
    return pd.DataFrame(rows)


def test_create_sliding_windows_tags_every_window_with_its_source_engine():
    df = _two_unit_df(cycles_per_unit=5)

    X, y, groups, _scaler = create_sliding_windows(df, window_size=3)

    # 5 ciclos, ventana de 3 -> 3 ventanas por motor, 2 motores -> 6 ventanas.
    assert X.shape == (6, 3, 1)
    assert len(y) == len(groups) == 6
    assert set(groups) == {"FD001_1", "FD001_2"}
    # sensor_1 se construyo como unit_idx*1000 + cycle: el rango de FD001_1
    # (1-5) queda muy por debajo de la media combinada con FD001_2
    # (1000-1004), asi que tras el StandardScaler interno el signo del
    # valor escalado por si solo ya delata de que motor vino cada ventana -
    # una forma robusta de verificar que create_sliding_windows nunca
    # mezcla filas de dos motores distintos dentro de una misma ventana.
    for window, group in zip(X, groups, strict=True):
        if group == "FD001_1":
            assert (window < 0).all()
        else:
            assert (window > 0).all()

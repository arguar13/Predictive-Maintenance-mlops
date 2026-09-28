import numpy as np
import pandas as pd
import pytest

from data_processing import (
    build_multiclass_target,
    clean_and_prepare,
    create_sliding_windows,
    load_and_combine_data,
    select_engines,
)


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


def test_create_sliding_windows_does_not_mutate_input_and_returns_float32():
    df = _two_unit_df(cycles_per_unit=5)
    original = df.copy()

    X, y, _groups, _scaler = create_sliding_windows(df, window_size=3)

    pd.testing.assert_frame_equal(df, original)
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_create_sliding_windows_orders_cycles_within_each_engine():
    df = _two_unit_df(cycles_per_unit=5).sample(frac=1.0, random_state=0)

    X, _y, groups, scaler = create_sliding_windows(df, window_size=3)

    # Con los ciclos ordenados, cada ventana es estrictamente creciente en
    # sensor_1 (que se construyo como unit_idx*1000 + cycle).
    for window in X:
        assert np.all(np.diff(window[:, 0]) > 0)
    assert list(scaler.feature_names_in_) == ["sensor_1"]
    assert len(groups) == 6


def test_create_sliding_windows_skips_engines_shorter_than_the_window():
    df = pd.concat(
        [
            _two_unit_df(cycles_per_unit=5),
            _two_unit_df(cycles_per_unit=2).iloc[:2].assign(global_unit="X"),
        ]
    )

    _X, _y, groups, _scaler = create_sliding_windows(df, window_size=3)

    assert "X" not in set(groups)


def _multi_dataset_df(engines_per_dataset: int = 10, cycles: int = 3) -> pd.DataFrame:
    rows = []
    for ds in ["FD001", "FD002", "FD003", "FD004"]:
        for unit in range(1, engines_per_dataset + 1):
            for cycle in range(1, cycles + 1):
                rows.append(
                    {
                        "unit_number": unit,
                        "time_in_cycles": cycle,
                        "dataset_id": ds,
                        "global_unit": f"{ds}_{unit}",
                    }
                )
    return pd.DataFrame(rows)


def test_select_engines_samples_whole_engines_evenly_across_sub_datasets():
    df = _multi_dataset_df(engines_per_dataset=10, cycles=3)

    selected = select_engines(df, max_engines=8, seed=0)

    per_dataset = selected.groupby("dataset_id")["global_unit"].nunique()
    assert per_dataset.to_dict() == {"FD001": 2, "FD002": 2, "FD003": 2, "FD004": 2}
    # Cada motor elegido conserva todos sus ciclos (nunca filas sueltas).
    assert (selected.groupby("global_unit").size() == 3).all()


def test_select_engines_is_deterministic_and_optional():
    df = _multi_dataset_df()

    first = select_engines(df, max_engines=8, seed=1)
    second = select_engines(df, max_engines=8, seed=1)

    pd.testing.assert_frame_equal(first, second)
    assert len(select_engines(df, max_engines=0)) == len(df)
    assert len(select_engines(df, max_engines=None)) == len(df)


def test_load_and_combine_data_fails_fast_when_no_files_are_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_FD00X"):
        load_and_combine_data(str(tmp_path))

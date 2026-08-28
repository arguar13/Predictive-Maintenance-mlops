import pandas as pd

from data_processing import build_multiclass_target, clean_and_prepare


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

import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_contracts import validate_labeled_telemetry, validate_raw_telemetry

logger = logging.getLogger(__name__)


def load_and_combine_data(data_dir: str) -> pd.DataFrame:
    """Carga y une los datasets FD001 a FD004 de C-MAPSS garantizando identificadores únicos.

    FAIL FAST: el resultado se valida contra el contrato de datos crudos
    antes de devolverse, para no propagar telemetría inválida (nulos,
    unidades/ciclos no positivos) al resto del pipeline.
    """
    index_names = ["unit_number", "time_in_cycles"]
    setting_names = ["op_setting_1", "op_setting_2", "op_setting_3"]
    sensor_names = [f"sensor_{i}" for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    datasets = ["FD001", "FD002", "FD003", "FD004"]
    train_list = []

    for ds in datasets:
        file_path = os.path.join(data_dir, f"train_{ds}.txt")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep=r"\s+", header=None, names=col_names)
            df["dataset_id"] = ds
            df["global_unit"] = df["dataset_id"] + "_" + df["unit_number"].astype(str)
            train_list.append(df)

    combined = pd.concat(train_list, ignore_index=True)
    return validate_raw_telemetry(combined)


def build_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma el problema en clasificación multiclase (Healthy, Alert, Critical).

    FAIL FAST: valida RUL/failure_type antes de devolver, evitando entrenar
    con etiquetas fuera del contrato (p.ej. una clase inexistente).
    """
    df = df.copy()
    max_cycles = df.groupby("global_unit")["time_in_cycles"].transform("max")
    df["RUL"] = max_cycles - df["time_in_cycles"]

    df["failure_type"] = 0  # Healthy
    df.loc[df["RUL"] <= 60, "failure_type"] = 1  # Alert
    df.loc[df["RUL"] <= 30, "failure_type"] = 2  # Critical
    return validate_labeled_telemetry(df)


def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados y sensores sin varianza (invariantes)."""
    df = df.drop_duplicates()
    sensor_cols = [col for col in df.columns if "sensor" in col]
    std_dev = df[sensor_cols].std()
    invariant_sensors = std_dev[std_dev < 1e-6].index.tolist()

    df = df.drop(columns=invariant_sensors + ["RUL", "dataset_id"])
    return df


def create_sliding_windows(df: pd.DataFrame, window_size: int = 30):
    """Genera ventanas tridimensionales (Muestras, Ventana Temporal,
    Características) para PyTorch.

    Devuelve también `groups`: el `global_unit` (motor físico) que originó
    cada ventana. Con stride=1, las ventanas de un mismo motor se solapan
    fuertemente entre sí, así que un split que no agrupe por motor (p.ej.
    train_test_split fila a fila) deja ventanas casi idénticas del mismo
    motor a ambos lados del split - el consumidor de este array (train.py)
    debe usar `groups` con GroupShuffleSplit, nunca un split IID sobre las
    filas de X/y directamente.

    Nota: el StandardScaler se ajusta sobre TODAS las filas de `df`, no solo
    sobre un futuro subconjunto de entrenamiento. Es una fuga de información
    más leve que la del split (estadísticas globales de escalado, no
    duplicación de muestras) y separar el fit por partición requeriría que
    esta etapa de preprocesamiento (que no conoce val_split, un parámetro de
    train.py que se decide por corrida) y el split de entrenamiento
    coordinen una misma partición de motores - un acoplamiento entre etapas
    desproporcionado para el tamaño del riesgo. Aceptado conscientemente,
    no pasado por alto.
    """
    features = [
        c
        for c in df.columns
        if c not in ["unit_number", "time_in_cycles", "global_unit", "failure_type"]
    ]

    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    X, y, groups = [], [], []
    for unit in df["global_unit"].unique():
        unit_data = df[df["global_unit"] == unit].copy()
        unit_data.reset_index(drop=True, inplace=True)

        for i in range(len(unit_data) - window_size + 1):
            X.append(unit_data[features].iloc[i : i + window_size].values)
            # Etiqueta del último ciclo de la ventana
            y.append(unit_data["failure_type"].iloc[i + window_size - 1])
            groups.append(unit)

    return np.array(X), np.array(y), np.array(groups), scaler

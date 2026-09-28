import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_contracts import validate_labeled_telemetry, validate_raw_telemetry

# Columnas que identifican/etiquetan una fila pero no son features del modelo.
NON_FEATURE_COLUMNS = ("unit_number", "time_in_cycles", "global_unit", "failure_type")


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

    if not train_list:
        # Sin esto, pd.concat([]) falla con un "No objects to concatenate"
        # que no dice que archivo faltaba ni donde se busco.
        raise FileNotFoundError(
            f"No se encontro ningun train_FD00X.txt en {data_dir}. "
            "Descarga los datos con `make dvc-pull` (o copialos a mano)."
        )

    combined = pd.concat(train_list, ignore_index=True)
    return validate_raw_telemetry(combined)


def select_engines(df: pd.DataFrame, max_engines: int | None, seed: int = 42) -> pd.DataFrame:
    """Submuestrea MOTORES completos (nunca filas sueltas), estratificado por sub-dataset.

    Limitar el volumen cortando "las primeras N ventanas" sesga el
    entrenamiento: los .txt estan ordenados por sub-dataset y motor, asi que
    las primeras ventanas son todas de los primeros motores de FD001 (una sola
    condicion operativa, un solo modo de fallo), mientras el scaler y el
    numero de features salen del dataset combinado. Aqui cada sub-dataset
    aporta el mismo cupo de motores elegidos al azar (con semilla fija), y
    cada motor se conserva entero para no romper su serie temporal.
    `max_engines=None` (o <= 0) conserva todos los motores.
    """
    if not max_engines or max_engines <= 0:
        return df

    engines = df[["dataset_id", "global_unit"]].drop_duplicates()
    if len(engines) <= max_engines:
        return df

    rng = np.random.default_rng(seed)
    per_dataset = max(1, max_engines // engines["dataset_id"].nunique())
    selected: list[str] = []
    for _, group in engines.groupby("dataset_id", sort=True):
        units = group["global_unit"].to_numpy()
        take = min(per_dataset, len(units))
        selected.extend(rng.choice(units, size=take, replace=False).tolist())

    return df[df["global_unit"].isin(selected)].reset_index(drop=True)


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


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columnas que entran al modelo, en el orden en que las espera el scaler."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def create_sliding_windows(
    df: pd.DataFrame, window_size: int = 30
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Genera ventanas tridimensionales (Muestras, Ventana Temporal,
    Características) SIN escalar, y el orden de las features.

    Devuelve también `groups`: el `global_unit` (motor físico) que originó
    cada ventana. Con stride=1, las ventanas de un mismo motor se solapan
    fuertemente entre sí, así que un split que no agrupe por motor (p.ej.
    train_test_split fila a fila) deja ventanas casi idénticas del mismo
    motor a ambos lados del split - el consumidor de este array (train.py)
    debe usar `groups` con GroupShuffleSplit, nunca un split IID sobre las
    filas de X/y directamente.

    El escalado NO ocurre aquí: esta etapa no conoce el split por motor, y
    ajustar el scaler sobre todos los motores filtraría estadísticas de
    val/test al entrenamiento. train.py lo ajusta solo con los motores de
    train (ver `fit_window_scaler`).
    """
    features = feature_columns(df)

    # Orden temporal garantizado dentro de cada motor: las ventanas asumen
    # ciclos consecutivos, y no se modifica el DataFrame del llamador.
    ordered = df.sort_values(["global_unit", "time_in_cycles"], kind="stable")

    values = ordered[features].to_numpy(dtype=np.float32)
    units = ordered["global_unit"].to_numpy()
    labels = ordered["failure_type"].to_numpy()

    # Tras ordenar, las filas de cada motor son contiguas: basta con los
    # indices donde cambia global_unit para delimitar cada serie.
    boundaries = np.flatnonzero(units[1:] != units[:-1]) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(units)]))

    X_parts, y_parts, group_parts = [], [], []
    for start, end in zip(starts, ends, strict=True):
        n_windows = (end - start) - window_size + 1
        if n_windows <= 0:
            continue
        # sliding_window_view devuelve (n_windows, n_features, window_size):
        # vistas sin copia, en vez de un bucle Python con .iloc por ventana.
        windows = np.lib.stride_tricks.sliding_window_view(values[start:end], window_size, axis=0)
        X_parts.append(windows.transpose(0, 2, 1))
        # Etiqueta del último ciclo de la ventana
        y_parts.append(labels[start + window_size - 1 : end])
        group_parts.append(np.full(n_windows, units[start], dtype=object))

    if not X_parts:
        raise ValueError(
            f"Ningun motor tiene al menos window_size={window_size} ciclos: no hay ventanas."
        )

    X = np.ascontiguousarray(np.concatenate(X_parts), dtype=np.float32)
    y = np.concatenate(y_parts).astype(np.int64)
    groups = np.concatenate(group_parts)
    return X, y, groups, features


def fit_window_scaler(X_train: np.ndarray, feature_names: list[str]) -> StandardScaler:
    """Ajusta el StandardScaler SOLO con las ventanas de train.

    Se ajusta sobre un DataFrame con los nombres de columna para que el scaler
    guarde `feature_names_in_`: la API lo usa para publicar el orden exacto de
    columnas que espera cada fila de /predict. Cada ciclo aparece en varias
    ventanas solapadas; eso pondera un poco más los ciclos centrales, pero
    media y desviación resultan prácticamente iguales a las por fila.
    """
    rows = X_train.reshape(-1, X_train.shape[2])
    return StandardScaler().fit(pd.DataFrame(rows, columns=feature_names))


def scale_windows(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Aplica el scaler (ajustado en train) a ventanas (n, window, features)."""
    flat = pd.DataFrame(X.reshape(-1, X.shape[2]), columns=scaler.feature_names_in_)
    return np.asarray(scaler.transform(flat), dtype=np.float32).reshape(X.shape)

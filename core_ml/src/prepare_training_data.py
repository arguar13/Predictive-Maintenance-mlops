"""Construye engine_features.parquet + feature_names.json a partir de los .txt
crudos de C-MAPSS.

Único paso de preparación de datos del pipeline: transforma los .txt crudos
en las ventanas que `train.py` consume, y las persiste como Parquet. Sin
online store ni nada sirviendo features en tiempo real, un `pd.read_parquet`
directo (ver `train.py`) resuelve la lectura sin infraestructura adicional
que mantener.
"""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from config_loader import load_config
from data_contracts import validate_windowed_features
from data_processing import (
    build_multiclass_target,
    clean_and_prepare,
    create_sliding_windows,
    load_and_combine_data,
    select_engines,
)
from logging_config import configure_logging

log = configure_logging("prepare-training-data")

BASE_DIR = Path(__file__).resolve().parent.parent
# 0 = todos los motores (709, ~140k ventanas, ~400MB en float32): con el
# gate evaluado sobre motores de test no vistos, una muestra de 100 motores
# no alcanzaba el piso de recall de Critical (0.70 < 0.75) y el dataset
# completo si (0.90). `--max-engines N` sirve para iteraciones rapidas.
DEFAULT_MAX_ENGINES = 0


def _utc_now_naive() -> datetime:
    """UTC "naive" (sin tzinfo): el contrato de datos (data_contracts.py::
    WINDOWED_FEATURE_SCHEMA) espera datetime64[ns] sin zona horaria."""
    return datetime.now(UTC).replace(tzinfo=None)


def generate_training_parquet(
    data_dir: str = "data",
    max_engines: int | None = DEFAULT_MAX_ENGINES,
    seed: int = 42,
) -> Path:
    """Construye engine_features.parquet (ventanas SIN escalar) + feature_names.json.

    `data_dir` es relativo a core_ml/ (p.ej. "data" para el dataset completo
    o "data_toy" para el dataset de ~1000 filas versionado con DVC). El
    scaler no se ajusta aqui: esta etapa no conoce el split por motor, asi
    que lo ajusta train.py solo con los motores de train y lo registra en
    MLflow junto al modelo.
    """
    window_size = load_config()["model"]["window_size"]
    data_path = BASE_DIR / data_dir
    log.info("preprocessing_started", data_path=str(data_path), max_engines=max_engines)

    # 1. Ejecutar el pipeline de data_processing.py (cada etapa valida su
    # propio contrato de datos internamente -> FAIL FAST). El submuestreo se
    # hace por MOTOR y ANTES de ventanear: el conjunto de features sale
    # exactamente de los datos con los que se entrena.
    raw_df = load_and_combine_data(str(data_path))
    raw_df = select_engines(raw_df, max_engines=max_engines, seed=seed)
    target_df = build_multiclass_target(raw_df)
    clean_df = clean_and_prepare(target_df)

    # Extraer las ventanas tridimensionales
    X, y, groups, feature_names = create_sliding_windows(clean_df, window_size=window_size)
    num_features = X.shape[2]
    log.info(
        "windows_created",
        engines=int(raw_df["global_unit"].nunique()),
        windows=len(X),
        num_features=num_features,
        features=feature_names,
    )

    # 2. Aplanar las ventanas a filas de un DataFrame (el formato que
    # train.py espera, ver data_contracts.py::WINDOWED_FEATURE_SCHEMA).
    log.info("flattening_windows")
    records = []

    # event_timestamp/created_timestamp son parte del contrato de datos
    # (WINDOWED_FEATURE_SCHEMA exige datetime64[ns]), aunque train.py no las
    # use como feature: documentan cuándo se generó cada ventana.
    base_time = _utc_now_naive() - timedelta(hours=23)

    for i in range(len(X)):
        # float32 (no la lista de floats de Python, que Parquet guarda como
        # double): mismo dtype que el tensor del modelo y la mitad de disco.
        flat_features = X[i].ravel()
        records.append(
            {
                # global_unit real (p.ej. "FD001_23"), no un id sintético: un
                # engine_id fabricado por índice (i % N) no se corresponde con
                # qué motor generó la ventana, y es exactamente el id que
                # train.py necesita para agrupar por motor al hacer el split
                # train/val (ver create_sliding_windows).
                "engine_id": str(groups[i]),
                "event_timestamp": base_time + timedelta(seconds=i * 10),
                "created_timestamp": _utc_now_naive(),
                "windowed_features": flat_features,
                "failure_type": int(y[i]),
            }
        )

    features_df = pd.DataFrame(records)

    # FAIL FAST: valida el contrato de las ventanas antes de escribir nada a disco.
    expected_length = window_size * num_features
    features_df = validate_windowed_features(features_df, expected_length=expected_length)

    # 3. Guardar como Parquet (única fuente de datos de entrenamiento que
    # train.py lee).
    features_path = data_path / "engine_features.parquet"
    features_df.to_parquet(features_path, index=False)
    log.info("parquet_written", features_path=str(features_path), rows=len(features_df))

    # 4. Orden de las columnas dentro de cada ventana aplanada: train.py lo
    # necesita para ajustar un scaler con nombres de feature (feature_names_in_),
    # que es lo que la API publica en /health.
    feature_names_path = data_path / "feature_names.json"
    feature_names_path.write_text(json.dumps(feature_names, indent=2))
    log.info("feature_names_written", path=str(feature_names_path), num_features=num_features)

    return data_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directorio (relativo a core_ml/) con los .txt crudos de entrada "
        "y donde se escriben los artefactos. Usa 'data_toy' para el dataset "
        "pequeño versionado con DVC.",
    )
    parser.add_argument(
        "--max-engines",
        type=int,
        default=DEFAULT_MAX_ENGINES,
        help="Numero maximo de motores a usar, repartidos por igual entre FD001-FD004 "
        "(0 = todos). Se submuestrean motores enteros, nunca ventanas sueltas.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_training_parquet(data_dir=args.data_dir, max_engines=args.max_engines, seed=args.seed)

import argparse
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd

from data_contracts import validate_windowed_features
from data_processing import (
    build_multiclass_target,
    clean_and_prepare,
    create_sliding_windows,
    load_and_combine_data,
)
from logging_config import configure_logging

log = configure_logging("prepare-feast-data")

BASE_DIR = Path(__file__).resolve().parent.parent
WINDOW_SIZE = 30


def _utc_now_naive() -> datetime:
    """UTC "naive" (sin tzinfo): Feast y el esquema de ventanas esperan
    datetime64[ns] sin zona horaria, no datetime64[ns, UTC]."""
    return datetime.now(UTC).replace(tzinfo=None)


def generate_feast_parquet(data_dir: str = "data", window_limit: int = 5000) -> Path:
    """Construye engine_features.parquet + training_entities.parquet + scaler.joblib.

    `data_dir` es relativo a core_ml/ (p.ej. "data" para el dataset completo
    o "data_toy" para el dataset de ~1000 filas versionado con DVC). Todos
    los artefactos de una ejecución (parquet + scaler) quedan en el MISMO
    directorio versionado por DVC, para que el scaler nunca sea un archivo
    suelto sin relación con los datos que lo generaron.
    """
    data_path = BASE_DIR / data_dir
    log.info("preprocessing_started", data_path=str(data_path))

    # 1. Ejecutar el pipeline de data_processing.py (cada etapa valida su
    # propio contrato de datos internamente -> FAIL FAST).
    raw_df = load_and_combine_data(str(data_path))
    target_df = build_multiclass_target(raw_df)
    clean_df = clean_and_prepare(target_df)

    # Extraer las ventanas tridimensionales
    X, y, groups, scaler = create_sliding_windows(clean_df, window_size=WINDOW_SIZE)
    num_features = X.shape[2]

    # 2. Formatear los datos para cumplir con el esquema de Feast (features.py)
    log.info("formatting_for_feast")
    records = []

    # Feast requiere timestamps. Simularemos que los datos llegaron en las últimas 24 horas
    # para que el TTL de Feast los acepte en el Online Store.
    base_time = _utc_now_naive() - timedelta(hours=23)

    # Para no saturar la RAM local, procesamos solo un subconjunto
    # (por defecto las primeras 5000 ventanas). En producción esto se hace con PySpark.
    limit = min(len(X), window_limit)

    for i in range(limit):
        # Feast espera un Array(Float32) aplanado
        flat_features = X[i].flatten().tolist()
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

    feast_df = pd.DataFrame(records)

    # FAIL FAST: valida el contrato de las ventanas antes de escribir nada a disco.
    feast_df = validate_windowed_features(feast_df, expected_length=WINDOW_SIZE * num_features)

    # 3. Guardar como Parquet maestro (para el Offline Store de Feast)
    master_path = data_path / "engine_features.parquet"
    feast_df.to_parquet(master_path, index=False)
    log.info("parquet_written", master_path=str(master_path), rows=len(feast_df))

    # 4. Generar el Entity DataFrame (para que train.py lo consuma)
    log.info("extracting_entity_dataframe")
    entity_df = feast_df[["engine_id", "event_timestamp"]].copy()

    # Práctica Senior: Barajar (shuffle) los eventos para evitar sesgos
    # de orden de llegada en el entrenamiento
    entity_df = entity_df.sample(frac=1, random_state=42).reset_index(drop=True)

    entity_path = data_path / "training_entities.parquet"
    entity_df.to_parquet(entity_path, index=False)
    log.info("entity_dataframe_written", entity_path=str(entity_path))

    # 5. Persistir el scaler DENTRO del mismo directorio versionado por DVC
    # que los datos que lo generaron (nunca un archivo suelto en models/).
    scaler_path = data_path / "scaler.joblib"
    joblib.dump(scaler, scaler_path)
    log.info("scaler_written", scaler_path=str(scaler_path), num_features=num_features)

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
        "--window-limit",
        type=int,
        default=5000,
        help="Numero maximo de ventanas a materializar (evita saturar RAM local).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_feast_parquet(data_dir=args.data_dir, window_limit=args.window_limit)

"""Entrenamiento del ConvTransformer de mantenimiento predictivo con trazabilidad MLflow.

Cada ejecución queda ligada, de forma inequívoca, a la tupla:
    Git Commit Hash + DVC Data Hash + Hyperparameters + MLflow Run ID +
    Container Image Tag
a través de tags/params del run de MLflow (ver `_build_lineage_tags`).

Modos de carga de datos (--data-source):
  * "feast"   (por defecto, producción): lee las features materializadas vía
    Feast Offline Store (S3 + point-in-time join). Requiere credenciales AWS.
  * "parquet" (toy / smoke test local): lee engine_features.parquet
    directamente del disco, sin Feast/S3/Redis. Es el modo usado para
    validar el pipeline end-to-end en segundos, con el dataset toy
    versionado con DVC, antes de gastar cómputo real (GPU) con el dataset
    completo.

El modelo y el scaler NUNCA quedan como archivos sueltos: el modelo se
registra en el MLflow Model Registry y el scaler se loguea como artefacto
del mismo run. Solo se promueve al alias "champion" (el que sirve la API)
si supera el Quality Gate (`monitoring.accuracy_threshold` en config.yaml).
"""

from __future__ import annotations

import argparse
import math
import os

# subprocess solo se usa con argv fijo (sin input externo) en _get_git_commit_sha
import subprocess  # nosec B404
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from config_loader import load_config
from data_contracts import validate_training_batch
from logging_config import configure_logging

log = configure_logging("train")

BASE_DIR = Path(__file__).resolve().parent.parent
WINDOW_SIZE = 30
NUM_CLASSES = 3


class PositionalEncoding(nn.Module):
    """Codificación posicional sinusoidal estándar (Vaswani et al.) para el
    TransformerEncoder de ConvTransformer: sin ella, la atención es invariante
    al orden temporal de la ventana, y el orden de los 30 timesteps es
    justamente la señal que la convolución+transformer deben aprovechar.
    """

    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ConvTransformer(nn.Module):
    """Hibrido CNN + self-attention: una Conv1d local extrae patrones de
    sensores de corto plazo antes de pasarlos al TransformerEncoder, que
    modela dependencias de largo plazo entre timesteps de la ventana.

    Reemplaza al FCN puramente convolucional que este proyecto usaba antes:
    en un benchmark propio (5 arquitecturas sobre este mismo tipo de tarea
    C-MAPSS) fue la de mejor Macro F1/Accuracy, por delante de
    InceptionTime, Vanilla Transformer, PatchTST y el FCN baseline. Se omiten
    las ramas de features estaticas/categoricas del benchmark original
    (embedding de "dataset_id", features numericas estaticas): el pipeline
    de Feast de este proyecto solo produce `windowed_features`, sin esas
    columnas adicionales.
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = NUM_CLASSES,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=num_features, out_channels=d_model, kernel_size=3, padding=1
        )
        self.bn = nn.BatchNorm1d(d_model)
        self.relu = nn.ReLU()
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window_size, num_features]
        x = x.transpose(1, 2)  # [batch, num_features, window_size] para Conv1d
        x = self.relu(self.bn(self.conv(x)))
        x = x.transpose(1, 2)  # [batch, window_size, d_model] para el transformer (batch_first)
        x = self.pos_encoder(x)
        out = self.transformer(x)
        pooled = out.mean(dim=1)  # Global average pooling sobre los timesteps
        return self.classifier(pooled)


# ---------------------------------------------------------------------------
# Tupla de trazabilidad: Git commit + DVC data hash + container image tag
# ---------------------------------------------------------------------------


def _get_git_commit_sha() -> str:
    """Prioriza la variable de CI (el checkout de un runner puede ser un
    shallow-clone sin refs completas) y cae a `git rev-parse HEAD` en local."""
    ci_sha = os.environ.get("CI_COMMIT_SHA")
    if ci_sha:
        return ci_sha
    try:
        # Lista de argv fija (sin input de usuario) y shell=False (por
        # defecto): bandit igual marca subprocess/ruta-parcial por defecto,
        # pero aquí no hay superficie de inyección de comandos.
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _get_dvc_data_hash(data_dir: str) -> str:
    """Lee el hash md5 de `<data_dir>.dvc` (p.ej. data.dvc o data_toy.dvc)."""
    dvc_file = BASE_DIR / f"{data_dir}.dvc"
    if not dvc_file.exists():
        return "unknown"
    with open(dvc_file) as file:
        metadata = yaml.safe_load(file) or {}
    outs = metadata.get("outs") or [{}]
    return outs[0].get("md5", "unknown")


def _get_container_image_tag() -> str:
    return os.environ.get("IMAGE_TAG") or os.environ.get("CI_COMMIT_SHA") or "local-dev"


def _build_lineage_tags(data_dir: str, data_source: str) -> dict[str, str]:
    return {
        "git_commit_sha": _get_git_commit_sha(),
        "dvc_data_hash": _get_dvc_data_hash(data_dir),
        "dvc_data_dir": data_dir,
        "container_image_tag": _get_container_image_tag(),
        "data_source": data_source,
    }


# ---------------------------------------------------------------------------
# Carga de datos de entrenamiento
# ---------------------------------------------------------------------------


def _load_training_data_via_parquet(data_path: Path) -> pd.DataFrame:
    parquet_path = data_path / "engine_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No se encontró {parquet_path}. Ejecuta antes:\n"
            f"  poetry run python src/prepare_feast_data.py --data-dir {data_path.name}"
        )
    return pd.read_parquet(parquet_path)


def _load_training_data_via_feast(data_path: Path, feature_store_path: Path) -> pd.DataFrame:
    from feast import FeatureStore  # import perezoso: solo requerido en modo feast

    entity_path = data_path / "training_entities.parquet"
    if not entity_path.exists():
        raise FileNotFoundError(
            f"No se encontró el Entity DataFrame en {entity_path}. "
            "Debes ejecutar 'prepare_feast_data.py' primero."
        )

    log.info("feast_offline_store_extraction_started")
    store = FeatureStore(repo_path=str(feature_store_path))

    log.info("loading_entities", entity_path=str(entity_path))
    entity_df = pd.read_parquet(entity_path)

    return store.get_historical_features(
        entity_df=entity_df,
        features=[
            "engine_sensor_window_features:windowed_features",
            "engine_sensor_window_features:failure_type",
        ],
    ).to_df()


# ---------------------------------------------------------------------------
# Pipeline de entrenamiento
# ---------------------------------------------------------------------------


def train_pipeline(
    data_dir: str = "data",
    data_source: str = "feast",
    epochs: int = 3,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    val_split: float = 0.2,
    enforce_quality_gate: bool = True,
    seed: int = 42,
    patience: int = 5,
) -> str:
    # Reproducibilidad: sin esto, la inicializacion de pesos de ConvTransformer
    # y el orden de batches del DataLoader (shuffle=True) son no-deterministas
    # -- el mismo commit + mismos datos + mismos hiperparametros podia pasar
    # el quality gate en una corrida del pipeline y fallar en la siguiente.
    # Misma semilla que train_test_split (random_state=42) mas abajo.
    torch.manual_seed(seed)
    config = load_config()
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", config["model"]["mlflow_tracking_uri"])
    )
    mlflow.set_experiment("Predictive_Maintenance_FCN")

    data_path = BASE_DIR / data_dir
    feature_store_path = BASE_DIR / "feature_store"

    log.info("loading_training_data", data_path=str(data_path), data_source=data_source)
    if data_source == "parquet":
        training_data = _load_training_data_via_parquet(data_path)
    elif data_source == "feast":
        training_data = _load_training_data_via_feast(data_path, feature_store_path)
    else:
        raise ValueError(f"data_source desconocido: {data_source!r} (usa 'feast' o 'parquet')")

    if len(training_data) == 0:
        raise ValueError(f"El dataset de entrenamiento en {data_path} está vacío.")

    # FAIL FAST: valida el contrato de las ventanas antes de construir tensores
    # y de gastar cómputo de entrenamiento. validate_training_batch (no
    # validate_windowed_features): a diferencia del parquet que escribe
    # prepare_feast_data.py, el batch ya cargado no trae "created_timestamp"
    # cuando viene de Feast (metadata de ingestion, no una feature pedida en
    # get_historical_features) y su "event_timestamp" es tz-aware (Feast
    # normaliza a UTC en el join), no naive.
    sample_array_length = len(training_data["windowed_features"].iloc[0])
    training_data = validate_training_batch(training_data, expected_length=sample_array_length)

    num_features = sample_array_length // WINDOW_SIZE
    log.info(
        "features_detected", num_features=num_features, sample_array_length=sample_array_length
    )

    X = np.array(
        [
            np.array(value).reshape(WINDOW_SIZE, num_features)
            for value in training_data["windowed_features"]
        ]
    )
    y = training_data["failure_type"].to_numpy()

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42, stratify=y
        )
    except ValueError:
        # Alguna clase tiene muy pocas muestras para estratificar (dataset toy diminuto)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=42
        )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True
    )

    model = ConvTransformer(num_features=num_features, num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lineage_tags = _build_lineage_tags(data_dir, data_source)
    accuracy_threshold = config["monitoring"]["accuracy_threshold"]

    with mlflow.start_run(run_name=f"ConvTransformer_Training_{data_source}") as run:
        mlflow.set_tags(lineage_tags)
        mlflow.log_params(
            {
                "window_size": WINDOW_SIZE,
                "num_features": num_features,
                "epochs": epochs,
                "patience": patience,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "val_split": val_split,
                "train_samples": len(X_train),
                "val_samples": len(X_val),
            }
        )

        # Selecciona el mejor checkpoint por val_accuracy en vez de asumir que
        # el ultimo epoch es el mejor: con el dataset completo (mucho mas
        # solapamiento entre ventanas deslizantes que en el toy dataset) se
        # observo val_accuracy colapsando por overfitting bien entrado el
        # entrenamiento (epoch 20: train_loss bajando a 0.21 pero
        # val_accuracy cayendo a 0.18, peor que adivinar al azar) mientras
        # una epoch intermedia generalizaba mejor. Sin esto, subir "epochs"
        # para dejar converger al modelo es una apuesta: puede terminar
        # exactamente en un pico malo. Es el equivalente a early stopping
        # sobre el mejor checkpoint, no una forma de forzar el quality gate.
        # Early stopping (patience): "epochs" es un TECHO, no un objetivo --
        # con ConvTransformer casi siempre converge mucho antes. Cortar en
        # cuanto `patience` epochs seguidos no mejoran val_accuracy ahorra
        # computo real sin arriesgar nada (el checkpoint restaurado sigue
        # siendo siempre el de mejor val_accuracy, nunca el ultimo).
        best_val_accuracy = -1.0
        best_state_dict: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_predictions = torch.argmax(val_outputs, dim=1)
                epoch_val_accuracy = (val_predictions == y_val_tensor).float().mean().item()
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)

            log.info(
                "epoch_completed",
                epoch=epoch + 1,
                total_epochs=epochs,
                train_loss=avg_loss,
                val_accuracy=epoch_val_accuracy,
            )

            if epoch_val_accuracy > best_val_accuracy:
                best_val_accuracy = epoch_val_accuracy
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    log.info(
                        "early_stopping",
                        epoch=epoch + 1,
                        patience=patience,
                        best_val_accuracy=best_val_accuracy,
                    )
                    break

        # best_state_dict siempre queda seteado (epoch 0 ya actualiza el
        # maximo desde -1.0 si epochs >= 1); restaurarlo deja el modelo que
        # efectivamente se registra/evalua en el estado de su mejor epoch,
        # no del ultimo. Un RuntimeError explicito (no assert: se elimina en
        # bytecode optimizado) documenta que epochs=0 nunca es un uso valido.
        if best_state_dict is None:
            raise RuntimeError(
                "epochs debe ser >= 1: no se completo ningún epoch de entrenamiento."
            )
        model.load_state_dict(best_state_dict)
        val_accuracy = best_val_accuracy

        log.info(
            "validation_completed", val_accuracy=val_accuracy, accuracy_threshold=accuracy_threshold
        )

        example_input = X_train_tensor[:1]
        signature = mlflow.models.infer_signature(
            example_input.numpy(), model(example_input).detach().numpy()
        )
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name=config["model"]["model_name"],
            input_example=example_input,
            signature=signature,
            # MLflow >=3 serializa con torch.export ("pt2") por defecto, lo
            # que produce un GraphModule que NO soporta .eval()/.train() al
            # cargarlo (rompe la API de serving). "pickle" conserva un
            # nn.Module normal, compatible con el resto del pipeline.
            serialization_format="pickle",
        )

        # El scaler viaja SIEMPRE junto al modelo, como artefacto del mismo
        # run (nunca un .joblib suelto en un directorio local).
        scaler_path = data_path / "scaler.joblib"
        if scaler_path.exists():
            mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")
        else:
            log.warning("scaler_not_found", scaler_path=str(scaler_path))

        quality_gate_passed = val_accuracy >= accuracy_threshold
        mlflow.set_tag("quality_gate_passed", str(quality_gate_passed))
        mlflow.set_tag("quality_gate_threshold", str(accuracy_threshold))

        run_id = run.info.run_id
        log.info("lineage_tuple", mlflow_run_id=run_id, **lineage_tags)

        if quality_gate_passed and model_info.registered_model_version is not None:
            client = MlflowClient()
            client.set_registered_model_alias(
                name=config["model"]["model_name"],
                alias="champion",
                version=str(model_info.registered_model_version),
            )
            log.info(
                "quality_gate_passed",
                val_accuracy=val_accuracy,
                accuracy_threshold=accuracy_threshold,
                promoted_version=model_info.registered_model_version,
                alias="champion",
            )
        else:
            log.warning(
                "quality_gate_failed",
                val_accuracy=val_accuracy,
                accuracy_threshold=accuracy_threshold,
            )

    if enforce_quality_gate and not quality_gate_passed:
        raise SystemExit(
            f"Quality gate fallido: val_accuracy={val_accuracy:.4f} < "
            f"umbral={accuracy_threshold}. No se avanza (revisa datos/hiperparámetros)."
        )

    return run_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--data-source", choices=["feast", "parquet"], default="feast")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping: corta el entrenamiento tras N epochs seguidos sin mejorar "
        "val_accuracy (siempre se restaura el mejor checkpoint, nunca el ultimo).",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument(
        "--no-enforce-quality-gate",
        dest="enforce_quality_gate",
        action="store_false",
        help="No falles el proceso si no se supera el quality gate (uso: smoke tests con "
        "el dataset toy, que no es representativo para certificar calidad de modelo).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_pipeline(
        data_dir=args.data_dir,
        data_source=args.data_source,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_split=args.val_split,
        enforce_quality_gate=args.enforce_quality_gate,
    )

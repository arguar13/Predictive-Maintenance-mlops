"""Entrenamiento del ConvTransformer de mantenimiento predictivo con trazabilidad MLflow.

Cada ejecución queda ligada, de forma inequívoca, a la tupla:
    Git Commit Hash + DVC Data Hash + Hyperparameters + MLflow Run ID +
    Container Image Tag
a través de tags/params del run de MLflow (ver `_build_lineage_tags`).

Carga de datos: lee `engine_features.parquet` directamente del disco (ver
`prepare_training_data.py`, que lo genera a partir de los .txt crudos de
C-MAPSS). Mismo camino para el dataset toy (smoke test local, segundos, sin
GPU) y para el dataset completo (entrenamiento real).

El modelo y el scaler NUNCA quedan como archivos sueltos: ambos se loguean
como artefactos del mismo run de MLflow. Solo se REGISTRA una version en el
Model Registry, y se le mueve el alias "champion" (el que sirve la API), si:
  1. supera el Quality Gate: F2 ponderado Y recall de la clase Critical, cada
     uno contra su propio umbral en `monitoring.*` de config.yaml (ver
     `_evaluate_quality_gate`) - no un accuracy plano, ciego al costo
     asimetrico de confundir un motor "Critical" con "Healthy"/"Alert"; y
  2. no es peor que el champion actual cuando ambos se evaluaron sobre los
     mismos datos y el mismo split (ver `_should_replace_champion`).
El gate se decide sobre un conjunto de TEST de motores que no participan ni
en el entrenamiento ni en la eleccion del checkpoint (esa usa VAL), para que
las metricas que certifican el modelo no esten sesgadas por la seleccion.
Un modelo rechazado queda solo como run (auditable), sin ensuciar el
Registry con versiones que nunca deben servirse.
"""

from __future__ import annotations

import argparse
import hashlib
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
from mlflow.exceptions import MlflowException
from sklearn.metrics import fbeta_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from config_loader import load_config
from data_contracts import validate_training_batch
from logging_config import configure_logging

log = configure_logging("train")

BASE_DIR = Path(__file__).resolve().parent.parent
# Raiz del repositorio (padre de core_ml/): ahi vive el .git del proyecto.
REPO_ROOT = BASE_DIR.parent
NUM_CLASSES = 3
CHAMPION_ALIAS = "champion"
EXPERIMENT_NAME = "Predictive_Maintenance_FCN"
# 0=Healthy, 1=Alert, 2=Critical (ver data_processing.py::build_multiclass_target).
# El costo de un falso negativo aqui (decir Healthy/Alert de un motor que en
# realidad esta Critical) es el que justifica todo el quality gate de abajo.
CRITICAL_CLASS_INDEX = 2


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
    de preparación de datos de este proyecto solo produce `windowed_features`,
    sin esas columnas adicionales.
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
    ci_sha = os.environ.get("CI_COMMIT_SHA") or os.environ.get("GITHUB_SHA")
    if ci_sha:
        return ci_sha
    try:
        # Lista de argv fija (sin input de usuario) y shell=False (por
        # defecto): bandit igual marca subprocess/ruta-parcial por defecto,
        # pero aquí no hay superficie de inyección de comandos.
        # cwd=REPO_ROOT (no core_ml/): si core_ml/ contiene un .git anidado
        # (p.ej. restos de un clon previo), `git` resolveria ESE repo y el
        # tag de linaje apuntaria a un commit que no es el del proyecto.
        result = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
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


def _build_lineage_tags(data_dir: str) -> dict[str, str]:
    return {
        "git_commit_sha": _get_git_commit_sha(),
        "dvc_data_hash": _get_dvc_data_hash(data_dir),
        "dvc_data_dir": data_dir,
        "container_image_tag": _get_container_image_tag(),
    }


def _training_data_fingerprint(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_split: float,
    test_split: float,
    seed: int,
) -> str:
    """Huella del contenido EXACTO con el que se entrena y valida.

    `dvc_data_hash` solo refleja el puntero `.dvc` commiteado, no lo que hay
    realmente en disco (un `data/` modificado localmente seguiria reportando
    el mismo hash) ni el submuestreo de prepare_training_data.py. Esta huella
    se calcula sobre los arrays reales + los parametros del split, y es lo que
    permite decidir si dos runs son comparables metrica a metrica.
    """
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(X, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(y, dtype=np.int64).tobytes())
    digest.update("|".join(map(str, groups)).encode())
    digest.update(f"val_split={val_split};test_split={test_split};seed={seed}".encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Carga de datos de entrenamiento
# ---------------------------------------------------------------------------


def _load_training_data(data_path: Path) -> pd.DataFrame:
    parquet_path = data_path / "engine_features.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"No se encontró {parquet_path}. Ejecuta antes:\n"
            f"  poetry run python src/prepare_training_data.py --data-dir {data_path.name}"
        )
    return pd.read_parquet(parquet_path)


# ---------------------------------------------------------------------------
# Pipeline de entrenamiento
# ---------------------------------------------------------------------------


def _group_split_indices(
    indices: np.ndarray, groups: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Parte `indices` en dos sin separar nunca las ventanas de un mismo motor."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    first, second = next(splitter.split(indices, groups=groups[indices]))
    return indices[first], indices[second]


def _split_train_val_test(
    groups: np.ndarray, val_split: float, test_split: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Indices de train / val / test, agrupados por motor (ver _split_by_engine).

    - train: ajusta los pesos.
    - val: elige el checkpoint y decide el early stopping.
    - test: SOLO decide el quality gate y la comparacion con el champion.

    Si el gate se evaluara sobre val, las metricas serian optimistas: val ya
    se uso para elegir el mejor epoch entre muchos, asi que ese epoch esta
    "ajustado" a val. Con un test de motores que nunca influyen en ninguna
    decision de entrenamiento, el gate mide generalizacion real.
    `val_split` y `test_split` son fracciones de MOTORES sobre el total.
    """
    if not (0 < val_split < 1 and 0 < test_split < 1 and val_split + test_split < 1):
        raise ValueError(
            f"val_split ({val_split}) y test_split ({test_split}) deben estar en (0, 1) "
            "y sumar menos de 1."
        )
    all_indices = np.arange(len(groups))
    n_engines = len(np.unique(groups))
    if n_engines < 3:
        # Mismo criterio que _split_by_engine: solo el smoke test con un
        # dataset diminuto llega aqui; sin 3 motores no hay tres particiones.
        log.warning("too_few_engines_for_three_way_split", n_engines=n_engines)
        return all_indices, all_indices, all_indices

    rest, test = _group_split_indices(all_indices, groups, test_split, seed)
    # val_split es fraccion del total: se reescala al resto que queda tras test.
    train, val = _group_split_indices(rest, groups, val_split / (1 - test_split), seed)
    return train, val, test


def _evaluate(model: nn.Module, X: torch.Tensor, y: np.ndarray) -> dict[str, float]:
    """Accuracy, F2 ponderado y recall de Critical del modelo sobre (X, y)."""
    model.eval()
    with torch.no_grad():
        predictions = torch.argmax(model(X), dim=1).numpy()
    return {
        "accuracy": float((predictions == y).mean()),
        "f2_weighted": float(
            fbeta_score(y, predictions, beta=2, average="weighted", zero_division=0)
        ),
        "critical_recall": float(
            recall_score(
                y, predictions, average=None, labels=np.arange(NUM_CLASSES), zero_division=0
            )[CRITICAL_CLASS_INDEX]
        ),
    }


def _split_by_engine(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/val split que nunca parte un motor entre los dos lados.

    Las ventanas se generan con stride=1 por motor (ver create_sliding_windows
    en data_processing.py), asi que ventanas consecutivas del mismo motor se
    solapan en ~29 de sus 30 timesteps. Un split IID fila a fila (el
    train_test_split que este proyecto usaba antes) deja copias casi
    identicas del mismo motor a ambos lados: el modelo "ve" en validation
    ventanas practicamente iguales a las que acaba de entrenar, lo que infla
    val_accuracy -- y por lo tanto el quality gate -- sin que el modelo
    generalice mejor. GroupShuffleSplit, agrupando por `groups` (engine_id),
    garantiza que todas las ventanas de un motor caen enteras en train o en
    val. No estratifica por clase (GroupShuffleSplit no lo soporta): con
    cientos de motores distintos en el dataset completo el balance de clases
    entre train/val ya sale razonablemente parejo sin forzarlo, y agrupar por
    motor es la prioridad no negociable aqui.
    """
    n_engines = len(np.unique(groups))
    if n_engines < 2:
        # Solo puede pasar contra el dataset toy diminuto del smoke test
        # (--no-enforce-quality-gate): con menos de dos motores distintos no
        # hay forma de reservar val sin partir un motor por la mitad.
        # GroupShuffleSplit fallaria con "el train set quedaria vacio" - usar
        # todo como train y como val sigue probando que el pipeline corre de
        # punta a punta, que es lo unico que el smoke test necesita. El
        # dataset completo tiene cientos de motores; esta rama nunca se
        # activa fuera del smoke test.
        log.warning("too_few_engines_for_group_split", n_engines=n_engines)
        return X, X, y, y

    train_idx, val_idx = _group_split_indices(np.arange(len(X)), groups, val_split, seed)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def _should_replace_champion(
    client: MlflowClient, model_name: str, fingerprint: str, challenger_f2: float
) -> bool:
    """Champion/challenger: no reemplazar un champion mejor por uno peor.

    Pasar el gate no basta: un reentrenamiento puede superar los umbrales y
    aun asi ser peor que el modelo que ya esta sirviendo. Solo se comparan
    metricas si ambos runs tienen la misma huella de datos/split (misma
    validacion); si no son comparables (datos nuevos, otro split), decide
    solo el gate - comparar F2 de validaciones distintas no significa nada.
    """
    try:
        champion = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except MlflowException:
        return True  # Todavia no hay champion.

    if champion.run_id is None:
        return True
    champion_run = client.get_run(champion.run_id)
    if champion_run.data.tags.get("training_data_fingerprint") != fingerprint:
        log.info("champion_not_comparable", champion_version=champion.version)
        return True

    champion_f2 = champion_run.data.metrics.get("test_f2_weighted")
    if champion_f2 is None or challenger_f2 >= champion_f2:
        return True

    log.warning(
        "challenger_worse_than_champion",
        champion_version=champion.version,
        champion_f2_weighted=champion_f2,
        challenger_f2_weighted=challenger_f2,
    )
    return False


def _evaluate_quality_gate(
    f2_weighted: float,
    critical_recall: float,
    f2_threshold: float,
    critical_recall_threshold: float,
) -> bool:
    """Ambos umbrales tienen que pasar, no uno u otro.

    F2 pondera el recall el doble que la precision sobre las 3 clases, asi
    que ya favorece detectar Critical por encima de un accuracy plano. Pero
    un modelo puede tener buen F2 global compensando con buen desempeño en
    Healthy/Alert (las clases mayoritarias) mientras falla sistematicamente
    en Critical -- justo la clase que mas importa -- y seguir pasando un
    umbral de F2 solo. critical_recall_threshold es el piso duro que cierra
    ese hueco: sin el, el gate podria promover exactamente el tipo de
    modelo que este cambio existe para rechazar.
    """
    return f2_weighted >= f2_threshold and critical_recall >= critical_recall_threshold


def _checkpoint_score(
    f2_weighted: float,
    critical_recall: float,
    f2_threshold: float,
    critical_recall_threshold: float,
) -> tuple[bool, float]:
    """Clave de orden para elegir el mejor checkpoint: (pasa el gate, F2).

    Elegir el checkpoint SOLO por F2 dejaba fuera el piso de recall de
    Critical que el gate tambien exige. Observado con datos reales: el epoch
    de mejor F2 (0.826) tenia recall Critical 0.61 y fallaba el gate, mientras
    otro epoch (F2 0.807, recall 0.86) lo pasaba - y se restauraba el que
    fallaba. Con esta clave, cualquier epoch que pasa el gate gana a uno que
    no, y entre iguales decide el F2. No fuerza el gate: si ningun epoch lo
    pasa, se restaura el de mejor F2 y el gate falla igual.
    """
    passes = _evaluate_quality_gate(
        f2_weighted, critical_recall, f2_threshold, critical_recall_threshold
    )
    return passes, f2_weighted


def train_pipeline(
    data_dir: str = "data",
    epochs: int = 3,
    learning_rate: float = 0.001,
    batch_size: int = 64,
    val_split: float = 0.2,
    test_split: float = 0.2,
    enforce_quality_gate: bool = True,
    seed: int = 42,
    patience: int = 5,
) -> str:
    # Reproducibilidad: sin esto, la inicializacion de pesos de ConvTransformer
    # y el orden de batches del DataLoader (shuffle=True) son no-deterministas
    # -- el mismo commit + mismos datos + mismos hiperparametros podia pasar
    # el quality gate en una corrida del pipeline y fallar en la siguiente.
    # Misma semilla que _split_by_engine mas abajo.
    torch.manual_seed(seed)
    config = load_config()
    window_size = config["model"]["window_size"]
    model_name = config["model"]["model_name"]
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", config["model"]["mlflow_tracking_uri"])
    )
    mlflow.set_experiment(EXPERIMENT_NAME)

    data_path = BASE_DIR / data_dir

    # FAIL FAST: sin el scaler del mismo prepare, el modelo resultante no es
    # servible (la API no puede escalar las lecturas crudas). Antes esto era
    # solo un warning al FINAL del entrenamiento, con el modelo ya registrado.
    scaler_path = data_path / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"No se encontró {scaler_path}. Ejecuta antes prepare_training_data.py "
            f"--data-dir {data_dir} (genera parquet y scaler juntos)."
        )

    log.info("loading_training_data", data_path=str(data_path))
    training_data = _load_training_data(data_path)

    if len(training_data) == 0:
        raise ValueError(f"El dataset de entrenamiento en {data_path} está vacío.")

    # FAIL FAST: valida el contrato del batch antes de construir tensores y
    # de gastar cómputo de entrenamiento (ver data_contracts.py).
    sample_array_length = len(training_data["windowed_features"].iloc[0])
    training_data = validate_training_batch(training_data, expected_length=sample_array_length)

    if sample_array_length % window_size != 0:
        raise ValueError(
            f"windowed_features tiene longitud {sample_array_length}, que no es multiplo de "
            f"window_size={window_size}: el parquet se genero con otra config."
        )
    num_features = sample_array_length // window_size
    log.info(
        "features_detected", num_features=num_features, sample_array_length=sample_array_length
    )

    X = np.stack(
        [
            np.asarray(value, dtype=np.float32).reshape(window_size, num_features)
            for value in training_data["windowed_features"]
        ]
    )
    y = training_data["failure_type"].to_numpy()
    groups = training_data["engine_id"].to_numpy()
    fingerprint = _training_data_fingerprint(X, y, groups, val_split, test_split, seed)

    train_idx, val_idx, test_idx = _split_train_val_test(groups, val_split, test_split, seed)
    X_train, y_train = X[train_idx], y[train_idx]
    y_val, y_test = y[val_idx], y[test_idx]
    log.info(
        "engine_split",
        train_engines=len(np.unique(groups[train_idx])),
        val_engines=len(np.unique(groups[val_idx])),
        test_engines=len(np.unique(groups[test_idx])),
    )

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X[val_idx], dtype=torch.float32)
    X_test_tensor = torch.tensor(X[test_idx], dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True
    )

    model = ConvTransformer(num_features=num_features, num_classes=NUM_CLASSES)
    # Pesos de clase "balanced" (inversos a la frecuencia en y_train, no un
    # valor fijo adivinado): RUL esta dominado por Healthy, y sin ponderar
    # la loss el gradiente apenas "ve" Critical durante el entrenamiento -
    # el modelo puede converger a un minimo que ignora la clase que mas
    # importa y aun asi reportar accuracy alto. Se calcula sobre y_train
    # (nunca sobre y_val) para no filtrar informacion de validation al
    # entrenamiento.
    # Si alguna clase no aparece en y_train (dataset pequeño o muy
    # submuestreado), compute_class_weight con las 3 clases lanza un
    # ValueError críptico: se calculan pesos solo para las presentes y se
    # avisa, en vez de abortar.
    present_classes = np.unique(y_train)
    class_weights = np.ones(NUM_CLASSES)
    class_weights[present_classes] = compute_class_weight(
        class_weight="balanced", classes=present_classes, y=y_train
    )
    if len(present_classes) < NUM_CLASSES:
        log.warning("classes_missing_in_train_split", present=present_classes.tolist())
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lineage_tags = _build_lineage_tags(data_dir)
    f2_threshold = config["monitoring"]["f2_weighted_threshold"]
    critical_recall_threshold = config["monitoring"]["critical_recall_threshold"]

    with mlflow.start_run(run_name="ConvTransformer_Training") as run:
        mlflow.set_tags({**lineage_tags, "training_data_fingerprint": fingerprint})
        mlflow.log_params(
            {
                "window_size": window_size,
                "num_features": num_features,
                "epochs": epochs,
                "patience": patience,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "val_split": val_split,
                "test_split": test_split,
                "seed": seed,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "test_samples": len(test_idx),
            }
        )

        # Selecciona el mejor checkpoint con los MISMOS criterios del quality
        # gate (ver _checkpoint_score: primero que lo pase, luego F2), no por
        # val_accuracy ni por
        # asumir que el ultimo epoch es el mejor: con el dataset completo
        # (mucho mas solapamiento entre ventanas deslizantes que en el toy
        # dataset) se observo val_accuracy colapsando por overfitting bien
        # entrado el entrenamiento (epoch 20: train_loss bajando a 0.21 pero
        # accuracy cayendo a 0.18, peor que adivinar al azar) mientras una
        # epoch intermedia generalizaba mejor. Elegir el checkpoint por una
        # metrica y gatear la promocion por otra distinta abriria la puerta
        # a promover el mejor-por-accuracy aunque no sea el mejor-por-F2 que
        # el gate en realidad exige. Es el equivalente a early stopping
        # sobre el mejor checkpoint, no una forma de forzar el quality gate.
        # Early stopping (patience): "epochs" es un TECHO, no un objetivo --
        # con ConvTransformer casi siempre converge mucho antes. Cortar en
        # cuanto `patience` epochs seguidos no mejoran el F2 ahorra computo
        # real sin arriesgar nada (el checkpoint restaurado sigue siendo
        # siempre el mejor, nunca el ultimo). La paciencia mide F2 a secas y
        # NO la clave del checkpoint: si midiera la clave, un epoch que pasa el
        # gate temprano congelaria el contador y cortaria el entrenamiento
        # antes de llegar a epochs posteriores mejores en ambos criterios.
        best_score: tuple[bool, float] | None = None
        best_f2_seen = -1.0
        best_f2_weighted = -1.0
        best_val_accuracy = -1.0
        best_critical_recall = -1.0
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

            val_metrics = _evaluate(model, X_val_tensor, y_val)
            epoch_val_accuracy = val_metrics["accuracy"]
            epoch_f2_weighted = val_metrics["f2_weighted"]
            epoch_critical_recall = val_metrics["critical_recall"]
            mlflow.log_metric("val_accuracy", epoch_val_accuracy, step=epoch)
            mlflow.log_metric("val_f2_weighted", epoch_f2_weighted, step=epoch)
            mlflow.log_metric("val_critical_recall", epoch_critical_recall, step=epoch)

            log.info(
                "epoch_completed",
                epoch=epoch + 1,
                total_epochs=epochs,
                train_loss=avg_loss,
                val_accuracy=epoch_val_accuracy,
                val_f2_weighted=epoch_f2_weighted,
                val_critical_recall=epoch_critical_recall,
            )

            epoch_score = _checkpoint_score(
                epoch_f2_weighted, epoch_critical_recall, f2_threshold, critical_recall_threshold
            )
            if best_score is None or epoch_score > best_score:
                best_score = epoch_score
                best_f2_weighted = epoch_f2_weighted
                best_val_accuracy = epoch_val_accuracy
                best_critical_recall = epoch_critical_recall
                best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

            if epoch_f2_weighted > best_f2_seen:
                best_f2_seen = epoch_f2_weighted
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    log.info(
                        "early_stopping",
                        epoch=epoch + 1,
                        patience=patience,
                        best_f2_weighted=best_f2_weighted,
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
        # Metricas finales del checkpoint restaurado. Las de val se reportan
        # para diagnostico; el gate usa SOLO las de test (motores no vistos).
        test_metrics = _evaluate(model, X_test_tensor, y_test)
        test_accuracy = test_metrics["accuracy"]
        f2_weighted = test_metrics["f2_weighted"]
        critical_recall = test_metrics["critical_recall"]
        mlflow.log_metrics(
            {
                "best_val_accuracy": best_val_accuracy,
                "best_val_f2_weighted": best_f2_weighted,
                "best_val_critical_recall": best_critical_recall,
                "test_accuracy": test_accuracy,
                "test_f2_weighted": f2_weighted,
                "test_critical_recall": critical_recall,
            }
        )

        log.info(
            "evaluation_completed",
            val_f2_weighted=best_f2_weighted,
            val_critical_recall=best_critical_recall,
            test_accuracy=test_accuracy,
            test_f2_weighted=f2_weighted,
            test_critical_recall=critical_recall,
            f2_threshold=f2_threshold,
            critical_recall_threshold=critical_recall_threshold,
        )

        example_input = X_train_tensor[:1]
        signature = mlflow.models.infer_signature(
            example_input.numpy(), model(example_input).detach().numpy()
        )
        # Se loguea SIN registered_model_name: registrar es una decision
        # posterior al gate (ver mas abajo), no un efecto de loguear.
        model_info = mlflow.pytorch.log_model(
            model,
            name="model",
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
        mlflow.log_artifact(str(scaler_path), artifact_path="preprocessing")

        quality_gate_passed = _evaluate_quality_gate(
            f2_weighted, critical_recall, f2_threshold, critical_recall_threshold
        )
        mlflow.set_tag("quality_gate_passed", str(quality_gate_passed))
        mlflow.set_tag("quality_gate_f2_threshold", str(f2_threshold))
        mlflow.set_tag("quality_gate_critical_recall_threshold", str(critical_recall_threshold))

        run_id = run.info.run_id
        log.info("lineage_tuple", mlflow_run_id=run_id, **lineage_tags)

        client = MlflowClient()
        if quality_gate_passed and _should_replace_champion(
            client, model_name, fingerprint, f2_weighted
        ):
            registered = mlflow.register_model(model_info.model_uri, model_name)
            client.set_registered_model_alias(
                name=model_name, alias=CHAMPION_ALIAS, version=str(registered.version)
            )
            mlflow.set_tag("promoted_version", str(registered.version))
            log.info(
                "quality_gate_passed",
                test_accuracy=test_accuracy,
                test_f2_weighted=f2_weighted,
                critical_recall=critical_recall,
                promoted_version=registered.version,
                alias=CHAMPION_ALIAS,
            )
        elif quality_gate_passed:
            mlflow.set_tag("promoted_version", "none: el champion actual es mejor")
        else:
            log.warning(
                "quality_gate_failed",
                test_accuracy=test_accuracy,
                test_f2_weighted=f2_weighted,
                critical_recall=critical_recall,
                f2_threshold=f2_threshold,
                critical_recall_threshold=critical_recall_threshold,
            )

    if enforce_quality_gate and not quality_gate_passed:
        raise SystemExit(
            f"Quality gate fallido (test): f2_weighted={f2_weighted:.4f} "
            f"(umbral={f2_threshold}), critical_recall={critical_recall:.4f} "
            f"(umbral={critical_recall_threshold}). "
            "No se avanza (revisa datos/hiperparámetros)."
        )

    return run_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping: corta el entrenamiento tras N epochs seguidos sin mejorar "
        "val_f2_weighted (siempre se restaura el mejor checkpoint, nunca el ultimo).",
    )
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraccion de motores para validacion (eleccion de checkpoint).",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraccion de motores de test: solo deciden el quality gate.",
    )
    parser.add_argument("--seed", type=int, default=42)
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
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        enforce_quality_gate=args.enforce_quality_gate,
        seed=args.seed,
    )

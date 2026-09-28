"""Genera el dataset "toy" (~1000 filas) versionado con DVC.

Extrae un subconjunto FIJO y determinista de train_FD001.txt (las primeras
`TOY_UNITS` unidades de motor, ~1000 ciclos en total) copiando las líneas
crudas tal cual, sin re-parsear ni reformatear los valores, para que el toy
dataset sea una muestra real y bit-a-bit fiel de los datos de producción.

Uso:
    poetry -C core_ml run python scripts/build_toy_dataset.py

Tras regenerarlo, versiona el resultado con:
    dvc add data_toy
    git add data_toy.dvc
    dvc push
"""

from pathlib import Path

TOY_UNITS = {1, 2, 3, 4, 5}
SOURCE_DATASET = "FD001"

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_FILE = BASE_DIR / "data" / f"train_{SOURCE_DATASET}.txt"
TARGET_DIR = BASE_DIR / "data_toy"
TARGET_FILE = TARGET_DIR / f"train_{SOURCE_DATASET}.txt"


def build_toy_dataset() -> None:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"No se encontro el dataset fuente en {SOURCE_FILE}. "
            "Ejecuta 'dvc pull' en core_ml/ para descargar los datos crudos."
        )

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    kept_lines = []
    with open(SOURCE_FILE, encoding="utf-8") as source:
        for line in source:
            stripped = line.strip()
            if not stripped:
                continue
            unit_number = int(stripped.split()[0])
            if unit_number in TOY_UNITS:
                kept_lines.append(line)

    if not kept_lines:
        raise RuntimeError("El filtro de unidades no produjo ninguna fila; revisa TOY_UNITS.")

    with open(TARGET_FILE, "w", encoding="utf-8") as target:
        target.writelines(kept_lines)

    print(
        f"Toy dataset generado en {TARGET_FILE}: {len(kept_lines)} filas "
        f"(unidades {sorted(TOY_UNITS)} de {SOURCE_DATASET})."
    )


if __name__ == "__main__":
    build_toy_dataset()

FROM python:3.12-slim

# Debe coincidir con la version usada para generar core_ml/poetry.lock (ver
# su cabecera) para garantizar que Docker resuelva exactamente el mismo
# entorno que local y CI.
ENV POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    # Default de Poetry (15s) corta descargas de ruedas ML grandes (dask,
    # feast, pyarrow, mlflow, torch) con ReadTimeoutError en redes lentas.
    POETRY_REQUESTS_TIMEOUT=600

# --no-install-recommends (DS-0029): evita arrastrar paquetes sugeridos que
# no se usan (mas peso de imagen y mas CVEs que triar en cada escaneo).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel "setuptools<81"

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

# Se conserva el mismo anidado core_ml/ que en el repo (en vez de aplanarlo)
# para que config_loader.py resuelva config/config.yaml con la misma lógica
# relativa que usa en local (sube 3 niveles desde core_ml/src/).
COPY core_ml/pyproject.toml core_ml/poetry.lock* ./core_ml/
# DS-0013: `WORKDIR` en vez de `RUN cd ...`; el cambio de directorio queda
# en los metadatos de la capa y no depende de la shell.
WORKDIR /app/core_ml
RUN poetry install --only main --no-interaction --no-ansi --no-root
WORKDIR /app

COPY core_ml/src/ ./core_ml/src/
COPY core_ml/streaming/ ./core_ml/streaming/
COPY core_ml/feature_store/ ./core_ml/feature_store/
COPY config/ ./config/

WORKDIR /app/core_ml
# src/ en PYTHONPATH: kafka_consumer.py importa config_loader/data_contracts
# como módulos sueltos (mismo estilo que el resto de scripts de core_ml/).
ENV PYTHONPATH=src

# DS-0002: no ejecutar como root. El consumer solo lee su codigo y escribe
# en /tmp (cache de Feast y descargas de artefactos de MLflow).
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

CMD ["poetry", "run", "python", "streaming/kafka_consumer.py"]

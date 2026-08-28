FROM python:3.12-slim

# Debe coincidir con la version usada para generar core_ml/poetry.lock (ver
# su cabecera): este servicio reutiliza las dependencias de core_ml/
# (evidently, kafka-python-ng, prometheus-client, requests) en vez de
# duplicar un pyproject.toml propio para un unico script.
ENV POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    # Default de Poetry (15s) corta descargas de ruedas ML grandes (dask,
    # feast, pyarrow, mlflow) con ReadTimeoutError en redes lentas/domesticas.
    POETRY_REQUESTS_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel "setuptools<81"

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

# Mismo anidado core_ml/ que en el repo: evidently_service.py resuelve
# config_loader/logging_config sumando "../core_ml/src" a sys.path desde su
# propia ubicacion (monitoring/evidently_service.py).
COPY core_ml/pyproject.toml core_ml/poetry.lock* ./core_ml/
WORKDIR /app/core_ml
RUN poetry install --only main --no-interaction --no-ansi --no-root
WORKDIR /app

COPY core_ml/src/ ./core_ml/src/
COPY config/ ./config/
COPY monitoring/evidently_service.py ./monitoring/evidently_service.py

WORKDIR /app/core_ml

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/metrics || exit 1

CMD ["poetry", "run", "python", "../monitoring/evidently_service.py"]

FROM python:3.12-slim

# Proyecto Poetry propio (monitoring/pyproject.toml), NO core_ml/: este
# servicio solo necesita evidently/kafka-python-ng/prometheus-client/pandas/
# requests + config_loader/logging_config, nunca torch/feast/dvc/mlflow/
# sklearn/testcontainers que core_ml/ arrastra para entrenamiento. Reusar
# todo core_ml/poetry.lock (version anterior de este Dockerfile) triplicaba
# la descarga/build/push de esas dependencias pesadas -- ya las construyen
# por separado el Dockerfile de la API y el del streaming-consumer -- e
# inflaba el tiempo del job build_image de .gitlab-ci.yml sin necesidad.
ENV POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    POETRY_REQUESTS_TIMEOUT=600

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel "setuptools<81"

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY monitoring/pyproject.toml monitoring/poetry.lock* ./monitoring/
WORKDIR /app/monitoring
RUN poetry install --no-interaction --no-ansi
WORKDIR /app

# Solo los modulos que evidently_service.py realmente importa (via
# sys.path.append hacia "../core_ml/src" y "../core_ml/streaming") -- no todo
# core_ml/src|streaming/, que incluye train.py/data_processing.py/
# prepare_feast_data.py/kafka_consumer.py y sus propias dependencias pesadas
# (torch, pandera) sin uso aqui.
COPY core_ml/src/config_loader.py core_ml/src/config_schema.py core_ml/src/logging_config.py ./core_ml/src/
COPY core_ml/streaming/kafka_security.py ./core_ml/streaming/
COPY config/ ./config/
COPY monitoring/evidently_service.py ./monitoring/evidently_service.py

WORKDIR /app/monitoring

RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/metrics || exit 1

CMD ["poetry", "run", "python", "evidently_service.py"]

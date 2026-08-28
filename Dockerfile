FROM python:3.12-slim

# Debe coincidir con la version usada para generar api/poetry.lock (ver su
# cabecera) para garantizar que Docker resuelva exactamente el mismo
# entorno que local y CI.
ENV POETRY_VERSION=2.4.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1 \
    # Default de Poetry (15s) corta descargas de ruedas grandes (mlflow,
    # matplotlib, scikit-learn) con ReadTimeoutError en redes lentas.
    POETRY_REQUESTS_TIMEOUT=600

# --no-install-recommends (DS-0029): sin el, apt arrastra decenas de
# paquetes sugeridos que no se usan -- mas peso de imagen y mas superficie
# de CVEs que reportar en cada escaneo.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel "setuptools<81"

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY api/pyproject.toml api/poetry.lock* ./
RUN poetry install --only main --no-interaction --no-ansi
RUN pip install --force-reinstall "setuptools==80.9.0"
# POETRY_VIRTUALENVS_CREATE=false hace que `poetry install` comparta site-
# packages con la propia instalacion de Poetry (no crea un venv aislado del
# proyecto). api/poetry.lock resuelve "packaging" en 23.2 (suficiente para
# black/pytest/mlflow-skinny/matplotlib/skops), pero Poetry 2.4.1 necesita
# el submodulo packaging.licenses (agregado en 24.2) para su propio CLI --
# "poetry run ..." se rompe con "No module named 'packaging.licenses'" sin
# este force-reinstall posterior al install del proyecto.
RUN pip install --force-reinstall "packaging>=24.2"

COPY api/ ./api/
# config/config.yaml es OBLIGATORIO en runtime: api/config_loader.py lo
# resuelve como <parent-of-api>/config/config.yaml. Sin este COPY la imagen
# arranca y muere en el import de api.main con FileNotFoundError.
COPY config/ ./config/

# api/main.py importa sus modulos hermanos como top-level (`from
# config_loader import load_config`), igual que en local y en los tests
# (pytest pythonpath = ["."] con rootdir api/). Sin esto, uvicorn arranca
# desde /app y el import falla con ModuleNotFoundError.
ENV PYTHONPATH=/app/api

# DS-0002: no ejecutar como root. La aplicacion no escribe en disco (modelo
# y scaler se descargan de MLflow a un temporal), asi que basta con que el
# usuario pueda leer /app.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# DS-0026: healthcheck a nivel de imagen, para que cualquier runtime (no
# solo docker-compose) sepa distinguir "proceso vivo" de "servicio listo".
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

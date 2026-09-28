# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Etapa 1 (builder): resuelve api/poetry.lock en un venv aislado.
#
# Poetry vive en SU PROPIO venv (/opt/poetry) y las dependencias de la app en
# otro (/opt/venv). Antes ambos compartian el site-packages del sistema
# (POETRY_VIRTUALENVS_CREATE=false), y el lock de la app pisaba librerias que
# el propio Poetry necesita: de ahi los `pip install --force-reinstall` de
# setuptools/packaging que habia que encadenar para que la imagen arrancara.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Debe coincidir con la version usada para generar api/poetry.lock (ver su
# cabecera) para garantizar que Docker resuelva exactamente el mismo
# entorno que local y CI.
ENV POETRY_VERSION=2.4.1 \
    # Default de Poetry (15s) corta descargas de ruedas grandes (torch,
    # mlflow) con ReadTimeoutError en redes lentas.
    POETRY_REQUESTS_TIMEOUT=600 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN python -m venv /opt/poetry \
    && /opt/poetry/bin/pip install "poetry==${POETRY_VERSION}" \
    && python -m venv /opt/venv

# Con VIRTUAL_ENV activo, Poetry instala en ese venv en vez de crear uno propio.
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
COPY api/pyproject.toml api/poetry.lock ./
RUN /opt/poetry/bin/poetry install --only main --no-interaction --no-ansi

# ---------------------------------------------------------------------------
# Etapa 2 (runtime): solo el venv resuelto + el codigo. Sin Poetry, sin
# compiladores ni curl: menos peso y menos superficie de CVEs.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:${PATH}" \
    # api/main.py importa sus modulos hermanos como top-level (`from
    # config_loader import load_config`), igual que en local y en los tests
    # (pytest pythonpath = ["."] con rootdir api/).
    PYTHONPATH=/app/api

# DS-0002: no ejecutar como root. UID 10001 = el que fija el securityContext
# de kubernetes/base/api.yaml.
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser api/ ./api/
# config/config.yaml es OBLIGATORIO en runtime: api/config_loader.py lo
# resuelve como <parent-of-api>/config/config.yaml.
COPY --chown=appuser:appuser config/ ./config/

USER appuser
EXPOSE 8000

# DS-0026: healthcheck a nivel de imagen. Con el interprete de Python (la
# imagen runtime no trae curl).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

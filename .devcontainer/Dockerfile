FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Instalar dependencias del sistema y Poetry
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update && apt-get install -y postgresql-client \
    && curl -sSL https://install.python-poetry.org | python3 -

# Instalar Poetry
RUN pip install poetry

ENV PATH="/root/.local/bin:$PATH"
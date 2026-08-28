FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Debe coincidir con la version usada para generar api/poetry.lock y
# core_ml/poetry.lock (ver su cabecera) para que el devcontainer resuelva
# exactamente el mismo entorno que CI y Docker.
ENV POETRY_VERSION=2.4.1

# Instalar dependencias del sistema y make (interfaz unica del Makefile).
# --no-install-recommends (DS-0029): sin el, apt arrastra paquetes sugeridos
# que este contenedor no usa.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client make \
    && rm -rf /var/lib/apt/lists/*

# DS-0002: la imagen base de devcontainers ya trae el usuario no-root
# "vscode"; antes el Dockerfile se quedaba en root e instalaba Poetry en
# /root/.local, de modo que el usuario real del contenedor ni siquiera lo
# tenia en el PATH. Instalarlo COMO vscode arregla las dos cosas a la vez.
USER vscode

# Poetry mediante el instalador oficial (evita instalarlo con pip, que
# puede generar conflictos de dependencias con el propio entorno de Poetry).
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/home/vscode/.local/bin:$PATH"

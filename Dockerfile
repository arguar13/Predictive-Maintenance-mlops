FROM python:3.12-slim

ENV POETRY_VERSION=1.7.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip wheel "setuptools<81"

RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app

COPY api/pyproject.toml api/poetry.lock* ./
RUN poetry install --only main --no-interaction --no-ansi
RUN pip install --force-reinstall "setuptools==80.9.0"

COPY api/ ./api/
COPY models/ ./models/

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
FROM python:3.12-slim

# Instalar dependencias del sistema requeridas para PyTorch y Kafka
RUN apt-get update && apt-get install -y build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Exponer el puerto por defecto (para FastAPI o Prometheus metrics)
EXPOSE 8000

# Por defecto arranca la API, pero docker-compose sobrescribe este comando para el consumidor Kafka y Evidently
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
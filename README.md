# Predictive Maintenance MLOps

Arquitectura completa para detección de anomalías en turbinas (C-MAPSS) usando redes neuronales 1D, mensajería en tiempo real y reentrenamiento automático.

## Requisitos
* Docker y Docker Compose
* DVC (Para control de versiones de datos)

## Cómo iniciar el clúster
1. Ejecuta toda la infraestructura:
   ```bash
   docker-compose up -d --build

Accede a los servicios:

Airflow Web UI: http://localhost:8080 (Trigger automático de reentrenamiento)

MLflow UI: http://localhost:5000 (Registro de modelos PyTorch)

Grafana: http://localhost:3000 (Monitoreo del Drift expuesto por Evidently y Prometheus)

FastAPI: http://localhost:8000/docs (Consultas manuales/Health checks)

Flujo de Trabajo (Drift & Retraining)
Kafka recibe telemetría continua en engine_telemetry.

evidently_service.py intercepta lotes y calcula el Data Drift.

Prometheus extrae las métricas. Si las métricas cruzan el umbral, Airflow (vía DAG model_retraining_pipeline) ejecuta el reentrenamiento (train.py).

Al finalizar, Airflow dispara el webhook hacia GitHub Actions (ci_cd.yml) para verificar y redesplegar el modelo.


**`src/__init__.py`**
*(Archivo vacío `__init__.py` creado en el directorio `src/` para que Python lo reconozca como un módulo válido).*

## Arquitectura
```text
predictive_maintenance_mlops/
├── .github/
│   └── workflows/ci_cd.yml
├── airflow/
│   └── dags/
│       └── retraining_pipeline.py      # DAG de Airflow para reentrenamiento
├── api/
│   └── main.py                         # API FastAPI (Opcional, para consultas manuales)
├── config/
│   └── config.yaml                     # Configuración centralizada
├── data/                               # Datos C-MAPSS locales o DVC
├── monitoring/
│   ├── prometheus.yml
│   └── evidently_service.py            # Servicio que calcula Drift y expone a Prometheus
├── src/
│   ├── __init__.py
│   ├── config_loader.py
│   ├── data_processing.py              # Limpieza y ventanas deslizantes (C-MAPSS)
│   └── train.py                        # FCN Baseline en PyTorch
├── streaming/
│   └── kafka_consumer.py               # Inferencia en tiempo real consumiendo de Kafka
├── docker-compose.yml                  # Infraestructura completa
├── Dockerfile
├── requirements.txt
└── README.md
```


predictive_maintenance_mlops/
├── .github/
│ └── workflows/
│ └── ci_cd.yml
├── .gitignore
├── .dockerignore
├── Dockerfile
├── docker-compose-localstack.yml
├── README.md
├── requirements.txt
├── airflow/
│ └── drags/
│ └── retraining_pipeline.py
├── api/
│ ├── init.py
│ └── main.py
├── config/
│ └── config.yaml
├── data/
├── k8s/
│ ├── 01-config.yaml
│ ├── 02-messaging.yaml
│ ├── 03-mlops.yaml
│ └── 04-monitoring.yaml
├── monitoring/
│ ├── evidently_service.py
│ └── prometheus.yml
├── src/
│ ├── init.py
│ ├── config_loader.py
│ ├── data_processing.py
│ └── train.py
└── streaming/
└── kafka_consumer.py


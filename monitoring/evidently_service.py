import time
import pandas as pd
from prometheus_client import start_http_server, Gauge
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from sklearn.metrics import accuracy_score
from kafka import KafkaConsumer
import json
import logging
import requests
from src.config_loader import load_config

# Cargar Configuración
config = load_config()
drift_threshold = config['monitoring']['drift_threshold']
accuracy_threshold = config['monitoring']['accuracy_threshold'] # Umbral del 85%

# Métricas de Prometheus
DATA_DRIFT_SCORE = Gauge('evidently_data_drift_score', 'Porcentaje de features con Drift')
DATA_DRIFT_DETECTED = Gauge('evidently_data_drift_detected', '1 si hay drift, 0 si no')
CONCEPT_DRIFT_ACCURACY = Gauge('model_accuracy', 'Precisión actual del modelo')

# Cargar dataset de referencia (datos limpios con los que se entrenó)
# En producción, esto se descarga de DVC o S3
reference_data = pd.read_csv("data/reference_data_clean.csv")

# Función para disparar GitHub Actions
def trigger_github_actions_retraining():
    """Hace una petición HTTPS a GitHub Actions para iniciar el reentrenamiento"""
    github_token = "TU_GITHUB_PAT" # Usar variables de entorno en producción
    repo_owner = "TU_USUARIO"
    repo_name = "TU_REPOSITORIO"
    
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/dispatches"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization": f"token {github_token}"
    }
    data = {"event_type": "model_retrained"}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        logging.info("🚀 Petición enviada a GitHub Actions: ¡Iniciando reentrenamiento!")
    except Exception as e:
        logging.error(f"Error al disparar GitHub Actions: {e}")

def detect_concept_drift(y_true, y_pred):
    """Calcula la precisión y dispara el reentrenamiento si baja del umbral"""
    accuracy = accuracy_score(y_true, y_pred)
    CONCEPT_DRIFT_ACCURACY.set(accuracy)
    
    logging.info(f"Accuracy Actual: {accuracy:.2f} | Umbral: {accuracy_threshold}")
    
    if accuracy < accuracy_threshold:
        logging.warning("🚨 Concept Drift Detectado! La precisión cayó. Iniciando reentrenamiento.")
        trigger_github_actions_retraining()

def detect_drift(current_batch: pd.DataFrame):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_batch)
    
    result = report.as_dict()
    drift_share = result['metrics'][0]['result']['drift_share']
    drift_detected = result['metrics'][0]['result']['dataset_drift']
    
    DATA_DRIFT_SCORE.set(drift_share)
    DATA_DRIFT_DETECTED.set(1 if drift_detected else 0)
    
    logging.info(f"Drift Score: {drift_share:.2f} | Drift Detected: {drift_detected}")

    # Condición para disparar la alerta
    if drift_detected:
        logging.warning("🚨 Drift detectado. Iniciando pipeline de reentrenamiento...")
        trigger_github_actions_retraining()

def run_monitoring_service():
    start_http_server(config['monitoring']['prometheus_port'])
    logging.info(f"Métricas en puerto {config['monitoring']['prometheus_port']}")
    
    # Consumidor de telemetría (Data Drift)
    consumer_telemetry = KafkaConsumer(
        config['kafka']['telemetry_topic'],
        bootstrap_servers=[config['kafka']['broker']],
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    # Consumidor de "Ground Truth" (Concept Drift)
    # En la vida real, las etiquetas reales llegan con retraso en otro tópico
    consumer_ground_truth = KafkaConsumer(
        'ground_truth_topic',
        bootstrap_servers=[config['kafka']['broker']],
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    # Lógica simplificada: en hilos separados o un bucle asíncrono, 
    # escucharía ambos tópicos. Cuando el tópico de ground_truth junte
    # un batch de etiquetas reales comparadas con las predicciones, ejecuto:
    # detect_concept_drift(y_true_batch, y_pred_batch)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_monitoring_service()
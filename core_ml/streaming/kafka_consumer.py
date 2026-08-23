import json
import os
import torch
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import mlflow.pytorch
from feast import FeatureStore

os.environ["MLFLOW_TRACKING_URI"] = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def run_streaming_inference():
    model_uri = "models:/Turbofan_FCN/latest"
    model = mlflow.pytorch.load_model(model_uri)
    if hasattr(model, "eval"):
        try:
            model.eval()
        except NotImplementedError:
            pass
    
    # Ruta corregida apuntando a core_ml/feature_store
    feature_store_path = os.path.join(os.path.dirname(__file__), "../feature_store")
    store = FeatureStore(repo_path=feature_store_path)
    
    kafka_broker = os.environ.get('KAFKA_BROKER', 'localhost:9092')
    telemetry_topic = os.environ.get('KAFKA_TELEMETRY_TOPIC', 'engine_telemetry')
    alert_topic = os.environ.get('KAFKA_ALERT_TOPIC', 'engine_alerts')

    consumer = KafkaConsumer(
        telemetry_topic,
        bootstrap_servers=[kafka_broker],
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    producer = KafkaProducer(
        bootstrap_servers=[kafka_broker],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    window_size = 30
    for message in consumer:
        engine_id = message.value.get('engine_id')
        
        feature_vector = store.get_online_features(
            features=["engine_sensor_window_features:windowed_features"],
            entity_rows=[{"engine_id": engine_id}]
        ).to_dict()
        
        if feature_vector["windowed_features"][0] is None:
            continue
            
        raw_features = np.array(feature_vector["windowed_features"][0])
        if raw_features.size % window_size != 0:
            continue
        num_features = raw_features.size // window_size
        window_matrix = raw_features.reshape(window_size, num_features)
        
        input_tensor = torch.tensor(np.array([window_matrix]), dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            prediction = torch.argmax(outputs, dim=1).item()
            
        classes = ["Healthy", "Alert", "Critical"]
        result = {"engine_id": engine_id, "status": classes[prediction]}
        
        producer.send(alert_topic, result)
        producer.flush()
        print(f"Predicción para {engine_id}: {result['status']}", flush=True)

if __name__ == "__main__":
    run_streaming_inference()
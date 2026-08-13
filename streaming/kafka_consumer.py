import json
import torch
import numpy as np
from kafka import KafkaConsumer, KafkaProducer
import mlflow.pytorch
import joblib

import os
os.environ["MLFLOW_TRACKING_URI"] = "http://mlflow:5000"


def run_streaming_inference():
    # Cargar modelo desde MLflow y Scaler
    model_uri = "models:/Turbofan_FCN/latest"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    scaler = joblib.load("models/scaler.joblib")
    
    consumer = KafkaConsumer(
        'engine_telemetry',
        bootstrap_servers=['kafka:9092'],
        auto_offset_reset='latest',
        value_deserializer=lambda x: json.loads(x.decode('utf-8'))
    )
    
    producer = KafkaProducer(
        bootstrap_servers=['kafka:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )

    print("Esperando telemetría de sensores en Kafka...")
    
    for message in consumer:
        data = message.value # Espera una lista 2D de dimensiones (30, num_features)
        
        # Preprocesar
        scaled_data = scaler.transform(data['sensor_readings'])
        input_tensor = torch.tensor(np.array([scaled_data]), dtype=torch.float32)
        
        # Inferencia
        with torch.no_grad():
            outputs = model(input_tensor)
            prediction = torch.argmax(outputs, dim=1).item()
            
        classes = ["Healthy", "Alert", "Critical"]
        result = {
            "engine_id": data['engine_id'],
            "status": classes[prediction]
        }
        
        # Publicar resultado
        producer.send('engine_alerts', result)
        print(f"Engine: {data['engine_id']} | Predict: {classes[prediction]}")

if __name__ == "__main__":
    run_streaming_inference()
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import mlflow.pytorch
import joblib
import numpy as np
from src.config_loader import load_config

app = FastAPI(title="Predictive Maintenance API", version="2.0")
config = load_config()

# Cargar modelo y scaler globalmente
try:
    model_uri = f"models:/{config['model']['model_name']}/latest"
    model = mlflow.pytorch.load_model(model_uri)
    model.eval()
    scaler = joblib.load("models/scaler.joblib")
except Exception as e:
    model = None
    scaler = None

class SensorData(BaseModel):
    engine_id: str
    readings: list[list[float]] # Matriz de 30x14

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado.")
    return {"status": "ok", "service": "online"}

@app.post("/predict")
def predict_manual(data: SensorData):
    if np.array(data.readings).shape != (config['model']['window_size'], config['model']['num_features']):
        raise HTTPException(status_code=400, detail="Formato de ventana incorrecto.")
    
    scaled_data = scaler.transform(data.readings)
    input_tensor = torch.tensor(np.array([scaled_data]), dtype=torch.float32)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = torch.argmax(outputs, dim=1).item()
        
    classes = ["Healthy", "Alert", "Critical"]
    return {"engine_id": data.engine_id, "prediction": classes[prediction]}
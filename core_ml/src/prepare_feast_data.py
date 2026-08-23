import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from data_processing import load_and_combine_data, build_multiclass_target, clean_and_prepare, create_sliding_windows

def generate_feast_parquet():
    print("🚀 Iniciando preprocesamiento de datos crudos...")
    data_dir = "data"
    
    # 1. Ejecutar tu pipeline de data_processing.py
    raw_df = load_and_combine_data(data_dir)
    target_df = build_multiclass_target(raw_df)
    clean_df = clean_and_prepare(target_df)
    
    # Extraer las ventanas tridimensionales
    X, y, scaler = create_sliding_windows(clean_df, window_size=30)
    
    # 2. Formatear los datos para cumplir con el esquema de Feast (features.py)
    print("📦 Formateando para Feast Feature Store...")
    records = []
    
    # Feast requiere timestamps. Simularemos que los datos llegaron en las últimas 24 horas 
    # para que el TTL de Feast los acepte en el Online Store.
    base_time = datetime.utcnow() - timedelta(hours=23) 
    
    # Para no saturar tu RAM local, procesaremos solo un subconjunto (ej. las primeras 5000 ventanas)
    # En producción, esto se hace con PySpark
    limit = min(len(X), 5000) 
    
    for i in range(limit):
        # Feast espera un Array(Float32) aplanado
        flat_features = X[i].flatten().tolist() 
        records.append({
            "engine_id": f"ENG_{(i % 100) + 1:03d}", # Simulamos IDs de motor
            "event_timestamp": base_time + timedelta(seconds=i*10),
            "created_timestamp": datetime.utcnow(),
            "windowed_features": flat_features,
            "failure_type": int(y[i])
        })
        
    feast_df = pd.DataFrame(records)
    
# 3. Guardar como Parquet maestro (Para el Offline Store de Feast en S3)
    master_path = "data/engine_features.parquet"
    feast_df.to_parquet(master_path, index=False)
    print(f"✅ Archivo Parquet maestro generado en: {master_path}")

    # 4. Generar el Entity DataFrame (Para que train.py lo consuma)
    print("🧠 Extrayendo Entity DataFrame para el entrenamiento...")
    # Extraemos solo las llaves primarias y el timestamp
    entity_df = feast_df[['engine_id', 'event_timestamp']].copy()
    
    # Práctica Senior: Barajar (shuffle) los eventos para evitar sesgos de orden de llegada en el entrenamiento
    entity_df = entity_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    entity_path = "data/training_entities.parquet"
    entity_df.to_parquet(entity_path, index=False)
    print(f"✅ Entity DataFrame guardado en: {entity_path}")

if __name__ == "__main__":
    generate_feast_parquet()
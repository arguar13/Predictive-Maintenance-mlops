import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import os
import pandas as pd
import numpy as np
from feast import FeatureStore
from pathlib import Path  

class FCNBaseline(nn.Module):
    def __init__(self, num_features, num_classes=3):
        super(FCNBaseline, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=8, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding='same'),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, padding='same'),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return self.classifier(x)

def train_pipeline():
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"))
    mlflow.set_experiment("Predictive_Maintenance_FCN")
    
    # --- 2. LÓGICA SENIOR DE RESOLUCIÓN DE RUTAS ABSOLUTAS ---
    # __file__ es este script. .parent es 'src'. .parent.parent es 'core_ml'
    base_dir = Path(__file__).resolve().parent.parent 
    entity_path = base_dir / "data" / "training_entities.parquet"
    feature_store_path = base_dir / "feature_store"
    
    if not entity_path.exists():
        raise FileNotFoundError(
            f"🚨 No se encontró el Entity DataFrame en {entity_path}. "
            "Debes ejecutar 'prepare_feast_data.py' primero."
        )

    # 1. Integración con Feast: Extracción desde Offline Store
    print("Extrayendo características históricas desde Feast Offline Store...")
    
    # Usando las rutas dinámicas absolutas
    store = FeatureStore(repo_path=str(feature_store_path))
    
    print(f"Cargando entidades desde: {entity_path}")
    entity_df = pd.read_parquet(entity_path)
    
    training_data = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "engine_sensor_window_features:windowed_features",
            "engine_sensor_window_features:failure_type"
        ]
    ).to_df()
    
    # 2. Reconstrucción de Tensores PyTorch (Lógica Senior Dinámica)
    window_size = 30
    
    # 1. Inferir dinámicamente el número de features basándose en los datos entrantes
    # Tomamos el tamaño del primer arreglo devuelto por Feast y lo dividimos por la ventana
    sample_array_length = len(training_data['windowed_features'].iloc[0])
    num_features = sample_array_length // window_size
    
    print(f"📊 Detectados {num_features} features por timestep (Arreglo total: {sample_array_length})")
    
    # 2. Reformatear el array aplanado a (N, 30, num_features)
    X = np.array([np.array(val).reshape(window_size, num_features) for val in training_data['windowed_features']])
    y = training_data['failure_type'].values
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 3. Inicializar el modelo con las dimensiones descubiertas dinámicamente
    model = FCNBaseline(num_features=num_features, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Entrenamiento con Logging
    with mlflow.start_run(run_name="FCN_Feast_Training"):
        mlflow.log_param("window_size", window_size)
        
        epochs = 3
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            avg_loss = total_loss / len(dataloader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"✨ Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

        example_input = X_tensor[:1]
        signature = mlflow.models.infer_signature(example_input.numpy(), model(example_input).detach().numpy())
        mlflow.pytorch.log_model(
            model,
            name="model",
            registered_model_name="Turbofan_FCN",
            input_example=example_input,
            signature=signature
        )

if __name__ == "__main__":
    train_pipeline()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import mlflow
import mlflow.pytorch
import joblib
import os
from src.data_processing import load_and_combine_data, build_multiclass_target, clean_and_prepare, create_sliding_windows

class FCNBaseline(nn.Module):
    """Fully Convolutional Network Baseline para series de tiempo."""
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
        # PyTorch espera formato (Batch, Channels, Length) -> Transponemos
        x = x.transpose(1, 2)
        x = self.conv_block(x)
        x = self.global_avg_pool(x).squeeze(-1)
        return self.classifier(x)

def train_pipeline():
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Predictive_Maintenance_FCN")
    
    # 1. Carga y preprocesamiento
    df_raw = load_and_combine_data("data/")
    df_target = build_multiclass_target(df_raw)
    df_clean = clean_and_prepare(df_target)
    
    window_size = 30
    X, y, scaler = create_sliding_windows(df_clean, window_size=window_size)
    num_features = X.shape[2]
    
    # 2. Tensores PyTorch
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = FCNBaseline(num_features=num_features, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Entrenamiento con Logging
    with mlflow.start_run(run_name="FCN_Baseline_Training"):
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
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.joblib")# Para uso en inferencia

if __name__ == "__main__":
    train_pipeline()
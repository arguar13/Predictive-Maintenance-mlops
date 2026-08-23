import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def load_and_combine_data(data_dir: str) -> pd.DataFrame:
    """Carga y une los datasets FD001 a FD004 de C-MAPSS garantizando identificadores únicos."""
    index_names = ['unit_number', 'time_in_cycles']
    setting_names = ['op_setting_1', 'op_setting_2', 'op_setting_3']
    sensor_names = [f'sensor_{i}' for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names
    
    datasets = ['FD001', 'FD002', 'FD003', 'FD004']
    train_list = []
    
    for ds in datasets:
        file_path = os.path.join(data_dir, f'train_{ds}.txt')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
            df['dataset_id'] = ds
            df['global_unit'] = df['dataset_id'] + '_' + df['unit_number'].astype(str)
            train_list.append(df)
            
    return pd.concat(train_list, ignore_index=True)

def build_multiclass_target(df: pd.DataFrame) -> pd.DataFrame:
    """Transforma el problema en clasificación multiclase (Healthy, Alert, Critical)."""
    df = df.copy()
    max_cycles = df.groupby('global_unit')['time_in_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_in_cycles']
    
    df['failure_type'] = 0 # Healthy
    df.loc[df['RUL'] <= 60, 'failure_type'] = 1 # Alert
    df.loc[df['RUL'] <= 30, 'failure_type'] = 2 # Critical
    return df

def clean_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina duplicados y sensores sin varianza (invariantes)."""
    df = df.drop_duplicates()
    sensor_cols = [col for col in df.columns if 'sensor' in col]
    std_dev = df[sensor_cols].std()
    invariant_sensors = std_dev[std_dev < 1e-6].index.tolist()
    
    df = df.drop(columns=invariant_sensors + ['RUL', 'dataset_id'])
    return df

def create_sliding_windows(df: pd.DataFrame, window_size: int = 30):
    """Genera ventanas tridimensionales (Muestras, Ventana Temporal, Características) para PyTorch."""
    features = [c for c in df.columns if c not in ['unit_number', 'time_in_cycles', 'global_unit', 'failure_type']]
    
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    X, y = [], []
    for unit in df['global_unit'].unique():
        unit_data = df[df['global_unit'] == unit].copy()
        unit_data.reset_index(drop=True, inplace=True)
        
        for i in range(len(unit_data) - window_size + 1):
            X.append(unit_data[features].iloc[i:i + window_size].values)
            # Etiqueta del último ciclo de la ventana
            y.append(unit_data['failure_type'].iloc[i + window_size - 1])
            
    return np.array(X), np.array(y), scaler
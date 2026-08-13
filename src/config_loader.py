import yaml
import os
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    base_dir = Path(__file__).resolve().parent.parent
    full_path = os.path.join(base_dir, config_path)
    
    with open(full_path, "r") as file:
        config = yaml.safe_load(file)
    return config
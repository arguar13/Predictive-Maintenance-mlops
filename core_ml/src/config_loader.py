import yaml
import os
import re
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    base_dir = Path(__file__).resolve().parent.parent
    full_path = os.path.join(base_dir, config_path)
    
    # Expresión regular para encontrar ${VAR}
    path_matcher = re.compile(r'\$\{([^}^{]+)\}')
    
    def path_constructor(loader, node):
        value = node.value
        match = path_matcher.match(value)
        env_var = match.group(1)
        # Retorna la variable de entorno, si no existe, mantiene el valor original o un fallback
        return os.environ.get(env_var, f"localhost:{'9092' if 'KAFKA' in env_var else '5000'}")
        
    yaml.add_implicit_resolver('!path', path_matcher, None, yaml.SafeLoader)
    yaml.add_constructor('!path', path_constructor, yaml.SafeLoader)
    
    with open(full_path, "r") as file:
        config = yaml.safe_load(file)
    return config
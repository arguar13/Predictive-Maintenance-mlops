import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from config_schema import AppConfig

# Expresión regular para encontrar valores del tipo "${VAR}"
_ENV_VAR_PATTERN = re.compile(r"^\$\{([^}^{]+)\}$")


# Valores por defecto para desarrollo local (docker-compose.yml). En
# Kubernetes TODOS llegan inyectados por kubernetes/base/configmap.yaml, asi
# que estos defaults solo se usan al ejecutar en un portatil.
#
# El fallback anterior era `f"localhost:{'9092' if 'KAFKA' in env_var else '5000'}"`,
# que devolvia "localhost:5000" para MLFLOW_TRACKING_URI -- una URI SIN
# esquema, que MLflow rechaza con UnsupportedModelRegistryStoreURIException.
# Consecuencia: `make smoke-test` / `make train-toy` fallaban siempre salvo
# que el operador exportase la variable a mano, pese a que README y guia los
# documentan como comandos autonomos.
_LOCAL_DEFAULTS = {
    "MLFLOW_TRACKING_URI": "http://localhost:5000",
}


def _resolve_env_placeholders(value: Any) -> Any:
    """Sustituye recursivamente placeholders "${VAR}" por variables de entorno.

    Opera sobre los valores ya parseados (no sobre nodos YAML) para que la
    sustitución funcione sin importar si el valor estaba citado o no en el
    YAML de origen: PyYAML nunca aplica resolvers implícitos a scalars
    citados, por lo que un enfoque basado en `add_implicit_resolver` nunca
    llega a dispararse para valores como `mlflow_tracking_uri: "${MLFLOW_TRACKING_URI}"`.
    """
    if isinstance(value, dict):
        return {key: _resolve_env_placeholders(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_resolve_env_placeholders(item) for item in value]
    if isinstance(value, str):
        match = _ENV_VAR_PATTERN.match(value)
        if match is None:
            return value
        env_var = match.group(1)
        resolved = os.environ.get(env_var)
        if resolved is not None:
            return resolved
        if env_var not in _LOCAL_DEFAULTS:
            # FAIL FAST: antes se inventaba un "localhost:5000" para
            # CUALQUIER variable desconocida. Un placeholder mal escrito en
            # config.yaml se resolvia silenciosamente a un valor absurdo en
            # vez de avisar.
            raise ValueError(
                f"La variable de entorno {env_var!r}, referenciada en config.yaml, "
                f"no esta definida y no tiene un valor por defecto de desarrollo. "
                f"Variables con default: {sorted(_LOCAL_DEFAULTS)}."
            )
        return _LOCAL_DEFAULTS[env_var]
    return value


def load_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Carga, resuelve variables de entorno y valida config.yaml.

    FAIL FAST: si el YAML resultante no cumple el contrato de `AppConfig`
    (clave faltante, tipo incorrecto, valor fuera de rango) se lanza
    inmediatamente al arrancar, en vez de propagar un KeyError críptico
    más adelante en medio de una petición o de un entrenamiento.
    """
    # __file__ = core_ml/src/config_loader.py -> sube 3 niveles hasta la
    # raíz del repo, donde vive config/config.yaml (compartido con api/).
    base_dir = Path(__file__).resolve().parent.parent.parent
    full_path = os.path.join(base_dir, config_path)

    with open(full_path) as file:
        raw_config = yaml.safe_load(file)

    resolved_config = _resolve_env_placeholders(raw_config)

    try:
        validated = AppConfig.model_validate(resolved_config)
    except ValidationError as exc:
        raise ValueError(f"config.yaml inválido ({full_path}):\n{exc}") from exc

    return validated.model_dump()

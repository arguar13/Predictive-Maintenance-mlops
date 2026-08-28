"""Configuracion de transporte de los clientes de Kafka (kafka-python).

Por que existe este modulo: `terraform/msk.tf` provisiona el cluster con
`client_broker = "TLS"`, es decir, el trafico entre cliente y broker viaja
cifrado (hallazgo AWS-0073 de trivy: un cluster en PLAINTEXT expone toda la
telemetria y todas las alertas de inferencia a cualquiera con acceso de red
a la VPC). Los clientes tienen que hablar TLS en consecuencia.

Pero el stack local de `docker-compose.yml` levanta un Kafka de desarrollo
en PLAINTEXT, y forzar TLS ahi solo añadiria certificados autofirmados que
nadie valida. Por eso el protocolo se inyecta por entorno
(`KAFKA_SECURITY_PROTOCOL`), igual que `KAFKA_BROKER`:

  * local (docker-compose): sin definir -> "PLAINTEXT"
  * EKS (kubernetes/base/configmap.yaml): "SSL"

Con "SSL", kafka-python construye un `ssl.create_default_context()` que usa
el almacen de CAs del sistema. Los brokers de MSK presentan certificados
emitidos por Amazon Trust Services, que ya esta en ese almacen: no hace
falta distribuir ningun truststore propio.
"""

from __future__ import annotations

import os
from typing import Any

_VALID_PROTOCOLS = {"PLAINTEXT", "SSL", "SASL_PLAINTEXT", "SASL_SSL"}


def kafka_client_kwargs() -> dict[str, Any]:
    """kwargs de transporte comunes a KafkaConsumer y KafkaProducer."""
    protocol = os.environ.get("KAFKA_SECURITY_PROTOCOL", "PLAINTEXT").strip().upper()

    if protocol not in _VALID_PROTOCOLS:
        # FAIL FAST: un valor mal escrito en el ConfigMap (p.ej. "TLS" en vez
        # de "SSL") degradaria silenciosamente a texto plano contra un broker
        # que solo acepta TLS, y el sintoma seria un timeout de conexion sin
        # relacion aparente con la causa.
        raise ValueError(
            f"KAFKA_SECURITY_PROTOCOL invalido: {protocol!r}. "
            f"Valores admitidos: {sorted(_VALID_PROTOCOLS)}. "
            "Para Amazon MSK con encryption_in_transit.client_broker = TLS, usa 'SSL'."
        )

    return {"security_protocol": protocol}

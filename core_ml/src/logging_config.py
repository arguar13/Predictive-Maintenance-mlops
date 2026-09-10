"""Logging estructurado (JSON) apto para CloudWatch Logs / Elasticsearch.

Sustituye los `print()` dispersos por logs JSON con timestamp, nivel,
logger, evento y contexto adicional (kwargs) — indexable/filtrable
directamente en CloudWatch Logs Insights o Kibana, sin parsers ad hoc.
"""

from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(service_name: str) -> structlog.stdlib.BoundLogger:
    """Configura structlog para emitir un objeto JSON por línea a stdout.

    `LOG_LEVEL` (env var, por defecto "INFO") controla el nivel mínimo.
    Se llama una vez al arrancar el proceso (api/main.py, src/train.py).
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Los logs de librerias de terceros (mlflow, botocore, urllib3...) usan
    # logging estandar: este formatter los emite como el mismo JSON, en vez
    # de mezclar texto plano con lineas JSON en la misma salida.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(level)

    return structlog.get_logger(service_name)

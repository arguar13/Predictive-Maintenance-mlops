"""Crea los topics de Kafka del proyecto (idempotente) si no existen.

Por que existe este script: `terraform/msk.tf` provisiona el CLUSTER de MSK,
pero un topic de Kafka no es un recurso de AWS -- es un recurso del propio
protocolo Kafka, gestionado por el broker, y crearlo requiere conectividad de
red directa al broker. MSK esta correctamente en subnets privadas sin acceso
publico (mismo criterio que RDS/ElastiCache), asi que Terraform corriendo
desde fuera de la VPC no podria alcanzarlo aunque se declarara un
`kafka_topic` (provider Mongey/kafka). Sin este script, ambos topics
("engine_telemetry", "engine_alerts") nunca llegaban a existir contra el MSK
real: streaming-consumer y evidently_service se conectaban sin error, pero
KafkaProducer.send() se quedaba colgado indefinidamente en
"_wait_on_metadata" (MetadataResponse error_code=3: UNKNOWN_TOPIC_OR_PARTITION),
sin ningun mensaje jamas llegando a ninguno de los dos.

Uso (una sola vez por cluster, desde dentro de la VPC -- p.ej. via
`kubectl exec` en cualquier pod de mlops-env, que ya tiene kafka-python-ng y
la variable KAFKA_BROKER del ConfigMap):
    poetry run python scripts/bootstrap_kafka_topics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "streaming"))
from kafka_security import kafka_client_kwargs  # noqa: E402

from config_loader import load_config  # noqa: E402
from logging_config import configure_logging  # noqa: E402

log = configure_logging("bootstrap-kafka-topics")


def bootstrap_topics(num_partitions: int = 1, replication_factor: int = 2) -> None:
    config = load_config()
    kafka_broker = config["kafka"]["broker"]
    topics = [config["kafka"]["telemetry_topic"], config["kafka"]["alert_topic"]]

    admin = KafkaAdminClient(
        bootstrap_servers=kafka_broker.split(","),
        api_version=(2, 6, 0),
        **kafka_client_kwargs(),
    )
    try:
        existing = set(admin.list_topics())
        to_create = [
            NewTopic(name=t, num_partitions=num_partitions, replication_factor=replication_factor)
            for t in topics
            if t not in existing
        ]
        if not to_create:
            log.info("topics_already_exist", topics=topics)
            return

        try:
            admin.create_topics(to_create)
            log.info("topics_created", topics=[t.name for t in to_create])
        except TopicAlreadyExistsError:
            # Condicion de carrera benigna (otra ejecucion concurrente ya los
            # creo entre el list_topics() de arriba y este create_topics()).
            log.info("topics_already_exist_race", topics=[t.name for t in to_create])
    finally:
        admin.close()


if __name__ == "__main__":
    bootstrap_topics()

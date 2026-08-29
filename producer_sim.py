# producer_sim.py
#
# Simulador manual de telemetria: sin Dockerfile ni despliegue en
# kubernetes/, se ejecuta a mano (local contra docker-compose, o copiado a
# un pod dentro del cluster para probar contra el MSK real, que no tiene
# acceso publico).
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from kafka import KafkaProducer

sys.path.append(str(Path(__file__).resolve().parent / "core_ml" / "streaming"))
from kafka_security import kafka_client_kwargs  # noqa: E402

topic = "engine_telemetry"
broker = os.environ.get("KAFKA_BROKER", "localhost:9092")
window_size = 30
# 14 (el valor historico de config/config.yaml) ya no coincide con el modelo
# real: el dataset C-MAPSS completo produce 24 columnas de sensor tras el
# preprocesamiento (ver "num_features" en la respuesta real de GET /health).
# Sin este fix, un productor contra el cluster real generaria ventanas con
# la forma equivocada, y el streaming-consumer las rechazaria en su propio
# contrato de datos.
num_features = int(os.environ.get("NUM_FEATURES", "24"))

producer = KafkaProducer(
    # KAFKA_BROKER puede traer varios brokers separados por coma (MSK, ver la
    # nota equivalente en core_ml/streaming/kafka_consumer.py): bootstrap_servers=[broker]
    # (sin split) le pasaba a kafka-python un unico host:puerto con una coma
    # en medio, y KafkaProducer.__init__ fallaba con
    # "ValueError: invalid literal for int() with base 10" al intentar
    # parsear el puerto.
    bootstrap_servers=broker.split(","),
    # Sin esto, KafkaProducer intenta auto-detectar la version del broker
    # antes de poder enviar el primer mensaje; contra MSK con SSL eso se
    # colgaba indefinidamente en el productor (a diferencia del consumer,
    # que si logra el auto-detect). (2, 6, 0): misma version que
    # kafka-python ya detecta para este cluster en el streaming-consumer
    # real (ver sus logs: "Broker version identified as 2.6.0").
    api_version=(2, 6, 0),
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    # Mismo modulo que ya usan el streaming-consumer real y evidently_service.py:
    # sin esto, contra Amazon MSK (encryption_in_transit.client_broker = "TLS")
    # el productor falla con kafka.errors.UnrecognizedBrokerVersion.
    **kafka_client_kwargs(),
)

print(f"Simulando telemetría hacia {topic} en {broker}...")
engine_id_counter = 1

try:
    while True:
        dummy_readings = np.random.rand(window_size, num_features).tolist()
        data = {
            "engine_id": f"ENG_{engine_id_counter:03d}",
            "sensor_readings": dummy_readings,
        }
        producer.send(topic, data)
        print(f"Enviada telemetría del motor {data['engine_id']}")
        engine_id_counter += 1
        time.sleep(5)
except KeyboardInterrupt:
    print("Simulación detenida.")
finally:
    producer.close()

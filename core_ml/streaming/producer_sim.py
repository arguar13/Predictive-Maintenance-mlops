import json
import time

import numpy as np
from kafka import KafkaProducer


def run_simulation():
    topic = "engine_telemetry"
    broker = "localhost:9092"
    window_size = 30
    num_features = 24

    producer = KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )

    print(f"Simulando telemetría hacia {topic}...", flush=True)
    engine_id_counter = 1

    try:
        while True:
            readings = np.random.rand(window_size, num_features).tolist()
            data = {
                "engine_id": f"ENG_{engine_id_counter:03d}",
                "sensor_readings": readings,
            }
            producer.send(topic, data).get(timeout=10)
            print(f"Enviada telemetría del motor {data['engine_id']}", flush=True)
            engine_id_counter += 1
            time.sleep(5)
    except KeyboardInterrupt:
        print("Simulación detenida.", flush=True)
    finally:
        producer.close()


if __name__ == "__main__":
    run_simulation()
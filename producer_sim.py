# producer_sim.py
import json
import time
import numpy as np
from kafka import KafkaProducer

topic = "engine_telemetry"
broker = "localhost:9092" 
window_size = 30
num_features = 14

producer = KafkaProducer(
    bootstrap_servers=[broker],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

print(f"Simulando telemetría hacia {topic}...")
engine_id_counter = 1

try:
    while True:
        dummy_readings = np.random.rand(window_size, num_features).tolist()
        data = {
            "engine_id": f"ENG_{engine_id_counter:03d}",
            "sensor_readings": dummy_readings
        }
        producer.send(topic, data)
        print(f"Enviada telemetría del motor {data['engine_id']}")
        engine_id_counter += 1
        time.sleep(5)
except KeyboardInterrupt:
    print("Simulación detenida.")
finally:
    producer.close()
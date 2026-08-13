import json
import time
import numpy as np
from kafka import KafkaProducer

# Configuración basada en tu config.yaml
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
        # Generar datos aleatorios simulando los 14 sensores durante 30 ciclos
        dummy_readings = np.random.rand(window_size, num_features).tolist()
        
        data = {
            "engine_id": f"ENG_{engine_id_counter:03d}",
            "sensor_readings": dummy_readings
        }
        
        producer.send(topic, data)
        print(f"Enviada telemetría del motor {data['engine_id']}")
        
        engine_id_counter += 1
        time.sleep(5) # Enviar un nuevo lote cada 5 segundos
except KeyboardInterrupt:
    print("Simulación detenida.")
finally:
    producer.close()
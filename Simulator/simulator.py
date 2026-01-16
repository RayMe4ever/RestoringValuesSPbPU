import time
import random
import requests
import os
from datetime import datetime
import pandas as pd

# Конфигурация
SERVER_URL = "http://127.0.0.1:8001/data"  # ← поменяйте при необходимости
INTERVAL = 5.0  # примерно как было в websocket-варианте

PORTS = [8092, 8093, 8094, 8095]

# Вероятность пропуска значения для "плохих" портов
files = ["PowerConsumption1.csv", "energydata_complete.csv"]
chances = [0.30, 0.20]
file_readers = []

# Порты без пропусков
CLEAN_PORTS = {8093, 8095}

class file_reader:
    def __init__(self, file_name, chance, ports):
        """Считать данные из .csv файла"""
        csv_path = os.path.join(os.path.dirname(__file__), file_name)
        data = pd.read_csv(csv_path).dropna()
        self.chance = chance
        self.points = data.values
        self.columns = data.columns[1:]
        self.row_min = self.row_cur = 0
        self.row_max = data.iloc[:, 1].size - 5
        self.time_format='%Y-%m-%d %H:%M:%S'
        self.ports = ports


    def parse_timestamp(self, timestamp):
        """Привести временную метку к единому формату"""
        return pd.to_datetime(timestamp).strftime(self.time_format)

    def generate_values(self):
        """Генерация значений примерно как в оригинале"""
        res_clean = {  # Формирование пакета данных
            'names': self.columns.tolist(),
            'values': self.points[self.row_cur, 1:].tolist(),
        }

        points_out = []
        for i in range(1, self.points.shape[1]):
            if random.random() <= self.chance:
                points_out.append(-400000)
            else:
                points_out.append(self.points[self.row_cur, i])

        res = {
            'names': self.columns.tolist(),
            'values': points_out,
        }

        return res, res_clean

    def next_step(self):
        self.row_cur += 1
        if self.row_cur >= self.row_max:
            self.row_cur = 0

def main():
    #print("REST-simulator started")
    #print(f"Sending to: {SERVER_URL}")
    #print(f"Ports: {PORTS}")
    #print(f"Clean ports (no misses): {sorted(CLEAN_PORTS)}")
    #print("-" * 60)

    file_reader1 = file_reader(files[0], chances[0], PORTS[0:2])
    file_reader2 = file_reader(files[1], chances[1], PORTS[2:4])
    file_readers = [file_reader1, file_reader2]

    session = requests.Session()

    while True:
        for f_reader in file_readers:
            try:
                data_missed, data_clean =  f_reader.generate_values()
                f_reader.next_step()
                payload = {
                    "port": f_reader.ports[0],
                    "timestamp": time.time(),
                    "values": data_missed
                }
                payload_clean = {
                    "port": f_reader.ports[1],
                    "timestamp": time.time(),
                    "values": data_clean
                }

                r = session.post(SERVER_URL, json=payload, timeout=2.5)
                if r.status_code == 200:
                    status = "✓"
                else:
                    status = f"✗ {r.status_code}"

                misses = sum(1 for v in payload["values"].values() if v is None)
                #print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                #      f"port {f_reader.ports[0]} | {status} | misses: {misses}")

                r_clean = session.post(SERVER_URL, json=payload_clean, timeout=2.5)
                if r_clean.status_code == 200:
                    status = "✓"
                else:
                    status = f"✗ {r.status_code}"

                misses = sum(1 for v in payload["values"].values() if v is None)
                #print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
                #      f"port {f_reader.ports[1]} | {status} | misses: {misses}")


            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] → error: {e}")

            time.sleep(INTERVAL / len(PORTS))

        # Небольшая пауза между циклами по всем портам
        time.sleep(0.1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nОстановлено")
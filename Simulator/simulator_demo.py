import json
import sys
import numpy as np
import pandas as pd
import subprocess
import websockets, os, socket
import asyncio
import random

files = ["PowerConsumption1.csv", "energydata_complete.csv"]
ports = [8092, 8093, 8094, 8095]
chances = [0.30, 0.20] # 0.0125, 0.025
intervals = [5000, 7000]
time_format='%Y-%m-%d %H:%M:%S'

class Facility:
    port_main = None # Порт для имитации реальной работы установки
    port_test = None # Порт для отправки данных без помех
    file_path = None # Путь к файлу с данными
    client_main = None # Объект клиента для главного порта
    client_test = None # Объект клиента для тестового порта

    row_min = None # Минимальная строка, в которой может считываться файл
    row_cur = None # Текущая строка, в которой считывается файл
    row_max = None # Максимальная строка, в которой может считываться файл

    interval = None  # Время в миллисекундах между переходами на следующие строчки

    points = None # Точки, полученные из файла
    columns = None # Список колонок файла

    chance = None # Вероятность пропуска данных
    chance_seq = None # Мультипликатор вероятости в случае если предыдущая запись - пропуск
    _is_empty = None # Предыдущая запись - пропуск?

    def __init__(self, port_main, port_test, file_path, interval, chance, time_format):
        self.port_main = port_main
        self.port_test = port_test
        self.file_path = file_path
        self.interval = interval
        self.chance = chance

        self.read_file()
        self.time_format = time_format
        _is_empty = False
        asyncio.get_event_loop().run_until_complete(self.run_websocket_main())
        asyncio.get_event_loop().run_until_complete(self.run_websocket_test())

    def read_file(self):
        """Считать данные из .csv файла"""
        csv_path = os.path.join(os.path.dirname(__file__), self.file_path)
        data = pd.read_csv(csv_path).dropna()

        self.points = data.values #self.data.iloc[:, [0, 1]].values
        self.columns = data.columns[1:]
        self.row_min = self.row_cur = 0
        self.row_max = data.iloc[:, 1].size - 5

    async def run_websocket_main(self):
        """Подключиться к главному порту"""
        host = os.getenv("WEBSOCKET_HOST", socket.gethostbyname(socket.gethostname()))
        url_main = f"ws://{host}:{self.port_main}"
        print(f"Подключаюсь к {url_main}")
        try:
            self.client_main = await websockets.connect(url_main)
            print("Подключение установлено")
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            raise

    async def run_websocket_test(self):
        """Подключиться к тестовому порту"""
        host = os.getenv("WEBSOCKET_HOST", socket.gethostbyname(socket.gethostname()))
        url_test = f"ws://{host}:{self.port_test}"
        print(f"Подключаюсь к {url_test}")
        try:
            self.client_test = await websockets.connect(url_test)
            print("Подключение установлено")
        except Exception as e:
            print(f"Ошибка подключения: {e}")
            raise
    def parse_timestamp(self, timestamp):
        """Привести временную метку к единому формату"""
        return pd.to_datetime(timestamp).strftime(self.time_format)
    async def upload_main(self, res):
        """Загрузить пакет данных на главный порт"""
        try:
            if self.client_main is None or not self.client_main.open:
                await self.run_websocket_main()
            await self.client_main.send(json.dumps(res))
        except Exception as e:
            print(f"Ошибка отправки (main): {e}")
            await self.run_websocket_main()  # Переподключение

    async def upload_test(self, res):
        """Загрузить пакет данных на тестовый порт"""
        try:
            if self.client_test is None or not self.client_test.open:
                await self.run_websocket_test()
            await self.client_test.send(json.dumps(res))
        except Exception as e:
            print(f"Ошибка отправки (test): {e}")
            await self.run_websocket_test()  # Переподключение

    async def simulation(self):
        """Имитация работы установки"""
        while True:
            try:
                self.row_cur += 1
                if self.row_cur >= self.row_max:
                    self.row_cur = self.row_min

                res = { #Формирование пакета данных
                    'names': self.columns.tolist(),
                    'values': self.points[self.row_cur, 1:].tolist(),
                    'timeStamp': self.parse_timestamp(self.points[self.row_cur, 0]),
                    'iteration': self.row_cur
                }

                await self.upload_test(res)

                points_out = []
                for i in range(1, self.points.shape[1]):
                    if random.random() <= self.chance:
                        points_out.append(np.nan)
                    else:
                        points_out.append(self.points[self.row_cur, i])

                res = {'names': self.columns.tolist(),
                       'values': points_out,
                       'timeStamp': self.parse_timestamp(self.points[self.row_cur, 0]),
                       'iteration': self.row_cur
                       }

                await self.upload_main(res)

            except Exception as e:
                print(f"Критическая ошибка в simulation: {e}")
                await asyncio.sleep(5)
                continue

            await asyncio.sleep(self.interval / 1000)

async def run_simulation():
    """Запустить параллельно симуляцию обеих установок"""
    await asyncio.gather(facility_1.simulation(), facility_2.simulation())

if __name__ == "__main__":
    print(sys.executable)
    print(sys.path)
    server_app = os.path.join(os.path.dirname(__file__), 'server_web.py')
    subprocess.Popen([sys.executable, server_app, f"{ports[0]}-{ports[1]}-{ports[2]}-{ports[3]}"])

    loop = asyncio.get_event_loop()

    facility_1 = Facility(
        port_main=ports[0],
        port_test=ports[1],
        file_path=files[0],
        interval=intervals[0],
        chance=chances[0],
        time_format=time_format
        )

    facility_2 = Facility(
        port_main=ports[2],
        port_test=ports[3],
        file_path=files[1],
        interval=intervals[1],
        chance=chances[1],
        time_format=time_format
    )

    try:
        loop.run_until_complete(run_simulation())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

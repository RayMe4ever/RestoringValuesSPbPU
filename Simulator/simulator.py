import os
import asyncio
import websockets
import json
import numpy as np
import pandas as pd
import random
from datetime import datetime


class Facility:

    def __init__(
        self,
        name: str,
        port_main: int,
        port_test: int,
        names: list[str],
        send_delay: int = 1000,
    ):
        self.name = name
        self.port_main = port_main
        self.port_test = port_test
        self.names = names
        self.send_delay = send_delay / 1000

        self.client_main = None
        self.client_test = None

        # Подключаемся (с retry)
        asyncio.get_event_loop().run_until_complete(self.run_websocket_main())
        asyncio.get_event_loop().run_until_complete(self.run_websocket_test())

    # =========================
    # RETRY CONNECT (MAIN)
    # =========================
    async def run_websocket_main(self):
        host = os.getenv("WEBSOCKET_HOST", "127.0.0.1")
        url_main = f"ws://{host}:{self.port_main}"

        while True:
            try:
                print(f"[{self.name}] Подключаюсь к {url_main}")
                self.client_main = await websockets.connect(url_main)
                print(f"[{self.name}] Подключение установлено (main)")
                return
            except Exception as e:
                print(f"[{self.name}] Ошибка подключения (main): {e}")
                print("Retry через 2 сек...")
                await asyncio.sleep(2)

    # =========================
    # RETRY CONNECT (TEST)
    # =========================
    async def run_websocket_test(self):
        host = os.getenv("WEBSOCKET_HOST", "127.0.0.1")
        url_test = f"ws://{host}:{self.port_test}"

        while True:
            try:
                print(f"[{self.name}] Подключаюсь к {url_test}")
                self.client_test = await websockets.connect(url_test)
                print(f"[{self.name}] Подключение установлено (test)")
                return
            except Exception as e:
                print(f"[{self.name}] Ошибка подключения (test): {e}. Retry in 2s...")
                await asyncio.sleep(2)

    # =========================
    # GENERATE DATA
    # =========================
    def generate_values(self):
        return list(np.random.rand(len(self.names)))

    # =========================
    # SEND LOOP
    # =========================
    async def send_data(self):
        while True:
            data = {
                "names": self.names,
                "values": self.generate_values(),
                "timeStamp": datetime.utcnow().isoformat(),
            }

            try:
                if self.client_main:
                    await self.client_main.send(json.dumps(data))
                if self.client_test:
                    await self.client_test.send(json.dumps(data))
            except Exception as e:
                print(f"[{self.name}] Ошибка отправки: {e}")

            await asyncio.sleep(self.send_delay)


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    print("Python:", os.sys.executable)
    print("PYTHONPATH:", os.sys.path)

    facility_1 = Facility(
        name="facility_1",
        port_main=8092,
        port_test=8093,
        names=["temp", "pressure", "flow"],
        send_delay=1000,
    )

    facility_2 = Facility(
        name="facility_2",
        port_main=8094,
        port_test=8095,
        names=["temp", "pressure", "flow"],
        send_delay=1000,
    )

    loop = asyncio.get_event_loop()
    tasks = [
        facility_1.send_data(),
        facility_2.send_data(),
    ]

    loop.run_until_complete(asyncio.gather(*tasks))
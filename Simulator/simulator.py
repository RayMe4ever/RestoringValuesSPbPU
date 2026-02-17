# Simulator/simulator.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logging_setup import setup_logging, new_op_id

import json
import numpy as np
import pandas as pd
import subprocess
import websockets
import socket
import asyncio
import random

logger = setup_logging("simulator")

files = ["PowerConsumption1.csv", "energydata_complete.csv"]
ports = [8092, 8093, 8094, 8095]
chances = [0.0125, 0.025]
intervals = [5000, 7000]
time_format = "%Y-%m-%d %H:%M:%S"


class Facility:
    def __init__(self, port_main, port_test, file_path, interval, chance, time_format):
        self.port_main = port_main
        self.port_test = port_test
        self.file_path = file_path
        self.interval = interval
        self.chance = chance

        self.client_main = None
        self.client_test = None

        self.row_min = None
        self.row_cur = None
        self.row_max = None

        self.points = None
        self.columns = None

        self.time_format = time_format
        self._is_empty = False

        op_id = new_op_id("facility-init")
        logger.info(
            "init facility file=%s port_main=%s port_test=%s interval_ms=%s chance=%s",
            file_path,
            port_main,
            port_test,
            interval,
            chance,
            extra={"op_id": op_id},
        )

        self.read_file(op_id=op_id)
        asyncio.get_event_loop().run_until_complete(self.run_websocket_main(op_id=op_id))
        asyncio.get_event_loop().run_until_complete(self.run_websocket_test(op_id=op_id))

    def read_file(self, op_id: str):
        """Считать данные из .csv файла"""
        csv_path = os.path.join(os.path.dirname(__file__), self.file_path)
        data = pd.read_csv(csv_path).dropna()

        self.points = data.values
        self.columns = data.columns[1:]
        self.row_min = self.row_cur = 0
        self.row_max = data.iloc[:, 1].size - 5

        logger.info(
            "read_file ok path=%s rows=%s cols=%s",
            csv_path,
            data.shape[0],
            len(self.columns),
            extra={"op_id": op_id},
        )

    async def run_websocket_main(self, op_id: str):
        """Подключиться к главному порту"""
        host = os.getenv("WEBSOCKET_HOST", socket.gethostbyname(socket.gethostname()))
        url_main = f"ws://{host}:{self.port_main}"
        logger.info("connect main url=%s", url_main, extra={"op_id": op_id})
        self.client_main = await websockets.connect(url_main)
        logger.info("connected main", extra={"op_id": op_id})

    async def run_websocket_test(self, op_id: str):
        """Подключиться к тестовому порту"""
        host = os.getenv("WEBSOCKET_HOST", socket.gethostbyname(socket.gethostname()))
        url_test = f"ws://{host}:{self.port_test}"
        logger.info("connect test url=%s", url_test, extra={"op_id": op_id})
        self.client_test = await websockets.connect(url_test)
        logger.info("connected test", extra={"op_id": op_id})

    def parse_timestamp(self, timestamp):
        """Привести временную метку к единому формату"""
        return pd.to_datetime(timestamp).strftime(self.time_format)

    async def upload_main(self, res, op_id: str):
        """Загрузить пакет данных на главный порт"""
        try:
            if self.client_main is None or not self.client_main.open:
                logger.warning("main ws not open -> reconnect", extra={"op_id": op_id})
                await self.run_websocket_main(op_id=op_id)
            await self.client_main.send(json.dumps(res))
        except Exception as e:
            logger.exception("send main failed err=%s", e, extra={"op_id": op_id})
            await self.run_websocket_main(op_id=op_id)

    async def upload_test(self, res, op_id: str):
        """Загрузить пакет данных на тестовый порт"""
        try:
            if self.client_test is None or not self.client_test.open:
                logger.warning("test ws not open -> reconnect", extra={"op_id": op_id})
                await self.run_websocket_test(op_id=op_id)
            await self.client_test.send(json.dumps(res))
        except Exception as e:
            logger.exception("send test failed err=%s", e, extra={"op_id": op_id})
            await self.run_websocket_test(op_id=op_id)

    async def simulation(self):
        """Имитация работы установки"""
        while True:
            op_id = new_op_id(f"sim-{self.port_main}-{self.row_cur:06d}")
            try:
                self.row_cur += 1
                if self.row_cur >= self.row_max:
                    self.row_cur = self.row_min

                ts = self.parse_timestamp(self.points[self.row_cur, 0])

                # Пакет без пропусков (test)
                res_test = {
                    "names": self.columns.tolist(),
                    "values": self.points[self.row_cur, 1:].tolist(),
                    "timeStamp": ts,
                    "iteration": self.row_cur,
                    "op_id": op_id,
                }
                await self.upload_test(res_test, op_id=op_id)

                # Пакет с пропусками (main)
                points_out = []
                missing = 0
                for i in range(1, self.points.shape[1]):
                    if random.random() <= self.chance:
                        points_out.append(np.nan)
                        missing += 1
                    else:
                        points_out.append(self.points[self.row_cur, i])

                res_main = {
                    "names": self.columns.tolist(),
                    "values": points_out,
                    "timeStamp": ts,
                    "iteration": self.row_cur,
                    "op_id": op_id,
                }

                logger.info(
                    "send_main port=%s iteration=%s missing=%s/%s ts=%s",
                    self.port_main,
                    self.row_cur,
                    missing,
                    len(points_out),
                    ts,
                    extra={"op_id": op_id},
                )

                await self.upload_main(res_main, op_id=op_id)

            except Exception as e:
                logger.exception("critical simulation error err=%s", e, extra={"op_id": op_id})
                await asyncio.sleep(5)

            await asyncio.sleep(self.interval / 1000)


async def run_simulation():
    """Запустить параллельно симуляцию обеих установок"""
    await asyncio.gather(facility_1.simulation(), facility_2.simulation())


if __name__ == "__main__":
    op_id = new_op_id("sim-main")
    server_app = os.path.join(os.path.dirname(__file__), "server_web.py")
    ports_arg = f"{ports[0]}-{ports[1]}-{ports[2]}-{ports[3]}"

    logger.info("start server_web %s arg=%s", server_app, ports_arg, extra={"op_id": op_id})
    subprocess.Popen([sys.executable, server_app, ports_arg])

    loop = asyncio.get_event_loop()

    facility_1 = Facility(
        port_main=ports[0],
        port_test=ports[1],
        file_path=files[0],
        interval=intervals[0],
        chance=chances[0],
        time_format=time_format,
    )

    facility_2 = Facility(
        port_main=ports[2],
        port_test=ports[3],
        file_path=files[1],
        interval=intervals[1],
        chance=chances[1],
        time_format=time_format,
    )

    try:
        loop.run_until_complete(run_simulation())
    except KeyboardInterrupt:
        logger.warning("shutdown by keyboard interrupt", extra={"op_id": op_id})
    finally:
        loop.close()
        logger.info("event loop closed", extra={"op_id": op_id})

# Reciever/reciever.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logging_setup import setup_logging, new_op_id

import asyncio
import websockets
import json
import socket
import csv
from collections import deque

logger = setup_logging("reciever")

# {port: {'buffer': deque, 'names': list}}
port_data = {}
port_data_long = {}


def _parse_ports_arg() -> str:
    # приоритет: env -> argv -> default
    env_val = os.getenv("WEBSOCKET_PORTS")
    if env_val:
        return env_val
    if len(sys.argv) > 1:
        return sys.argv[1]
    return "8092-8093-8094-8095"


async def write_csv(port: int, buffer: deque, filename: str, op_id: str):
    """Записывает весь буфер в CSV файл"""
    filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    try:
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)

            if port in port_data:
                writer.writerow(["DateTime"] + port_data[port]["names"])

            for row in buffer:
                writer.writerow(row)

        logger.debug(
            "csv written path=%s rows=%s",
            filepath,
            len(buffer),
            extra={"op_id": op_id},
        )
    except Exception as e:
        logger.exception("csv write failed path=%s err=%s", filepath, e, extra={"op_id": op_id})


async def update_csv(port: int, values, timestamp=None, op_id: str = "-"):
    """Обновляет данные и периодически записывает в CSV файл"""
    if port not in port_data:
        logger.warning("port not initialized port=%s", port, extra={"op_id": op_id})
        return

    if not isinstance(values, (list, tuple)):
        logger.warning("invalid values type=%s", type(values), extra={"op_id": op_id})
        return

    try:
        full_values = [timestamp] + list(values)

        port_data[port]["buffer"].append(full_values)
        port_data_long[port]["buffer"].append(full_values)

        await write_csv(port, port_data[port]["buffer"], f"data_port_{port}.csv", op_id=op_id)
        await write_csv(port, port_data_long[port]["buffer"], f"data_port_{port}_long.csv", op_id=op_id)

    except Exception as e:
        logger.exception("update_csv failed port=%s err=%s", port, e, extra={"op_id": op_id})


async def receive_data(websocket_port: int):
    """Получить данные с websocket-порта"""
    host = os.getenv("WEBSOCKET_HOST", socket.gethostbyname(socket.gethostname()))
    uri = f"ws://{host}:{websocket_port}"

    while True:
        conn_op = new_op_id(f"recv-conn-{websocket_port}")
        try:
            logger.info("connecting uri=%s", uri, extra={"op_id": conn_op})
            async with websockets.connect(uri) as websocket:
                logger.info("connected port=%s", websocket_port, extra={"op_id": conn_op})

                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(response)

                        op_id = data.get("op_id") or new_op_id(f"recv-{websocket_port}")
                        timestamp = data.get("timeStamp", "None")

                        if "names" not in data:
                            logger.warning("invalid packet: no names", extra={"op_id": op_id})
                            continue

                        # Инициализация буферов (если первый пакет или изменились names)
                        if (
                            websocket_port not in port_data
                            or port_data[websocket_port]["names"] != data["names"]
                        ):
                            port_data[websocket_port] = {
                                "buffer": deque(maxlen=10),
                                "names": data["names"],
                            }
                            port_data_long[websocket_port] = {
                                "buffer": deque(maxlen=1000),
                                "names": data["names"],
                            }
                            logger.info(
                                "init buffers port=%s cols=%s",
                                websocket_port,
                                len(data["names"]),
                                extra={"op_id": op_id},
                            )

                        if "values" in data:
                            await update_csv(websocket_port, data["values"], timestamp=timestamp, op_id=op_id)
                            logger.debug(
                                "packet stored port=%s iteration=%s ts=%s",
                                websocket_port,
                                data.get("iteration"),
                                timestamp,
                                extra={"op_id": op_id},
                            )
                        else:
                            logger.warning("invalid packet: no values keys=%s", list(data.keys()), extra={"op_id": op_id})

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("connection closed -> reconnect", extra={"op_id": conn_op})
                        break
                    except json.JSONDecodeError as e:
                        logger.warning("json decode error err=%s", e, extra={"op_id": conn_op})

        except Exception as e:
            logger.exception("connect failed port=%s err=%s", websocket_port, e, extra={"op_id": conn_op})
            await asyncio.sleep(5)


async def listen_ports(ports):
    """Обрабатывать каждый из портов"""
    tasks = [asyncio.create_task(receive_data(port)) for port in ports]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    arg = _parse_ports_arg()
    ports = [int(p) for p in arg.split("-")]

    logger.info("receiver started ports=%s", ports, extra={"op_id": new_op_id("recv-main")})
    asyncio.run(listen_ports(ports))

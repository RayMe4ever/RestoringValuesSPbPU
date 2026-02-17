import asyncio
import json
import os
import csv
from collections import deque
import websockets

# Буферы данных по порту
port_data = {}       # {port: {'buffer': deque, 'names': list}}
port_data_long = {}  # {port: {'buffer': deque, 'names': list}}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def csv_path(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)


async def write_csv(port: int, buffer: deque, filename: str):
    filepath = csv_path(filename)
    try:
        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            names = port_data.get(port, {}).get("names", [])
            w.writerow(["DateTime"] + names)
            for row in buffer:
                w.writerow(row)
    except Exception as e:
        print(f"[reciever] write_csv error {filepath}: {e}")


async def update_csv(port: int, values, timestamp: str):
    if port not in port_data:
        # Если пришли values раньше names — инициализируем пустыми
        port_data[port] = {"buffer": deque(maxlen=10), "names": []}
        port_data_long[port] = {"buffer": deque(maxlen=1000), "names": []}

    full_row = [timestamp] + list(values)
    port_data[port]["buffer"].append(full_row)
    port_data_long[port]["buffer"].append(full_row)

    # Пишем на каждый апдейт (у тебя так и было)
    await write_csv(port, port_data[port]["buffer"], f"data_port_{port}.csv")
    await write_csv(port, port_data_long[port]["buffer"], f"data_port_{port}_long.csv")


async def handler(websocket, port: int):
    print(f"[reciever] client connected on port {port}")
    async for msg in websocket:
        try:
            data = json.loads(msg)
        except json.JSONDecodeError:
            continue

        # names (заголовки) приходят в каждом сообщении у Simulator
        if "names" in data:
            names = data["names"]
            if port not in port_data or port_data[port]["names"] != names:
                port_data[port] = {"buffer": deque(maxlen=10), "names": names}
                port_data_long[port] = {"buffer": deque(maxlen=1000), "names": names}

        ts = data.get("timeStamp", "None")

        if "values" in data:
            await update_csv(port, data["values"], ts)
        else:
            # если пришёл странный пакет — просто игнорим
            continue


async def serve_one_port(port: int, host: str):
    async def _h(ws):
        return await handler(ws, port)

    async with websockets.serve(_h, host, port):
        print(f"[reciever] listening ws://{host}:{port}")
        await asyncio.Future()  # run forever


async def main():
    host = os.getenv("BIND_HOST", "0.0.0.0")   # важно для доступа по IP
    ports = [8092, 8093, 8094, 8095]
    await asyncio.gather(*(serve_one_port(p, host) for p in ports))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("[reciever] shutdown")
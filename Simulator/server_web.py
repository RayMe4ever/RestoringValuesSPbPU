# Simulator/server_web.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logging_setup import setup_logging, new_op_id

import asyncio
import websockets
import json
from collections import defaultdict

logger = setup_logging("server_web")

port_data = defaultdict(dict)      # Данные на каждом из портов
port_clients = defaultdict(set)    # Список подключенных клиентов


def _get_local_port(websocket) -> int | None:
    try:
        if websocket.local_address:
            return websocket.local_address[1]
    except Exception:
        pass
    return getattr(websocket, "port", None)


async def handle_connection(websocket):
    """Обслуживать клиентов на порту"""
    local_port = _get_local_port(websocket)
    conn_op = new_op_id(f"ws-conn-{local_port}")

    logger.info(
        "new connection local_port=%s remote=%s",
        local_port,
        getattr(websocket, "remote_address", None),
        extra={"op_id": conn_op},
    )

    port_clients[local_port].add(websocket)

    try:
        # Отправка текущих данных новому клиенту
        if local_port in port_data and "latest_data" in port_data[local_port]:
            await websocket.send(json.dumps(port_data[local_port]["latest_data"]))

        while True:
            message = await asyncio.wait_for(websocket.recv(), timeout=60)
            try:
                data = json.loads(message)
                msg_op = data.get("op_id") or new_op_id(f"ws-msg-{local_port}")
                port_data[local_port]["latest_data"] = data

                logger.debug(
                    "recv port=%s keys=%s iteration=%s ts=%s",
                    local_port,
                    list(data.keys()),
                    data.get("iteration"),
                    data.get("timeStamp"),
                    extra={"op_id": msg_op},
                )

                await broadcast_to_port(local_port, data, op_id=msg_op)

            except json.JSONDecodeError:
                logger.warning("json decode error port=%s", local_port, extra={"op_id": conn_op})

    except asyncio.TimeoutError:
        try:
            await websocket.ping()
        except Exception:
            logger.warning("ping failed -> closing", extra={"op_id": conn_op})
    except websockets.ConnectionClosed:
        logger.info("connection closed", extra={"op_id": conn_op})
    except Exception as e:
        logger.exception("unexpected error err=%s", e, extra={"op_id": conn_op})
    finally:
        port_clients[local_port].discard(websocket)
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("disconnected port=%s clients_now=%s", local_port, len(port_clients[local_port]), extra={"op_id": conn_op})


async def broadcast_to_port(port, data, op_id: str):
    """Безопасная рассылка с обработкой отключённых клиентов"""
    if port not in port_clients:
        return

    message = json.dumps(data)
    dead_clients = set()

    for client in port_clients[port]:
        try:
            if client.open:
                await client.send(message)
            else:
                dead_clients.add(client)
        except (websockets.ConnectionClosed, RuntimeError) as e:
            logger.warning("send failed port=%s err=%s", port, str(e), extra={"op_id": op_id})
            dead_clients.add(client)

    for client in dead_clients:
        port_clients[port].discard(client)

    logger.debug(
        "broadcast port=%s sent_to=%s dropped=%s",
        port,
        len(port_clients[port]),
        len(dead_clients),
        extra={"op_id": op_id},
    )


async def run_servers(ports):
    """Запустить сервера на каждом из портов"""
    servers = []
    for port in ports:
        server = await websockets.serve(handle_connection, "0.0.0.0", port, ping_interval=20, ping_timeout=60)
        servers.append(server)
        logger.info("ws server started port=%s", port, extra={"op_id": new_op_id("ws-start")})

    await asyncio.Future()  # Бесконечное ожидание


if __name__ == "__main__":
    ports = [int(p) for p in sys.argv[1].split("-")]
    logger.info("starting ws servers ports=%s", ports, extra={"op_id": new_op_id("ws-main")})
    asyncio.run(run_servers(ports))

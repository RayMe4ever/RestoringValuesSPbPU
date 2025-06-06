import asyncio
import websockets
import json
import sys
from collections import defaultdict

port_data = defaultdict(dict) # Данные на каждом из портов
port_clients = defaultdict(set) # Список подключенных клиентов

async def handle_connection(websocket, path):
    """Обслуживать клиентов на порту"""
    port = websocket.port
    print(f"Новое подключение на порту {port}")

    websocket.ping_interval = 20
    websocket.ping_timeout = 60

    port_clients[port].add(websocket)

    try:
        # Отправка текущих данных новому клиенту
        if port in port_data and 'latest_data' in port_data[port]:
            try:
                await websocket.send(json.dumps(port_data[port]['latest_data']))
            except websockets.ConnectionClosed:
                return

        # Мониторинг активности
        while True:
            try:
                message = await asyncio.wait_for(websocket.recv(), timeout=60)
                try:
                    data = json.loads(message)
                    port_data[port]['latest_data'] = data
                    print(f"Получены данные {port}: {data}")
                    await broadcast_to_port(port, data)
                except json.JSONDecodeError:
                    print(f"Ошибка декодирования JSON на порту {port}")
            except asyncio.TimeoutError:
                # Проверяем соединение
                try:
                    await websocket.ping()
                except:
                    break  # Соединение разорвано
            except websockets.ConnectionClosed:
                break
            except Exception as e:
                print(f"Неожиданная ошибка на порту {port}: {e}")
                break

    finally:
        port_clients[port].discard(websocket)
        await websocket.close()


async def broadcast_to_port(port, data):
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
            print(f"Ошибка отправки на порту {port}: {str(e)}")
            dead_clients.add(client)

    # Удаляем мёртвые соединения
    for client in dead_clients:
        port_clients[port].discard(client)


async def run_servers(ports):
    """Запустить сервера на каждом из портов"""
    servers = []
    for port in ports:
        server = await websockets.serve(
            handle_connection,
            "0.0.0.0",
            port,
            ping_interval=20,
            ping_timeout=60
        )
        servers.append(server)
        print(f"Сервер запущен на порту {port}")

    await asyncio.Future()  # Бесконечное ожидание


if __name__ == "__main__":
    ports = [int(p) for p in sys.argv[1].split('-')]

    try:
        asyncio.run(run_servers(ports))
    except KeyboardInterrupt:
        print("\nСервер завершает работу...")
    except Exception as e:
        print(f"Фатальная ошибка: {e}")
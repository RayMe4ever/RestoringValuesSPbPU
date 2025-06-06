import asyncio
import websockets
import json
import sys
import os
import socket
from time import monotonic


async def receive_data(websocket_port, max_retries=5, retry_delay=3):
    host = os.getenv("WEBSOCKET_HOST", socket.gethostbyname(socket.gethostname()))
    uri = f"ws://{host}:{websocket_port}"
    retry_count = 0

    while retry_count < max_retries:
        try:
            print(f"Попытка подключения к {uri} (попытка {retry_count + 1}/{max_retries})")
            async with websockets.connect(uri) as websocket:
                print("Подключение установлено")
                retry_count = 0  # Сброс счетчика при успешном подключении

                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                        data = json.loads(response)

                        with open(f'data_{websocket_port}.json', 'w') as f:
                            json.dump(data, f)

                        print(f"[Порт {websocket_port}] Получены данные: {json.dumps(data, indent=4)}")

                    except asyncio.TimeoutError:
                        print(f"[Порт {websocket_port}] Нет данных в течение 10 секунд")
                        # Отправляем ping для поддержания соединения
                        await websocket.ping()

                    except json.JSONDecodeError:
                        print(f"[Порт {websocket_port}] Ошибка декодирования JSON")

        except (websockets.ConnectionClosed, ConnectionRefusedError) as e:
            print(f"[Порт {websocket_port}] Ошибка соединения: {e}")
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(retry_delay)

        except Exception as e:
            print(f"[Порт {websocket_port}] Неожиданная ошибка: {e}")
            retry_count += 1
            if retry_count < max_retries:
                await asyncio.sleep(retry_delay)

    print(f"[Порт {websocket_port}] Превышено максимальное количество попыток подключения")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование: python websocket_scanner.py <port>")
        sys.exit(1)

    websocket_port = sys.argv[1]
    print(f"Запуск сканера для порта: {websocket_port}")
    asyncio.run(receive_data(websocket_port))
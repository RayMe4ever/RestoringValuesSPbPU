import asyncio
import websockets
import json
import os
import socket
import csv
from collections import deque

# Словарь для хранения данных для каждого порта
port_data = {}  # Формат: {port: {'buffer': deque(maxlen=300), 'names': list, 'columns_count': int}}
port_data_long = {}  # Формат: {port: {'buffer': deque(maxlen=300), 'names': list, 'columns_count': int}}


async def write_csv(port, buffer, filename):
    """Записывает весь буфер в CSV файл"""
    try:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
        with open(filepath, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Записываем заголовки (timeStamp + имена колонок)
            if port in port_data:
                writer.writerow(['DateTime'] + port_data[port]['names'])

            # Записываем все данные из буфера
            for row in buffer:
                writer.writerow(row)
        print(f"Данные записаны в {filepath} (строк: {len(buffer)})")

    except Exception as e:
        print(f"Ошибка при записи в файл {filepath}: {e}")


async def update_csv(port, values, timestamp=None):
    """Обновляет данные и периодически записывает в CSV файл"""
    if port not in port_data:
        print(f"Ошибка: данные для порта {port} не инициализированы")
        return

    try:
        # Добавляем timestamp в начало данных
        full_values = [timestamp] + values

        # Добавляем новые данные в буферы
        port_data[port]['buffer'].append(full_values)
        port_data_long[port]['buffer'].append(full_values)

        # Записываем в файлы только при достижении определенного размера буфера или периодически
        await write_csv(port, port_data[port]['buffer'], f"data_port_{port}.csv")
        await write_csv(port, port_data_long[port]['buffer'], f"data_port_{port}_long.csv")

    except Exception as e:
        print(f"Ошибка при обновлении CSV для порта {port}: {e}")


async def receive_data(websocket_port):
    """Получить данные с websocket-порта"""
    host = os.getenv("WEBSOCKET_HOST", socket.gethostbyname(socket.gethostname()))
    uri = f"ws://{host}:{websocket_port}"

    while True:  # Бесконечный цикл для переподключения
        try:
            async with websockets.connect(uri) as websocket:
                print(f"Подключено к порту {websocket_port}")

                while True:
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=1.0)

                        data = json.loads(response)

                        # Проверяем наличие необходимых полей
                        if 'names' in data:
                            # Получаем timestamp из пакета или None
                            timestamp = data.get('timeStamp', "None")

                            # Если это первый пакет или names изменились, инициализируем
                            if (websocket_port not in port_data or
                                   port_data[websocket_port]['names'] != data['names']):
                                port_data[websocket_port] = {
                                    'buffer': deque(maxlen=10),
                                    'names': data['names'],
                                    'columns_count': len(data['names']) + 1  # +1 для timeStamp
                                }
                                port_data_long[websocket_port] = {
                                    'buffer': deque(maxlen=1000),
                                    'names': data['names'],
                                    'columns_count': len(data['names']) + 1  # +1 для timeStamp
                                }
                            if 'None' in data:
                                print(f"Получен None-пакет от порта {websocket_port}")
                                await update_csv(websocket_port, "None", timestamp="None")
                            # Обновляем CSV с новыми данными
                            elif 'values' in data:
                                await update_csv(websocket_port, data['values'], timestamp=timestamp)
                        else:
                            print(f"Получен некорректный пакет от порта {websocket_port}: {response}")

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print(f"Соединение с портом {websocket_port} закрыто, переподключаемся...")
                        break
                    except json.JSONDecodeError as e:
                        print(f"Ошибка декодирования JSON от порта {websocket_port}: {e}")
                        continue

        except Exception as e:
            print(f"Ошибка подключения к порту {websocket_port}: {e}, повторная попытка через 5 секунд...")
            await asyncio.sleep(5)


async def listen_ports(ports):
    """Обрабатывать каждый из портов"""
    tasks = [asyncio.create_task(receive_data(port)) for port in ports]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    arg = "8092-8093-8094-8095"
    print(f"Ресивер-коллектор запущен с аргументами: {arg}")#{sys.argv}")
    ports = [int(p) for p in arg.split('-')]#sys.argv[1].split('-')]

    try:
        asyncio.run(listen_ports(ports))
    except KeyboardInterrupt:
        print("Завершение работы...")
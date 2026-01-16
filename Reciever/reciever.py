# Reciever/reciever.py
import csv
import time
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import datetime
import os
import shutil

app = FastAPI(title="Data Receiver - REST API with NaN replacement")

# Папка для хранения данных
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "received_data"
)

# Константы
SHORT_HISTORY = 10      # количество строк в обычном файле
LONG_HISTORY = 1000     # количество строк в _long файле

# Значения, которые нужно заменять на NaN
BAD_VALUES = {-100, -100000, -400000}


def save_to_file(port: int, timestamp: float, names: list, time_str: str, values: list):
    """
    Сохраняет данные в два файла: короткий (10 строк) и длинный (1000 строк)
    Заменяет плохие значения на np.nan
    """
    # Преобразуем значения: плохие → np.nan
    cleaned_values = []
    for v in values:
        if v in BAD_VALUES:
            cleaned_values.append(np.nan)
        else:
            cleaned_values.append(v)

    # Подготовка строки данных
    row = [timestamp] + cleaned_values

    # Заголовки
    headers = ["DateTime"] + names

    # Функция для сохранения с ограничением количества строк
    def append_and_truncate(filename, data_row: list, header_row: list, max_lines: int):
        lines = []
        file_exists = os.path.exists(filename)

        if file_exists:
            with open(filename, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                lines = list(reader)

        # Добавляем новую строку
        if not file_exists or len(lines) == 0:
            lines = [header_row]

        lines.append(data_row)

        # Оставляем только последние max_lines строк (сохраняем заголовок)
        if len(lines) > max_lines + 1:  # +1 потому что заголовок
            lines = [lines[0]] + lines[-(max_lines):]

        # Перезаписываем файл
        with open(filename, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(lines)

    # Имена файлов
    short_file = os.path.join(DATA_DIR, f"data_port_{port}.csv")
    long_file = os.path.join(DATA_DIR, f"data_port_{port}_long.csv")

    # Сохраняем в оба файла
    append_and_truncate(short_file, row, headers, SHORT_HISTORY)
    append_and_truncate(long_file, row, headers, LONG_HISTORY)


@app.post("/data")
async def receive_sensor_data(request: Request):
    try:
        data = await request.json()

        port = data.get("port")
        if port is None:
            raise ValueError("Поле 'port' обязательно")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        names_str = data.get("names", "[]")
        time_str = data.get("timeStamp", "")
        values_str = data.get("values", "[]")

        # Парсим списки (предполагаем, что приходят в виде строк JSON)
        try:
            names = values_str['names']
            values = values_str['values']
        except Exception as e:
            raise ValueError(f"Ошибка парсинга names/values: {e}")

        if not isinstance(names, list) or not isinstance(values, list):
            raise ValueError("names и values должны быть списками")

        if len(names) != len(values):
            raise ValueError("Количество имён и значений не совпадает")

        # Сохраняем данные
        save_to_file(port, timestamp, names, time_str, values)

        return JSONResponse({
            "status": "received",
            "port": port,
            "values_count": len(values),
            "received_at": time.time()
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    #print("Запуск REST-приёмника данных...")
    #print(f"Сохранение в: {DATA_DIR}")
    #print(f"Формат файлов:")
    #print(f"  • data_port_XXXX.csv      — последние {SHORT_HISTORY} значений")
    #print(f"  • data_port_XXXX_long.csv — последние {LONG_HISTORY} значений")
    #print("Значения -100 / -100000 / -400000 заменяются на NaN\n")

    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    os.makedirs(DATA_DIR, exist_ok=True)

    uvicorn.run(
        "reciever:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
from model import knn_model
from data_source import data_source

import asyncio
from aiohttp import web

model_delay = 3000

# ----------------------
#  HTTP‐API для управления
# ----------------------

async def set_interval_handler(request):
    """
    POST /set_interval
    Ждёт JSON {"period_ms": <int>}, меняет interpolation_period.
    """
    global model_delay
    try:
        data = await request.json()
        new_val = int(data.get("period_ms"))
        print(f"Recieved interval: {new_val}")
        if new_val < 100 or new_val > 60000:
            raise ValueError
        model_delay = new_val
        return web.json_response({"status": "ok", "interpolation_period": model_delay})
    except:
        return web.json_response({"status": "error", "message": "invalid period"}, status=400)

async def init_app():
    """
    Регистрирует два POST‐роута:
      • /set_interval
      • /switch_installation
    """
    app = web.Application()
    app.router.add_post("/set_interval", set_interval_handler)
    return app

async def prediction_loop():
    while True:
        for task in tasks:
            batch, batch_true = task[1].load_batches() # Реальный запуск
            batch_filled, metrics = task[0].imputation(batch, batch_true)
            task[1].write_out(batch_filled, metrics)

        await asyncio.sleep(model_delay/1000)

if __name__ == "__main__":
    tasks = []

    # Реальный прогон для установок 1 и 2
    tasks.append((knn_model(), data_source("data_port_8092.csv", None, "data_out_8092.csv", "data_out_8092_long.csv", None)))
    tasks.append((knn_model(), data_source("data_port_8094.csv", None, "data_out_8094.csv", "data_out_8094_long.csv", None)))

    # Тестовый запуск с вычислением метрик
    tasks.append((knn_model(), data_source("data_port_8092.csv", "data_port_8093.csv", "data_out_8093.csv", "data_out_8093_long.csv", "data_metrics_8093.csv")))
    tasks.append((knn_model(), data_source("data_port_8094.csv", "data_port_8095.csv", "data_out_8095.csv", "data_out_8095_long.csv", "data_metrics_8095.csv")))

    loop = asyncio.get_event_loop()

    # 1) Стартуем цикл прогнозирования
    loop.create_task(prediction_loop())

    # 2) Запускаем HTTP‐API на порту 8000
    aioapp = loop.run_until_complete(init_app())
    runner = web.AppRunner(aioapp)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "localhost", 8000)
    loop.run_until_complete(site.start())

    # 4) Бесконечный цикл
    loop.run_forever()



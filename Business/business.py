from model import knn_model
from data_source import data_source

import os
import asyncio
from aiohttp import web

# интервал цикла (мс)
model_delay = int(os.getenv("MODEL_DELAY_MS", "3000"))

# где поднимаем HTTP API
BUSINESS_BIND = os.getenv("BUSINESS_BIND", "0.0.0.0")
BUSINESS_PORT = int(os.getenv("BUSINESS_PORT", "8000"))


# ----------------------
#  HTTP‐API для управления
# ----------------------
async def set_interval_handler(request):
    """
    POST /set_interval
    JSON {"period_ms": <int>}
    """
    global model_delay
    try:
        data = await request.json()
        new_val = int(data.get("period_ms"))
        print(f"[business] Recieved interval: {new_val}")
        if new_val < 100 or new_val > 60000:
            raise ValueError("period_ms out of range")
        model_delay = new_val
        return web.json_response({"status": "ok", "interpolation_period": model_delay})
    except Exception as e:
        return web.json_response({"status": "error", "message": f"invalid period: {e}"}, status=400)


async def health_handler(_request):
    return web.json_response({"status": "ok"})


async def init_app():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_post("/set_interval", set_interval_handler)
    return app


async def prediction_loop(tasks):
    global model_delay
    while True:
        try:
            for model, ds in tasks:
                batch, batch_true = ds.load_batches()

                # если данных ещё нет — просто ждём
                if batch is None or len(batch) == 0:
                    continue

                batch_filled, metrics = model.imputation(batch, batch_true)

                # важный момент: если модель вернула None — не пишем
                if batch_filled is None:
                    continue

                ds.write_out(batch_filled, metrics)

        except Exception as e:
            print(f"[business] prediction_loop error: {e}")

        await asyncio.sleep(model_delay / 1000)


if __name__ == "__main__":
    tasks = []

    # Реальный прогон для установок 1 и 2 (main: 8092, 8094)
    tasks.append((knn_model(), data_source(
        "data_port_8092.csv", None,
        "data_out_8092.csv", "data_out_8092_long.csv",
        None
    )))
    tasks.append((knn_model(), data_source(
        "data_port_8094.csv", None,
        "data_out_8094.csv", "data_out_8094_long.csv",
        None
    )))

    # Тестовый запуск с вычислением метрик (test: 8093, 8095)
    tasks.append((knn_model(), data_source(
        "data_port_8092.csv", "data_port_8093.csv",
        "data_out_8093.csv", "data_out_8093_long.csv",
        "data_metrics_8093.csv"
    )))
    tasks.append((knn_model(), data_source(
        "data_port_8094.csv", "data_port_8095.csv",
        "data_out_8095.csv", "data_out_8095_long.csv",
        "data_metrics_8095.csv"
    )))

    loop = asyncio.get_event_loop()

    # 1) Стартуем цикл прогнозирования
    loop.create_task(prediction_loop(tasks))

    # 2) Запускаем HTTP‐API
    aioapp = loop.run_until_complete(init_app())
    runner = web.AppRunner(aioapp)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, BUSINESS_BIND, BUSINESS_PORT)
    loop.run_until_complete(site.start())
    print(f"[business] HTTP API listening on http://{BUSINESS_BIND}:{BUSINESS_PORT}")

    loop.run_forever()
# Business/business.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from logging_setup import setup_logging, new_op_id

from model import knn_model
from data_source import data_source

import asyncio
from aiohttp import web

logger = setup_logging("business")

model_delay = int(os.getenv("MODEL_DELAY_MS", "3000"))
BIND_HOST = os.getenv("BUSINESS_BIND_HOST", "127.0.0.1")
BIND_PORT = int(os.getenv("BUSINESS_BIND_PORT", "8000"))


async def healthz_handler(request):
    op_id = new_op_id("health")
    logger.info("healthz ok", extra={"op_id": op_id})
    return web.json_response({"status": "ok", "service": "business", "op_id": op_id})


async def set_interval_handler(request):
    """
    POST /set_interval
    JSON {"period_ms": <int>}, меняет задержку обработки.
    """
    global model_delay
    op_id = new_op_id("set-interval")
    try:
        data = await request.json()
        new_val = int(data.get("period_ms"))
        if new_val < 100 or new_val > 60000:
            raise ValueError("period_ms out of range")

        old_val = model_delay
        model_delay = new_val

        logger.info(
            "interval updated old_ms=%s new_ms=%s",
            old_val,
            model_delay,
            extra={"op_id": op_id},
        )
        return web.json_response({"status": "ok", "interpolation_period": model_delay, "op_id": op_id})
    except Exception as e:
        logger.warning("invalid period payload err=%s", e, extra={"op_id": op_id})
        return web.json_response({"status": "error", "message": "invalid period", "op_id": op_id}, status=400)


async def init_app():
    app = web.Application()
    app.router.add_get("/healthz", healthz_handler)
    app.router.add_post("/set_interval", set_interval_handler)
    return app


def _df_shape(df):
    if df is None:
        return None
    try:
        return {"rows": int(df.shape[0]), "cols": int(df.shape[1])}
    except Exception:
        return None


async def prediction_loop(tasks):
    while True:
        cycle_op = new_op_id("cycle")
        try:
            logger.info("prediction cycle start delay_ms=%s", model_delay, extra={"op_id": cycle_op})

            for model, source in tasks:
                op_id = new_op_id("pred")
                batch, batch_true = source.load_batches(op_id=op_id)

                logger.info(
                    "batches loaded main=%s test=%s",
                    _df_shape(batch),
                    _df_shape(batch_true),
                    extra={"op_id": op_id},
                )

                batch_filled, metrics = model.imputation(batch, batch_true, op_id=op_id)
                source.write_out(batch_filled, metrics, op_id=op_id)

            logger.info("prediction cycle done", extra={"op_id": cycle_op})

        except Exception as e:
            logger.exception("prediction loop error err=%s", e, extra={"op_id": cycle_op})

        await asyncio.sleep(model_delay / 1000)


if __name__ == "__main__":
    tasks = []

    tasks.append((knn_model(), data_source("data_port_8092.csv", None, "data_out_8092.csv", "data_out_8092_long.csv", None)))
    tasks.append((knn_model(), data_source("data_port_8094.csv", None, "data_out_8094.csv", "data_out_8094_long.csv", None)))

    tasks.append((knn_model(), data_source("data_port_8092.csv", "data_port_8093.csv", "data_out_8093.csv", "data_out_8093_long.csv", "data_metrics_8093.csv")))
    tasks.append((knn_model(), data_source("data_port_8094.csv", "data_port_8095.csv", "data_out_8095.csv", "data_out_8095_long.csv", "data_metrics_8095.csv")))

    loop = asyncio.get_event_loop()

    loop.create_task(prediction_loop(tasks))

    aioapp = loop.run_until_complete(init_app())
    runner = web.AppRunner(aioapp)
    loop.run_until_complete(runner.setup())

    site = web.TCPSite(runner, BIND_HOST, BIND_PORT)
    loop.run_until_complete(site.start())

    logger.info("business api started bind=%s:%s", BIND_HOST, BIND_PORT, extra={"op_id": new_op_id("api-start")})
    loop.run_forever()

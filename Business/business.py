import time

from model import knn_model
from data_source import data_source

model_delay = 3000

if __name__ == "__main__":
    tasks = []

    # Реальный прогон для установок 1 и 2
    tasks.append((knn_model(), data_source("data_port_8092.csv", None, "data_out_8092.csv", None)))
    tasks.append((knn_model(), data_source("data_port_8094.csv", None, "data_out_8094.csv", None)))

    # Тестовый запуск с вычислением метрик
    tasks.append((knn_model(), data_source("data_port_8092.csv", "data_port_8093.csv", "data_out_8093.csv", "data_metrics_8093.csv")))
    tasks.append((knn_model(), data_source("data_port_8094.csv", "data_port_8095.csv", "data_out_8095.csv", "data_metrics_8095.csv")))

    while True:
        for task in tasks:
            batch, batch_true = task[1].load_batches() # Реальный запуск
            batch_filled, metrics = task[0].imputation(batch, batch_true)
            task[1].write_out(batch_filled, metrics)

        time.sleep(model_delay/1000)

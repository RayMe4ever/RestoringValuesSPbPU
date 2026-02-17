# Business/data_source.py
from __future__ import annotations

import os
import pandas as pd
import numpy as np

from logging_setup import setup_logging

logger = setup_logging("data_source")


class data_source:
    def __init__(self, path_main, path_test, path_out, path_out_long, path_metrics):
        self.path_main = path_main
        self.path_test = path_test
        self.path_out = path_out
        self.path_out_long = path_out_long
        self.path_metrics = path_metrics

        self.dir_reciever = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Reciever")
        self.dir_business = os.path.dirname(os.path.abspath(__file__))
        self.out_long = None

    def _safe_read_csv(self, path: str, op_id: str):
        if not os.path.exists(path):
            logger.warning("csv missing path=%s", path, extra={"op_id": op_id})
            return None
        try:
            df = pd.read_csv(path)
            logger.debug("csv loaded path=%s rows=%s cols=%s", path, df.shape[0], df.shape[1], extra={"op_id": op_id})
            return df
        except Exception as e:
            logger.exception("csv read failed path=%s err=%s", path, e, extra={"op_id": op_id})
            return None

    def load_batches(self, op_id: str):
        batch_main = None
        if self.path_main is not None:
            batch_main = self._safe_read_csv(os.path.join(self.dir_reciever, self.path_main), op_id=op_id)

        batch_test = None
        if self.path_test is not None:
            batch_test = self._safe_read_csv(os.path.join(self.dir_reciever, self.path_test), op_id=op_id)

        return batch_main, batch_test

    def write_out(self, batch, metrics, op_id: str):
        if self.path_out is not None and batch is not None:
            out_path = os.path.join(self.dir_business, self.path_out)
            out_path_long = os.path.join(self.dir_business, self.path_out_long)

            if self.out_long is None:
                self.out_long = batch.copy()

            self.out_long = (
                pd.concat([self.out_long, batch])
                .drop_duplicates(subset="DateTime", keep="last")
                .sort_values("DateTime")
                .tail(1000)
            )

            self.out_long.to_csv(out_path_long, index=False)
            batch.to_csv(out_path, index=False)

            logger.info(
                "out written out=%s out_long=%s rows=%s",
                out_path,
                out_path_long,
                batch.shape[0],
                extra={"op_id": op_id},
            )

        if self.path_metrics is not None and metrics is not None:
            metrics_list = [{k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}]
            metrics_df = pd.DataFrame(metrics_list)
            metrics_path = os.path.join(self.dir_business, self.path_metrics)
            metrics_df.to_csv(metrics_path, index=False)

            logger.info("metrics written path=%s keys=%s", metrics_path, list(metrics.keys()), extra={"op_id": op_id})

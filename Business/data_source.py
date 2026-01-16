import pandas as pd
import os
import numpy as np

class data_source:
    path_main = None
    path_test = None
    path_out = None
    path_metrics = None

    def __init__(self, path_main, path_test, path_out, path_out_long, path_metrics):
        self.path_main = path_main
        self.path_test = path_test
        self.path_out = path_out
        self.path_out_long = path_out_long
        self.path_metrics = path_metrics

        self.dir_reciever = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Reciever", "received_data")
        self.dir_business = os.path.dirname(os.path.abspath(__file__))
        self.out_long = None

    def load_batches(self):
        batch_main = None
        if self.path_main is not None:
            batch_main = pd.read_csv(os.path.join(self.dir_reciever, self.path_main))

        batch_test = None
        if self.path_test is not None:
            batch_test = pd.read_csv(os.path.join(self.dir_reciever, self.path_test))

        return batch_main, batch_test

    def write_out(self, batch, metrics):
        if self.path_out is not None and batch is not None:
            out_path = os.path.join(self.dir_business, self.path_out)
            out_path_long = os.path.join(self.dir_business, self.path_out_long)
            if self.out_long is None:
                self.out_long = batch.copy()

            self.out_long = pd.concat([self.out_long, batch]).drop_duplicates(subset="DateTime", keep="last").sort_values("DateTime").tail(1000)

            # Сохраняем
            self.out_long.to_csv(out_path_long, index=False)
            batch.to_csv(out_path, index=False)

        if self.path_metrics is not None:
            if metrics is not None:
                metrics_list = [{k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}]
                metrics_df = pd.DataFrame(metrics_list)
                metrics_df.to_csv(os.path.join(self.dir_business, self.path_metrics), index=False)





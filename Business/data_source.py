import pandas as pd
import os
import numpy as np

class data_source:
    path_main = None
    path_test = None
    path_out = None
    path_metrics = None

    def __init__(self, path_main, path_test, path_out, path_metrics):
        self.path_main = path_main
        self.path_test = path_test
        self.path_out = path_out
        self.path_metrics = path_metrics

        self.dir_reciever = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Reciever")
        self.dir_business = os.path.dirname(__file__)

    def load_batches(self):
        batch_main = None
        if self.path_main is not None:
            batch_main = pd.read_csv(os.path.join(self.dir_reciever, self.path_main))

        batch_test = None
        if self.path_test is not None:
            batch_test = pd.read_csv(os.path.join(self.dir_reciever, self.path_test))

        return batch_main, batch_test

    def write_out(self, batch, metrics):
        if self.path_out is not None:
            batch.to_csv(os.path.join(self.dir_business, self.path_out), index=False)

        if self.path_metrics is not None:
            metrics_list = [{k: float(v) if isinstance(v, np.floating) else v for k, v in metrics.items()}]
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(os.path.join(self.dir_business, self.path_metrics), index=False)





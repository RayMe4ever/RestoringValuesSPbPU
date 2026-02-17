import os
import pandas as pd


class DataSource:

    def __init__(self, dir_reciever, path_main=None, path_test=None):
        self.dir_reciever = dir_reciever
        self.path_main = path_main
        self.path_test = path_test

    # =========================
    # SAFE CSV READ
    # =========================
    def _safe_read_csv(self, path):
        full = os.path.join(self.dir_reciever, path)
        try:
            if not os.path.exists(full) or os.path.getsize(full) == 0:
                return None
            return pd.read_csv(full)
        except Exception as e:
            print(f"[data_source] read_csv failed {full}: {e}")
            return None

    # =========================
    # LOAD BATCHES
    # =========================
    def load_batches(self):
        batch_main = self._safe_read_csv(self.path_main) if self.path_main is not None else None
        batch_test = self._safe_read_csv(self.path_test) if self.path_test is not None else None
        return batch_main, batch_test
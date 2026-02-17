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
        if path is None:
            return None

        full = os.path.join(self.dir_reciever, path)

        try:
            # Файла ещё нет
            if not os.path.exists(full):
                return None

            # Файл есть, но пустой
            if os.path.getsize(full) == 0:
                return None

            return pd.read_csv(full)

        except Exception as e:
            print(f"[DataSource] read_csv error {full}: {e}")
            return None

    # =========================
    # LOAD BATCHES
    # =========================
    def load_batches(self):
        batch_main = self._safe_read_csv(self.path_main)
        batch_test = self._safe_read_csv(self.path_test)
        return batch_main, batch_test
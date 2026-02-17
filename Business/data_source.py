import os
import tempfile
import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd


class data_source:
    """Источник данных для Business.

    Поддерживает новый вызов из Business/business.py:
      data_source(path_main, path_test, path_out, path_out_long, path_metrics)
    и legacy-вызов:
      data_source(dir_reciever, path_main=None, path_test=None)

    Выходные CSV пишутся атомарно: temp-file в той же директории + os.replace().
    """

    def __init__(
        self,
        arg1: str,
        arg2: Optional[str] = None,
        arg3: Optional[str] = None,
        arg4: Optional[str] = None,
        arg5: Optional[str] = None,
    ) -> None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_reciever_dir = os.path.join(repo_root, "Reciever")
        default_business_dir = os.path.join(repo_root, "Business")

        # Legacy: первый аргумент — директория (и это не *.csv)
        candidate_dir = arg1
        if not os.path.isabs(candidate_dir):
            candidate_dir = os.path.abspath(os.path.join(repo_root, candidate_dir))

        is_legacy = (arg1 is not None) and (not str(arg1).lower().endswith(".csv")) and os.path.isdir(candidate_dir)

        if is_legacy:
            # legacy: (dir_reciever, path_main, path_test)
            self.dir_reciever = candidate_dir
            self.dir_business = default_business_dir
            self.path_main = arg2
            self.path_test = arg3
            self.path_out = None
            self.path_out_long = None
            self.path_metrics = None
        else:
            # new: (path_main, path_test, path_out, path_out_long, path_metrics)
            self.dir_reciever = default_reciever_dir
            self.dir_business = default_business_dir
            self.path_main = arg1
            self.path_test = arg2
            self.path_out = arg3
            self.path_out_long = arg4
            self.path_metrics = arg5

        self._out_long_cache: Optional[pd.DataFrame] = None
        self._metrics_cache: Optional[pd.DataFrame] = None

    def _resolve_in_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        return path if os.path.isabs(path) else os.path.join(self.dir_reciever, path)

    def _resolve_out_path(self, path: Optional[str]) -> Optional[str]:
        if path is None:
            return None
        return path if os.path.isabs(path) else os.path.join(self.dir_business, path)

    def _safe_read_csv(self, full_path: str) -> Optional[pd.DataFrame]:
        try:
            if not os.path.exists(full_path) or os.path.getsize(full_path) == 0:
                return None
            return pd.read_csv(full_path)
        except Exception as e:
            print(f"[data_source] read_csv failed {full_path}: {e}")
            return None

    def load_batches(self) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        in_main = self._resolve_in_path(self.path_main)
        in_test = self._resolve_in_path(self.path_test)

        batch_main = self._safe_read_csv(in_main) if in_main else None
        batch_test = self._safe_read_csv(in_test) if in_test else None
        return batch_main, batch_test

    def _atomic_write_df(self, df: pd.DataFrame, dest_path: str) -> None:
        """Атомарная запись DataFrame в CSV через temp + os.replace.

        os.replace: при успехе rename является атомарной операцией (POSIX requirement),
        но может упасть, если src/dst на разных файловых системах. citeturn0search0
        """
        dest_dir = os.path.dirname(dest_path) or "."
        os.makedirs(dest_dir, exist_ok=True)

        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=dest_dir, text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
                df.to_csv(f, index=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, dest_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise

    @staticmethod
    def _normalize_metrics(metrics: Any, batch: Optional[pd.DataFrame]) -> Optional[Dict[str, Any]]:
        if metrics is None:
            return None
        if not isinstance(metrics, dict):
            return {"metrics": str(metrics)}

        row: Dict[str, Any] = {}
        for k, v in metrics.items():
            try:
                import numpy as np  # type: ignore
                if isinstance(v, np.generic):
                    v = v.item()
            except Exception:
                pass
            row[str(k)] = v

        if batch is not None and (not batch.empty) and ("DateTime" in batch.columns):
            row.setdefault("DateTime", batch["DateTime"].iloc[-1])
        else:
            row.setdefault("DateTime", time.strftime("%Y-%m-%dT%H:%M:%S"))
        return row

    @staticmethod
    def _cap_long_df(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if "DateTime" in df.columns:
            df = df.drop_duplicates(subset=["DateTime"], keep="last")
            df = df.sort_values("DateTime")
        if len(df) > max_rows:
            df = df.tail(max_rows)
        return df

    def write_out(self, batch: Optional[pd.DataFrame], metrics: Any) -> None:
        if batch is None:
            return

        out_path = self._resolve_out_path(self.path_out)
        if out_path:
            self._atomic_write_df(batch, out_path)

        out_long_path = self._resolve_out_path(self.path_out_long)
        if out_long_path:
            if self._out_long_cache is None:
                existing = self._safe_read_csv(out_long_path)
                self._out_long_cache = existing if existing is not None else pd.DataFrame()
            self._out_long_cache = pd.concat([self._out_long_cache, batch], ignore_index=True)
            self._out_long_cache = self._cap_long_df(self._out_long_cache, max_rows=1000)
            self._atomic_write_df(self._out_long_cache, out_long_path)

        metrics_path = self._resolve_out_path(self.path_metrics)
        row = self._normalize_metrics(metrics, batch)
        if metrics_path and row is not None:
            if self._metrics_cache is None:
                existing = self._safe_read_csv(metrics_path)
                self._metrics_cache = existing if existing is not None else pd.DataFrame()
            self._metrics_cache = pd.concat([self._metrics_cache, pd.DataFrame([row])], ignore_index=True)
            self._metrics_cache = self._cap_long_df(self._metrics_cache, max_rows=1000)
            self._atomic_write_df(self._metrics_cache, metrics_path)

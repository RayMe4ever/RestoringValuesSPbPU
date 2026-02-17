import pandas as pd
import numpy as np


class knn_model:
    mape_inter = []
    mape_mean = []

    # минимальный размер батча, но теперь мы НЕ блокируем вывод при меньшем размере
    batch_size = 10

    def time_based_knn_impute(self, df, target_col, time_col="DateTime", k=3):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df["TimeNumeric"] = (df[time_col] - df[time_col].min()).dt.total_seconds()

        for idx in df[df[target_col].isna()].index:
            time_i = df.loc[idx, "TimeNumeric"]
            known = df[df[target_col].notna()].copy()

            if known.empty:
                continue

            known["TimeDiff"] = np.abs(known["TimeNumeric"] - time_i)
            neighbors = known.nsmallest(k, "TimeDiff")

            weights = 1 / (neighbors["TimeDiff"] + 1e-5)
            imputed_value = np.average(neighbors[target_col], weights=weights)
            df.at[idx, target_col] = imputed_value

        return df.drop(columns=["TimeNumeric"], errors="ignore")

    def compare_fill_methods_and_calculate_mape_knn(self, batch, original_batch=None, k=3):
        batch_interpolation = batch.copy()
        batch_mean_fill = batch.copy()

        interpolation_errors = []
        mean_fill_errors = []

        is_test = original_batch is not None

        # приводим время
        batch_interpolation["TimeNumeric"] = pd.to_datetime(batch_interpolation.iloc[:, 0], errors="coerce")
        batch_interpolation["TimeNumeric"] = (
            batch_interpolation["TimeNumeric"] - batch_interpolation["TimeNumeric"].min()
        ).dt.total_seconds()

        for col_idx, col in enumerate(batch.columns[1:], start=1):
            for idx in range(len(batch)):
                if pd.isna(batch.iloc[idx, col_idx]):
                    original_value = None if not is_test else original_batch.iloc[idx, col_idx]
                    interpolated_value = None

                    # Простая интерполяция (между соседями)
                    if 0 < idx < len(batch) - 1:
                        prev_val = batch_interpolation.iloc[idx - 1, col_idx]
                        next_val = batch_interpolation.iloc[idx + 1, col_idx]
                        if not pd.isna(prev_val) and not pd.isna(next_val):
                            interpolated_value = (prev_val + next_val) / 2
                            batch_interpolation.iat[idx, col_idx] = interpolated_value
                            if is_test and original_value not in (None, 0):
                                interpolation_errors.append(abs((original_value - interpolated_value) / original_value))

                    # KNN по времени
                    if interpolated_value is None:
                        temp_df = batch_interpolation[[batch.columns[0], col]].rename(columns={batch.columns[0]: "DateTime"})
                        temp_df = self.time_based_knn_impute(temp_df, target_col=col, time_col="DateTime", k=k)
                        knn_value = temp_df.loc[idx, col]
                        batch_interpolation.iat[idx, col_idx] = knn_value
                        interpolated_value = knn_value
                        if is_test and original_value not in (None, 0):
                            interpolation_errors.append(abs((original_value - knn_value) / original_value))

                    # mean fill (только в test-режиме, для метрики)
                    if is_test:
                        mean_value = batch_mean_fill.iloc[:, col_idx].mean(skipna=True)
                        batch_mean_fill.iat[idx, col_idx] = mean_value
                        if original_value not in (None, 0):
                            mean_fill_errors.append(abs((original_value - mean_value) / original_value))

        batch_interpolation.drop(columns=["TimeNumeric"], inplace=True, errors="ignore")

        if is_test:
            mape_interpolation = np.mean(interpolation_errors) if interpolation_errors else None
            mape_mean_fill = np.mean(mean_fill_errors) if mean_fill_errors else None
            return batch_interpolation, mape_interpolation, mape_mean_fill

        return batch_interpolation, None, None

    def imputation(self, batch, batch_true=None):
        """
        Важно: раньше при batch < batch_size возвращалось None, из-за чего out/out_long не писались
        и график "Заполнение пропусков" начинался поздно.
        Теперь: при малом батче возвращаем батч как есть (passthrough), чтобы визуализация шла с начала.
        """
        if batch is None or batch.empty:
            return None, None

        if batch.shape[0] < self.batch_size:
            # просто отдаём входной батч, чтобы business.write_out писал out_long с первых секунд
            return batch.copy(), None

        batch_interpolation, inter, mean = self.compare_fill_methods_and_calculate_mape_knn(batch, batch_true)

        if inter is not None:
            self.mape_inter.append(inter)
        if mean is not None:
            self.mape_mean.append(mean)

        metrics = None
        if batch_true is not None:
            metrics = {}
            if self.mape_inter:
                metrics["MAPE"] = float(sum(self.mape_inter) / len(self.mape_inter))
            if self.mape_mean:
                metrics["MAPE_mean"] = float(sum(self.mape_mean) / len(self.mape_mean))
            if metrics.get("MAPE") and metrics.get("MAPE_mean"):
                metrics["improvement"] = float(metrics["MAPE_mean"] / metrics["MAPE"])

        return batch_interpolation, metrics
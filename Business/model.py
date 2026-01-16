import pandas as pd
import numpy as np

class knn_model:
    mape_inter = []
    mape_mean = []

    batch_size = 10

    def time_based_knn_impute(self, df, target_col, time_col='DateTime', k=3):
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df['TimeNumeric'] = (df[time_col] - df[time_col].min()).dt.total_seconds()  # время в секундах

        for idx in df[df[target_col].isna()].index:
            time_i = df.loc[idx, 'TimeNumeric']

            # Найдём все строки с непустым значением
            known = df[df[target_col].notna()].copy()

            # Считаем расстояние по времени
            known['TimeDiff'] = np.abs(known['TimeNumeric'] - time_i)

            # Берём k ближайших по времени
            neighbors = known.nsmallest(k, 'TimeDiff')

            # Взвешенное среднее по времени (меньше время — больше вес)
            weights = 1 / (neighbors['TimeDiff'] + 1e-5)  # чтоб не делить на ноль
            imputed_value = np.average(neighbors[target_col], weights=weights)

            # Подставляем
            df.at[idx, target_col] = imputed_value

        return df.drop(columns=['TimeNumeric'], errors='ignore')
    def compare_fill_methods_and_calculate_mape_knn(self, batch, original_batch=None, k=3):
        """
        Заполнение пропусков:
        - В режиме 'test': интерполяция + KNN по времени + MAPE + лог
        - В режиме 'standard': всё то же самое, но без логов и расчета метрик

        :param batch: DataFrame с пропущенными значениями
        :param original_batch: Оригинальный DataFrame без пропусков
        :return: batch_interpolation (заполненный), mape_interpolation, mape_mean_fill
        """
        batch_interpolation = batch.copy()
        batch_mean_fill = batch.copy()

        interpolation_errors = []
        mean_fill_errors = []

        batch_interpolation["TimeNumeric"] = pd.to_datetime(batch_interpolation.iloc[:, 0])
        batch_interpolation["TimeNumeric"] = (batch_interpolation["TimeNumeric"] - batch_interpolation["TimeNumeric"].min()).dt.total_seconds()

        is_test = not(original_batch is None)

        for col_idx, col in enumerate(batch.columns[1:], start=1):
            for idx in range(len(batch)):
                if pd.isna(batch.iloc[idx, col_idx]):
                    original_value = None if not is_test else original_batch.iloc[idx, col_idx]
                    interpolated_value = None

                    # Интерполяция
                    if 0 < idx < len(batch) - 1:
                        prev_val = batch_interpolation.iloc[idx - 1, col_idx]
                        next_val = batch_interpolation.iloc[idx + 1, col_idx]
                        if not pd.isna(prev_val) and not pd.isna(next_val):
                            interpolated_value = (prev_val + next_val) / 2
                            batch_interpolation.iat[idx, col_idx] = interpolated_value
                            if is_test:
                                #print(f"Интерполяция: строка {idx}, столбец {col_idx}, значение {interpolated_value:.4f}")
                                if original_value != 0:
                                    interpolation_errors.append(abs((original_value - interpolated_value) / original_value))

                    # KNN по времени
                    if interpolated_value is None:
                        temp_df = batch_interpolation[[batch.columns[0], col]].rename(columns={batch.columns[0]: "DateTime"})
                        temp_df = self.time_based_knn_impute(temp_df, target_col=col, time_col="DateTime", k=k)
                        knn_value = temp_df.loc[idx, col]
                        batch_interpolation.iat[idx, col_idx] = knn_value
                        interpolated_value = knn_value
                        if is_test:
                            #print(f"KNN (по времени): строка {idx}, столбец {col_idx}, значение {knn_value:.4f}")
                            if original_value != 0:
                                interpolation_errors.append(abs((original_value - knn_value) / original_value))
                    if is_test:
                        # Заполнение средним значением
                        mean_value = batch_mean_fill.iloc[:, col_idx].mean(skipna=True)
                        batch_mean_fill.iat[idx, col_idx] = mean_value
                    if is_test and original_value != 0:
                        mean_fill_errors.append(abs((original_value - mean_value) / original_value))

        batch_interpolation.drop(columns=["TimeNumeric"], inplace=True, errors='ignore')

        if is_test:
            mape_interpolation = np.mean(interpolation_errors) if interpolation_errors else None
            mape_mean_fill = np.mean(mean_fill_errors) if mean_fill_errors else None
            return batch_interpolation, mape_interpolation, mape_mean_fill
        else:
            return batch_interpolation, None, None

    def imputation(self, batch, batch_true=None):
        # Выполняем заполнение
        if batch.shape[0] < self.batch_size:
            print("Недостаточно данных! Сидим, не рыпаемся...")
            return None, None

        batch_interpolation, inter, mean = self.compare_fill_methods_and_calculate_mape_knn(batch, batch_true)
        if inter is not None:
            self.mape_inter.append(inter)
        if mean is not None:
            self.mape_mean.append(mean)

        # Показываем именно batch с пропущенными значениями
        #print("\n📄 Батч с пропущенными значениями:")
        #print(batch)
        #print("\n📄 Батч с заполненными значениями:")
        #print(batch_interpolation)

        metrics = None
        # Финальный отчёт
        if batch_true is None:
            pass
            #print(f'\n✅ Успешное завершение задачи')
        else:
            # Безопасный вывод метрик
            metrics = {}

            inter_str = f"{inter:.4f}" if inter is not None else "нет данных"
            mean_str = f"{mean:.4f}" if mean is not None else "нет данных"
            #print(
            #    f"\n📊 Метрики для текущего батча:\n  🔹 MAPE (интерполяция/KNN): {inter_str}\n  🔹 MAPE (среднее): {mean_str}")

            if len(self.mape_inter) != 0:
                #print(f'\n📈 Средняя ошибка модели (MAPE): {sum(self.mape_inter) / len(self.mape_inter):.6f}')
                metrics["MAPE"] = sum(self.mape_inter) / len(self.mape_inter)
            if len(self.mape_mean) != 0:
                #print(f'📉 Средняя ошибка при заполнении средним (MAPE): {sum(self.mape_mean) / len(self.mape_mean):.6f}')
                metrics["MAPE_mean"] = sum(self.mape_mean) / len(self.mape_mean)

            if (len(self.mape_mean) != 0) and (len(self.mape_inter) != 0):
                #print(
                #f'🎯 Наша модель лучше в {round((sum(self.mape_mean) / len(self.mape_mean)) / (sum(self.mape_inter) / len(self.mape_inter)), 1)} раз!')
                metrics["improvement"] = round((sum(self.mape_mean) / len(self.mape_mean)) / (sum(self.mape_inter) / len(self.mape_inter)), 3)

        return batch_interpolation, metrics



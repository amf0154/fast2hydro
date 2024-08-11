import os

import joblib
import numpy as np
import pandas as pd
from scipy.fft import fft
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import time  # Добавлено для измерения времени
from datetime import timedelta  # Для удобного отображения времени


def detect_frequency(df, date_column):
    """
    Определяет частоту временного ряда на основе столбца дат.

    Параметры:
    df (DataFrame): Датафрейм с временным рядом.
    date_column (str): Название столбца с датами.

    Возвращает:
    str: Частота временного ряда.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    freq = pd.infer_freq(df[date_column])
    if freq == "T":
        freq = "1min"
    return freq


def detect_seasonal_period(df, date_column, value_column):
    """
    Определяет сезонный период временного ряда с использованием FFT (Быстрое преобразование Фурье).

    Параметры:
    df (DataFrame): Датафрейм с временным рядом.
    date_column (str): Название столбца с датами.
    value_column (str): Название столбца со значениями.

    Возвращает:
    int: Сезонный период.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    freq = detect_frequency(df, date_column)
    if freq is None:
        raise ValueError("Не удалось определить частоту временного ряда.")

    y = df[value_column].values
    n = len(y)

    fft_result = fft(y)
    fft_amplitude = np.abs(fft_result)

    dominant_frequency = np.argmax(fft_amplitude[1 : n // 2]) + 1
    seasonal_period = n / dominant_frequency

    if seasonal_period <= 1:
        raise ValueError("Автоматически определенный сезонный период меньше или равен 1. Укажите значение вручную.")

    return int(seasonal_period)


class HoltWintersPressureModel:
    """
    Класс для модели прогнозирования давления с использованием метода Хольта-Винтерса.
    """

    def __init__(self):
        self.model = None
        self.original_freq = None
        self.aggregated_freq = None
        self.seasonal_periods = None
        self.trend = None
        self.seasonal = None
        self.training_start_date = None
        self.training_end_date = None
        self.num_original_data_points = None
        self.training_duration = None  # Временя обучения

    @staticmethod
    def load_data(file_path):
        """
        Загружает данные из файла и проверяет их формат.

        Параметры:
        file_path (str): Путь к файлу.

        Возвращает:
        DataFrame: Загруженный датафрейм.
        str: Название столбца с датами.
        str: Название столбца со значениями.
        """
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Поддерживаются только файлы формата CSV и Excel.")

        if df.shape[1] != 2:
            raise ValueError("Файл должен содержать две колонки: дата и значение.")

        date_column, value_column = df.columns
        df[date_column] = pd.to_datetime(df[date_column], errors="coerce")

        if df[date_column].isnull().any():
            raise ValueError("Некоторые значения в колонке дат не могут быть преобразованы в формат даты.")

        return df, date_column, value_column

    @staticmethod
    def aggregate_data(df, date_column, value_column, freq="5min"):
        """
        Агрегирует данные по заданной частоте и заполняет пропуски.

        Параметры:
        df (DataFrame): Исходный датафрейм.
        date_column (str): Название столбца с датами.
        value_column (str): Название столбца со значениями.
        freq (str): Частота агрегации данных ('5min', '10min', '30min', '1H'])

        Возвращает:
        DataFrame: Агрегированный датафрейм.
        """
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        aggregated_df = df.resample(freq).mean()
        aggregated_df[value_column] = aggregated_df[value_column].interpolate()
        return aggregated_df.reset_index()

    def get_aggregation_factor(self):
        """
        Возвращает коэффициент агрегации на основе исходной и текущей частот.

        Возвращает:
        float: Коэффициент агрегации.
        """
        if isinstance(self.aggregated_freq, str):
            if self.aggregated_freq.endswith("min"):
                try:
                    freq_minutes = int(self.aggregated_freq[:-3])
                except ValueError:
                    freq_minutes = 1
            elif self.aggregated_freq == "T":
                freq_minutes = 1
            else:
                freq_minutes = pd.to_timedelta(self.aggregated_freq).seconds // 60

            if self.original_freq.endswith("min"):
                try:
                    original_freq_minutes = int(self.original_freq[:-3])
                except ValueError:
                    original_freq_minutes = 1
            elif self.original_freq == "T":
                original_freq_minutes = 1
            else:
                original_freq_minutes = pd.to_timedelta(self.original_freq).seconds // 60

            return original_freq_minutes / freq_minutes
        return 1

    def train_holt_winters_model(
        self, df, date_column, value_column, seasonal_periods=None, trend=None, seasonal="add", aggregation_freq=None
    ):
        """
        Обучает модель Хольта-Винтерса на данных.

        Параметры:
        df (DataFrame): Исходный датафрейм.
        date_column (str): Название столбца с датами.
        value_column (str): Название столбца со значениями.
        seasonal_periods (int, optional): Сезонные периоды.
        trend (str, optional): Тип тренда ('add', 'mul' или None).
        seasonal (str, optional): Тип сезонности ('add' или 'mul').
        aggregation_freq (str, optional): Частота агрегации данных.

        Возвращает:
        ExponentialSmoothing: Обученная модель.
        """
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        self.original_freq = detect_frequency(df, date_column)

        original_data_len = len(df)
        self.num_original_data_points = original_data_len
        self.training_start_date = df[date_column].min()
        self.training_end_date = df[date_column].max()

        if aggregation_freq:
            df = self.aggregate_data(df, date_column, value_column, freq=aggregation_freq)
            self.aggregated_freq = aggregation_freq
        else:
            self.aggregated_freq = self.original_freq

        y = df.set_index(date_column)[value_column].asfreq(self.aggregated_freq)
        y = y.interpolate()

        if seasonal_periods is None:
            seasonal_periods = detect_seasonal_period(df, date_column, value_column)

        # Проверка на достаточность данных для двух полных сезонных циклов
        aggregation_factor = self.get_aggregation_factor()
        min_required_data_points = 2 * seasonal_periods / aggregation_factor
        if original_data_len < min_required_data_points:
            raise ValueError(
                f"Недостаточно данных для построения модели. "
                f"Необходимо не менее двух полных сезонных циклов с учетом частоты агрегации "
                f"(более {int(min_required_data_points)} точек данных)."
            )

        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal

        start_time = time.time()  # Запуск таймера

        self.model = ExponentialSmoothing(
            y, seasonal_periods=seasonal_periods, trend=trend, seasonal=seasonal, initialization_method="estimated"
        ).fit()

        end_time = time.time()  # Остановка таймера
        duration = timedelta(seconds=(end_time - start_time))  # Сохранение длительности обучения
        self.training_duration = str(duration)  # Сохранение длительности обучения в формате HH:MM:SS
        return self.model

    def save_model(self, file_path):
        """
        Сохраняет модель в файл.

        Параметры:
        file_path (str): Путь к файлу для сохранения модели.
        """
        if self.model is not None:
            with open(file_path, "wb") as f:
                joblib.dump(
                    {
                        "model": self.model,
                        "original_freq": self.original_freq,
                        "aggregated_freq": self.aggregated_freq,
                        "seasonal_periods": self.seasonal_periods,
                        "trend": self.trend,
                        "seasonal": self.seasonal,
                        "training_start_date": self.training_start_date,
                        "training_end_date": self.training_end_date,
                        "num_original_data_points": self.num_original_data_points,
                        "training_duration": self.training_duration,
                    },
                    f,
                )
        else:
            raise ValueError("Модель еще не создана. Сначала обучите модель.")

    def load_model(self, file_path):
        """
        Загружает модель из файла.

        Параметры:
        file_path (str): Путь к файлу с моделью.

        Вызывает:
        FileNotFoundError: Если файл модели не найден.
        """
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = joblib.load(f)
                self.model = data["model"]
                self.original_freq = data["original_freq"]
                self.aggregated_freq = data["aggregated_freq"]
                self.seasonal_periods = data["seasonal_periods"]
                self.trend = data["trend"]
                self.seasonal = data["seasonal"]
                self.training_start_date = data["training_start_date"]
                self.training_end_date = data["training_end_date"]
                self.num_original_data_points = data["num_original_data_points"]
                self.training_duration = data["training_duration"]

        else:
            raise FileNotFoundError(f"Модель не найдена {file_path}")

    def predict(self, start, end, restore_freq=True, interpolation_method="linear"):
        """
        Делает прогноз с использованием обученной модели.

        Параметры:
        start (str): Дата начала прогноза.
        end (str): Дата окончания прогноза.
        restore_freq (bool, optional): Восстанавливать ли исходную частоту данных.
        interpolation_method (str, optional): Метод интерполяции для восстановления частоты.

        Возвращает:
        DataFrame: Датафрейм с двумя столбцами: 'date' и 'values'.
        """
        if self.model is not None:
            predictions = self.model.predict(start=start, end=end)

            if restore_freq and self.original_freq != self.aggregated_freq:
                predictions = predictions.resample(self.original_freq).interpolate(method=interpolation_method)

            predictions = predictions.round(0)

            # Создаем DataFrame с двумя столбцами
            prediction_df = pd.DataFrame({"date": predictions.index, "values": predictions.values})

            return prediction_df
        else:
            raise ValueError("Модель еще не обучена.")

    def get_model_summary(self):
        """
        Возвращает резюме модели, если модель обучена.

        Возвращает:
        dict: Резюме модели.
        """
        if self.model is not None:
            summary = {
                "AIC": self.model.aic,
                "BIC": self.model.bic,
                "SSE": self.model.sse,
                "Training Start Date": str(self.training_start_date),
                "Training End Date": str(self.training_end_date),
                "Number of Original Data Points": self.num_original_data_points,
                "Original Frequency": self.original_freq,
                "Aggregated Frequency": self.aggregated_freq,
                "Seasonal Periods": self.seasonal_periods,
                "Trend": self.trend,
                "Seasonal": self.seasonal,
                "Training Duration": str(self.training_duration),  # Время обучения
            }
            return summary
        else:
            return {"error": "Модель еще не обучена. Пожалуйста, обучите модель."}

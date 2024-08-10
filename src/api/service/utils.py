import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np


def calculate_metrics(actual_file_path, predicted_file_path):
    """
    Рассчитывает различные виды погрешностей между реальными и прогнозируемыми данными.

    Параметры:
    actual_file_path (str): Путь к файлу с реальными данными.
    predicted_file_path (str): Путь к файлу с прогнозируемыми данными.

    Возвращает:
    dict: Словарь с различными видами погрешностей.
    """
    if actual_file_path.endswith((".xlsx", ".xls")):
        actual_df = pd.read_excel(actual_file_path, parse_dates=[0], usecols=[0, 1])
    else:
        actual_df = pd.read_csv(actual_file_path, parse_dates=[0], usecols=[0, 1])

    if predicted_file_path.endswith((".xlsx", ".xls")):
        predicted_df = pd.read_excel(predicted_file_path, parse_dates=[0], usecols=[0, 1])
    else:
        predicted_df = pd.read_csv(predicted_file_path, parse_dates=[0], usecols=[0, 1])

    date_column = actual_df.columns[0]

    # Переименование столбцов для единообразия
    actual_df.columns = [date_column, "actual"]
    predicted_df.columns = [date_column, "predicted"]

    merged_df = pd.merge(actual_df, predicted_df, on=date_column)

    if merged_df.empty:
        raise ValueError("Не найдено совпадений по дате между реальными и прогнозируемыми данными.")

    # Удаление строк с пропущенными значениями
    merged_df.dropna(inplace=True)

    if merged_df.empty:
        raise ValueError("Все совпадающие данные имеют пропуски, невозможно рассчитать погрешности.")

    actual_values = merged_df["actual"]
    predicted_values = merged_df["predicted"]

    errors = {
        "mean_absolute_error": mean_absolute_error(actual_values, predicted_values).round(2),
        "root_mean_squared_error": np.sqrt(mean_squared_error(actual_values, predicted_values)).round(2),
        "mean_absolute_percentage_error": mean_absolute_percentage_error(actual_values, predicted_values).round(2),
    }

    return errors

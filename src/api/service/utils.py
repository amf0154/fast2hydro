import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np


def calculate_metrics_from_excel(actual_file_path, predicted_file_path):
    """
    Рассчитывает различные виды погрешностей между реальными и прогнозируемыми данными на основе файлов.

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


def calculate_metrics_from_json(real_data, predictions_data):
    """
    Рассчитывает различные виды погрешностей между реальными и прогнозируемыми данными на основе JSON данных.

    Параметры:
    actual_data (list of dict): JSON данные с реальными значениями.
    predicted_data (list of dict): JSON данные с прогнозируемыми значениями.

    Возвращает:
    dict: Словарь с различными видами погрешностей.
    """
    actual_df = pd.DataFrame(real_data)
    predicted_df = pd.DataFrame(predictions_data)

    date_column = actual_df.columns[0]

    # Предполагается, что дата в формате "день-месяц-год часы:минуты"
    actual_df[date_column] = pd.to_datetime(actual_df[date_column], format="%d-%m-%Y %H:%M")
    predicted_df[date_column] = pd.to_datetime(predicted_df[date_column], format="%d-%m-%Y %H:%M")

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

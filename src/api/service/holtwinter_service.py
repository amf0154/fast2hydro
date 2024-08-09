import os

import pandas as pd

from ..models.holtwinter_model import HoltWintersPressureModel
from ..schemas.holtwinter_schemas import ModelSummary


def train_holtwinters_model(
    file_path, model_name=None, seasonal_periods=None, trend=None, seasonal="add", aggregation_freq=None
):
    model = HoltWintersPressureModel()
    df, date_column, value_column = model.load_data(file_path)
    model.train_holt_winters_model(df, date_column, value_column, seasonal_periods, trend, seasonal, aggregation_freq)

    model_folder = os.path.join("stored_models", "holtwinter")

    os.makedirs(model_folder, exist_ok=True)

    if not model_name:
        model_name = f"holtwinter_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

    model_file_path = os.path.join(model_folder, f"{model_name}.pkl")
    model.save_model(model_file_path)


def save_predictions_to_excel(predictions, output_file_path):
    """
    Сохраняет предсказания в файл Excel.

    Параметры:
    predictions (DataFrame): Датафрейм с предсказаниями.
    output_file_path (str): Путь к файлу для сохранения предсказаний.
    """
    predictions.to_excel(output_file_path, index=False)


def predict_holtwinters_model(model_file_path, start, end, restore_freq=True, output_format="json"):
    model = HoltWintersPressureModel()
    model.load_model(model_file_path)
    predictions = model.predict(start, end, restore_freq)

    # Получение имени модели из пути к файлу модели
    model_name = os.path.splitext(os.path.basename(model_file_path))[0]

    if output_format == "excel":
        # Формируем путь к файлу для сохранения предсказаний
        model_folder = os.path.join("src", "api", "result", "holtwinter")
        output_file_path = os.path.join(
            model_folder, f"{model_name}_predictions_{start.replace(':', '-')}_{end.replace(':', '-')}.xlsx"
        )
        save_predictions_to_excel(predictions, output_file_path)
        return {"message": f"Predictions saved to {output_file_path}"}
    else:
        # Преобразование DataFrame в список словарей для JSON
        predictions_dict = predictions.to_dict("records")
        return {"predictions": predictions_dict}


def load_model_summary(model_file_path):
    model = HoltWintersPressureModel()
    model.load_model(model_file_path)

    model_summary = ModelSummary(
        aic=round(model.model.aic),
        bic=round(model.model.bic),
        sse=round(model.model.sse),
        training_start_date=str(model.training_start_date),
        training_end_date=str(model.training_end_date),
        num_original_data_points=model.num_original_data_points,
        original_freq=model.original_freq,
        aggregated_freq=model.aggregated_freq,
        seasonal_periods=model.seasonal_periods,
        trend=model.trend,
        seasonal=model.seasonal,
    )

    return model_summary

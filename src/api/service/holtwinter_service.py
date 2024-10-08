"""
Этот модуль предоставляет функции для обучения, предсказания и получения сводной информации о модели Хольта-Винтерса:

1. **train_holtwinters_model** - Обучает модель Хольта-Винтерса на основе предоставленного файла с данными,
сохраняет модель на диск и возвращает путь к сохраненной модели.
2. **save_predictions_to_excel** - Сохраняет предсказания в файл Excel по указанному пути.
3. **predict_holtwinters_model** - Выполняет предсказания на основе загруженной модели, возвращая
результаты в формате JSON или Excel, в зависимости от запроса.
4. **load_model_summary** - Загружает модель и возвращает сводную информацию,
такую как AIC, BIC, SSE, даты обучения, количество точек данных и другие параметры.

Функции помогают управлять процессом машинного обучения и предсказаний, обеспечивая
сохранение моделей и результатов в удобном формате.
"""

import os
import pandas as pd

from ..models.holtwinter_model import HoltWintersPressureModel
from ..schemas.holtwinter_schemas import ModelSummary
from fastapi import HTTPException

# def train_holtwinters_model(
#     file_path, model_name=None, seasonal_periods=None, trend=None, seasonal="add", aggregation_freq=None
# ):
#     model = HoltWintersPressureModel()
#     df, date_column, value_column = model.load_data(file_path)
#     model.train_holt_winters_model(df, date_column, value_column, seasonal_periods, trend, seasonal, aggregation_freq)
#
#     model_folder = os.path.join("stored_models", "holtwinter")
#
#     os.makedirs(model_folder, exist_ok=True)
#
#     if not model_name:
#         model_name = f"holtwinter_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
#
#     model_file_path = os.path.join(model_folder, f"{model_name}.pkl")
#     model.save_model(model_file_path)
#
#     return model_file_path


def train_holtwinters_model(
    file_path, model_name=None, seasonal_periods=None, trend=None, seasonal="add", aggregation_freq=None
):
    try:
        # Создаем экземпляр модели и загружаем данные
        model = HoltWintersPressureModel()
        df, date_column, value_column = model.load_data(file_path)

        # Обучаем модель
        model.train_holt_winters_model(
            df, date_column, value_column, seasonal_periods, trend, seasonal, aggregation_freq
        )

        # Генерируем имя модели, если оно не задано
        if not model_name:
            model_name = f"holtwinter_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

        # Создаем папку для хранения модели, если она не существует
        model_folder = os.path.join("stored_models", "holtwinter")
        os.makedirs(model_folder, exist_ok=True)
        model_file_path = os.path.join(model_folder, f"{model_name}.pkl")

        # Сохраняем модель в файл
        model.save_model(model_file_path)
        return model_file_path

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Файл не найден. Проверьте путь к файлу.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при загрузке или обучении данных: {str(e)}")
    except IOError:
        raise HTTPException(status_code=500, detail="Ошибка при сохранении модели. Проверьте доступ к файлам.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Произошла непредвиденная ошибка: {str(e)}")


def save_predictions_to_excel(predictions, output_file_path):
    """
    Сохраняет предсказания в файл Excel.

    Параметры:
    predictions (DataFrame): Датафрейм с предсказаниями.
    output_file_path (str): Путь к файлу для сохранения предсказаний.
    """
    predictions.to_excel(output_file_path, index=False)


# def predict_holtwinters_model(model_file_path, start, end, restore_freq=True, output_format="json"):
#     model = HoltWintersPressureModel()
#     model.load_model(model_file_path)
#     predictions = model.predict(start, end, restore_freq)
#
#     # Получение имени модели из пути к файлу модели
#     model_name = os.path.splitext(os.path.basename(model_file_path))[0]
#
#     if output_format == "excel":
#         # Формируем путь к файлу для сохранения предсказаний
#         model_folder = os.path.join("result_predictions", "holtwinter")
#
#         # Создание папки, если её нет
#         os.makedirs(model_folder, exist_ok=True)
#
#         output_file_path = os.path.join(
#             model_folder, f"{model_name}_predictions_{start.replace(':', '-')}_{end.replace(':', '-')}.xlsx"
#         )
#         save_predictions_to_excel(predictions, output_file_path)
#         return {"message": f"Predictions saved to {output_file_path}"}
#     else:
#         # Преобразование DataFrame в список словарей для JSON
#         predictions_dict = predictions.to_dict("records")
#         return {"predicted_data": predictions_dict}


def predict_holtwinters_model(model_file_path, start, end, restore_freq=True, output_format="json"):
    try:
        # Загружаем модель
        model = HoltWintersPressureModel()
        model.load_model(model_file_path)

        # Выполняем предсказание
        predictions = model.predict(start, end, restore_freq)

        # Получение имени модели из пути к файлу модели
        model_name = os.path.splitext(os.path.basename(model_file_path))[0]

        # Обработка формата вывода
        if output_format == "excel":
            # Формируем путь к файлу для сохранения предсказаний
            model_folder = os.path.join("result_predictions", "holtwinter")
            os.makedirs(model_folder, exist_ok=True)

            output_file_path = os.path.join(
                model_folder, f"{model_name}_predictions_{start.replace(':', '-')}_{end.replace(':', '-')}.xlsx"
            )

            save_predictions_to_excel(predictions, output_file_path)
            return {"message": f"Предсказания сохранены в {output_file_path}"}

        # Преобразование DataFrame в список словарей для JSON
        predictions_dict = predictions.to_dict("records")
        return {"predicted_data": predictions_dict}

    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Файл модели не найден. Проверьте путь к файлу.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка при загрузке модели или параметрах предсказания: {str(e)}")
    except IOError:
        raise HTTPException(
            status_code=500, detail="Ошибка при сохранении предсказаний в Excel. Проверьте доступ к файлу."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Произошла непредвиденная ошибка: {str(e)}")


# def load_model_summary(model_file_path):
#     model = HoltWintersPressureModel()
#     model.load_model(model_file_path)
#
#     model_summary = ModelSummary(
#         aic=round(model.model.aic),
#         bic=round(model.model.bic),
#         sse=round(model.model.sse),
#         training_start_date=str(model.training_start_date),
#         training_end_date=str(model.training_end_date),
#         num_original_data_points=model.num_original_data_points,
#         original_freq=model.original_freq,
#         aggregated_freq=model.aggregated_freq,
#         seasonal_periods=model.seasonal_periods,
#         trend=model.trend,
#         seasonal=model.seasonal,
#         training_duration=model.training_duration,
#     )
#
#     return model_summary


def load_model_summary(model_file_path):
    try:
        # Создаем экземпляр модели и загружаем её
        model = HoltWintersPressureModel()
        try:
            model.load_model(model_file_path)
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="Файл модели не найден. Проверьте путь к файлу.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Ошибка при загрузке модели: {str(e)}")

        # Проверка наличия метрик качества модели
        try:
            aic = round(model.model.aic)
            bic = round(model.model.bic)
            sse = round(model.model.sse)
        except AttributeError as e:
            raise HTTPException(status_code=500, detail=f"Не удалось извлечь метрики качества модели: {str(e)}")

        # Создаем сводную информацию о модели
        model_summary = ModelSummary(
            aic=aic,
            bic=bic,
            sse=sse,
            training_start_date=str(model.training_start_date),
            training_end_date=str(model.training_end_date),
            num_original_data_points=model.num_original_data_points,
            original_freq=model.original_freq,
            aggregated_freq=model.aggregated_freq,
            seasonal_periods=model.seasonal_periods,
            trend=model.trend,
            seasonal=model.seasonal,
            training_duration=model.training_duration,
        )

        return model_summary

    except HTTPException as http_exc:
        # Повторно выбрасываем исключения HTTP, если они уже были сформированы
        raise http_exc
    except Exception as e:
        # Обработка любых других непредвиденных ошибок
        raise HTTPException(status_code=500, detail=f"Произошла непредвиденная ошибка: {str(e)}")

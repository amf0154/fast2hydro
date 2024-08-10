from fastapi import APIRouter, HTTPException, Query

from ..schemas.holtwinter_schemas import MetricsResponse, ModelSummary
from ..service.holtwinter_service import load_model_summary, predict_holtwinters_model, train_holtwinters_model
from ..service.utils import calculate_metrics

router = APIRouter()


@router.post("/train")
def train_holtwinter_model(
    file_path: str = Query(
        ...,
        description="Путь к файлу с данными (первый столбец Дата, второй Значение)",
        example="fast2hydro/data/test_train_data.xlsx",
    ),
    model_name: str = Query(None, description="Уникальное имя модели", example="my_holtwinter_model"),
    seasonal_periods: int = Query(
        None, description="Сезонный период (определяется автоматически, если не задан)", example=""
    ),
    trend: str = Query(None, description="Тип тренда ('add', 'mul' или None (по умолчанию))", example=""),
    seasonal: str = Query("add", description="Тип сезонности ('add' (по умолчанию) или 'mul')", example="add"),
    aggregation_freq: str = Query(
        None, description="Частота агрегации данных ('5min', '10min', '30min')", example="5min"
    ),
):
    try:
        train_holtwinters_model(file_path, model_name, seasonal_periods, trend, seasonal, aggregation_freq)
        return {"message": "Модель Хольта-Винтерса успешна обучена"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict")
def predict_holtwinter_model(
    model_file_path: str = Query(
        ...,
        description="Путь к файлу модели для предсказания",
        example="stored_models/holtwinter/my_holtwinter_model.pkl",
    ),
    start_date: str = Query(
        ...,
        description="Дата начала предсказания в формате YYYY-MM-DD или YYYY-MM-DD HH:MM",
        example="2024-06-07 23:59",
    ),
    end_date: str = Query(
        ...,
        description="Дата окончания предсказания в формате YYYY-MM-DD или YYYY-MM-DD HH:MM",
        example="2024-06-08 23:59",
    ),
    restore_freq: bool = Query(True, description="Восстановить исходную частоту данных", example=True),
    output_format: str = Query(
        "json", description="Формат вывода данных в result\\holtwinter: 'json' или 'excel'", example="json"
    ),
):
    try:
        result = predict_holtwinters_model(model_file_path, start_date, end_date, restore_freq, output_format)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/summary", response_model=ModelSummary)
def get_model_summary(
    model_name: str = Query(
        ...,
        description="Путь модели для получения резюме",
        example="fast2hydro/stored_models/my_holtwinter_model/my_holtwinter_model.pkl",
    )
):
    try:
        model_summary = load_model_summary(model_name)
        return model_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate_metrics", response_model=MetricsResponse)
def calculate_model_metrics(
    actual_file_path: str = Query(
        ..., description="Путь к файлу с реальными данными", example="/path/to/actual_data.xlsx"
    ),
    predicted_file_path: str = Query(
        ..., description="Путь к файлу с прогнозируемыми данными", example="/path/to/predicted_data.xlsx"
    ),
):
    try:
        metrics = calculate_metrics(actual_file_path, predicted_file_path)
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any


# Модель для входных данных
class TrainHoltWintersModelRequest(BaseModel):
    file_path: str = Field(
        ..., example="e:/+Python/fast2hydro/data/test_train_data.xlsx", description="Путь к файлу с данными"
    )
    model_name: str = Field(None, example="my_holtwinter_model", description="Уникальное имя модели")
    seasonal_periods: int = Field(
        None, example=288, description="Сезонный период (определяется автоматически, если не задан)"
    )
    trend: str = Field(None, example="add", description="Тип тренда ('add', 'mul' или None)")
    seasonal: str = Field("add", example="add", description="Тип сезонности ('add' (по умолчанию) или 'mul')")
    aggregation_freq: str = Field(
        None, example="5min", description="Частота агрегации данных ('5min', '10min', '30min')"
    )


# Модель входных данных
class PredictHoltWintersModelRequest(BaseModel):
    model_file_path: str = Field(
        ...,
        description="Путь к файлу обученной модели для предсказания",
        example="e:/+Python/fast2hydro/stored_models/holtwinter/my_holtwinter_model.pkl",
    )
    start: str = Field(
        ...,
        description="Дата начала предсказания в формате 'DD-MM-YYYY' или 'DD-MM-YYYY HH:MM'",
        example="08-06-2024 00:00",
    )
    end: str = Field(
        ...,
        description="Дата окончания предсказания в формате 'DD-MM-YYYY' или 'DD-MM-YYYY HH:MM'",
        example="08-06-2024 00:15",
    )
    restore_freq: bool = Field(
        default=True,
        description="Восстановить исходную частоту данных",
        example=True,
    )
    output_format: str = Field(
        default="json",
        description="Формат вывода данных: 'json' или 'excel'",
        example="json",
    )


# Модель ответа
class PredictHoltWintersModelResponse(BaseModel):
    success: bool = Field(
        ...,
        description="Указывает, была ли операция успешной",
        example=True,
    )
    message: str = Field(
        ...,
        description="Сообщение об успешности операции или описание ошибки",
        example="Предсказание успешно выполнено",
    )
    data: Optional[Any] = Field(
        None,
        description="Результаты предсказания в указанном формате",
        example={
            "07-06-2024- 23:59": 123.45,
            "08-06-2024 00:00": 127.89,
            # ... другие предсказанные значения
        },
    )


# Модель для ответа при обучении
class TrainHoltWintersModelResponse(BaseModel):
    success: bool  # Успешность операции
    message: str  # Сообщение об ошибке или успехе


class ModelSummary(BaseModel):
    aic: float
    bic: float
    sse: float
    training_start_date: str
    training_end_date: str
    num_original_data_points: int
    original_freq: str | None
    aggregated_freq: str | None
    seasonal_periods: int | None
    trend: str | None
    seasonal: str | None
    training_duration: str | None


class MetricsResponse(BaseModel):
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float


class MetricsRequest(BaseModel):
    real_data: List[Dict] = Field(
        ...,
        example=[{"date": "11-08-2024 00:00", "value": 100.5}, {"date": "12-08-2024 00:00", "value": 102.3}],
        description="Список реальных значений",
    )

    predicted_data: List[Dict] = Field(
        ...,
        example=[{"date": "11-08-2024 00:00", "value": 101.0}, {"date": "12-08-2024 00:00", "value": 103.0}],
        description="Список прогнозируемых значений",
    )

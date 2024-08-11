from pydantic import BaseModel, Field
from typing import List, Dict


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
        example=[{"date": "2024-08-11 00:00", "value": 100.5}, {"date": "2024-08-12 00:00", "value": 102.3}],
        description="Список реальных значений",
    )

    predicted_data: List[Dict] = Field(
        ...,
        example=[{"date": "2024-08-11 00:00", "value": 101.0}, {"date": "2024-08-12 00:00", "value": 103.0}],
        description="Список прогнозируемых значений",
    )

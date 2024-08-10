from pydantic import BaseModel


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


class MetricsResponse(BaseModel):
    mean_absolute_error: float
    root_mean_squared_error: float
    mean_absolute_percentage_error: float

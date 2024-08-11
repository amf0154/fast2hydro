from fastapi import APIRouter, HTTPException, Query, Body

from ..schemas.holtwinter_schemas import ModelSummary, MetricsResponse, MetricsRequest
from ..service.holtwinter_service import load_model_summary, predict_holtwinters_model, train_holtwinters_model
from ..service.utils import calculate_metrics_from_json

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
    """
    Обучает модель Хольта-Винтерса на основе предоставленных данных.

    Эндпоинт принимает путь к файлу с данными, которые содержат даты в первом столбце и соответствующие
    значения во втором столбце. Пользователь может задать уникальное имя для модели,
    указать сезонный период, тип тренда и сезонности, а также частоту агрегации данных.
    Если параметры не заданы, будут использованы значения по умолчанию или автоматически определенные параметры.
    После обучения модель сохраняется, и возвращается путь к файлу с моделью.

    Возвращает: обученную модель с расширением pkl с указанием пути сохранения
    """

    try:
        model_file_path = train_holtwinters_model(
            file_path, model_name, seasonal_periods, trend, seasonal, aggregation_freq
        )
        return {"message": f"Модель Хольта-Винтерса успешно обучена и сохранена в '{model_file_path}'"}
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
    """
    Выполняет предсказание на основе обученной модели Хольта-Винтерса.

    Эндпоинт принимает путь к файлу с обученной моделью и диапазон дат для выполнения предсказания.
    Пользователь может указать, восстановить ли исходную частоту данных и в каком формате сохранить
    результат (JSON или Excel). После успешного выполнения возвращает предсказанные данные в выбранном формате.

    Возвращает:
    - Предсказанные данные в указанном формате.
    """

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
    """
    Возвращает резюме обученной модели Хольта-Винтерса.

    Метод создает и возвращает резюме модели, включая информацию о критериях, частотах данных,
    длительности обучения и других ключевых параметрах.

    Возвращает:
    - **aic (float):** Значение критерия Акаике, используемого для оценки качества модели.
    - **bic (float):** Значение байесовского информационного критерия, используемого для оценки качества модели.
    - **sse (float):** Сумма квадратов ошибок (Sum of Squared Errors) модели.
    - **training_start_date (str):** Дата начала обучения модели.
    - **training_end_date (str):** Дата окончания обучения модели.
    - **num_original_data_points (int):** Количество исходных точек данных, использованных для обучения модели.
    - **original_freq (str | None):** Исходная частота данных до агрегации (если применимо).
    - **aggregated_freq (str | None):** Частота данных после агрегации (если применимо).
    - **seasonal_periods (int | None):** Сезонный период модели, если был использован.
    - **trend (str | None):** Тип тренда ('add', 'mul' или None), использованный в модели.
    - **seasonal (str | None):** Тип сезонности ('add' или 'mul'), использованный в модели.
    - **training_duration (str | None):** Длительность обучения модели в формате 'HH:MM:SS'.
    """

    try:
        model_summary = load_model_summary(model_name)
        return model_summary
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/calculate_metrics_from_json", response_model=MetricsResponse)
def calculate_metrics_from_json_endpoint(
    request: MetricsRequest = Body(..., description="JSON данные с реальными и прогнозируемыми значениями.")
):
    """
    Рассчитывает метрики погрешности между реальными и прогнозируемыми данными.

    Эндпоинт принимает два списка данных: реальные и прогнозируемые значения.
    Каждый элемент в списках должен содержать дату в формате 'YYYY-MM-DD HH:MM' и значение.
    Возвращает различные метрики погрешности, такие как средняя абсолютная ошибка,
    среднеквадратичная ошибка и процентная ошибка.
    """

    try:
        # Используем данные из запроса
        real_data = request.real_data
        predicted_data = request.predicted_data

        metrics = calculate_metrics_from_json(real_data, predicted_data)

        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

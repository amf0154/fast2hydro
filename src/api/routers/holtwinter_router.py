from fastapi import APIRouter, HTTPException, Query, Body

from ..schemas.holtwinter_schemas import (
    ModelSummary,
    MetricsResponse,
    MetricsRequest,
    TrainHoltWintersModelRequest,
    TrainHoltWintersModelResponse,
    PredictHoltWintersModelResponse,
    PredictHoltWintersModelRequest,
)
from ..service.holtwinter_service import load_model_summary, predict_holtwinters_model, train_holtwinters_model
from ..service.utils import calculate_metrics_from_json

router = APIRouter()


@router.post("/train")
def train_holtwinter_model(
    request: TrainHoltWintersModelRequest = Body(..., description="Параметры для обучения модели Хольта-Винтерса")
):
    """
    Обучает модель Хольта-Винтерса на основе предоставленных данных.

    Эндпоинт принимает путь к файлу с данными, которые должны содержать следующие данные:

    - **Дата:** Даты должны находиться в первом столбце файла.
    - **Значения:** Второй столбец должен содержать числовые значения, соответствующие каждой дате

    Пользователь может задать следующие параметры:

    - **model_name:** Уникальное имя для модели, чтобы отличать её от других моделей.
    - **seasonal_periods:** Сезонный период, который будет использоваться моделью.
    Если не задан, период будет определен автоматически.
    - **trend:** Тип тренда для модели ('add', 'mul' или None).
    - **seasonal:** Тип сезонности для модели ('add' или 'mul').
    - **aggregation_freq:** Частота агрегации данных (например, '5min', '10min', '30min').

    Если параметры не заданы, будут использованы значения по умолчанию или автоматически определенные параметры.

    После успешного обучения модель будет сохранена в файл с расширением .pkl, и возвращается путь к файлу с моделью.

    Возвращает:
    - **success:** Булево значение, указывающее на успешность операции.
    - **message:** Сообщение об успешности операции или описание ошибки.
    """
    try:
        train_holtwinters_model(
            file_path=request.file_path,
            model_name=request.model_name,
            seasonal_periods=request.seasonal_periods,
            trend=request.trend,
            seasonal=request.seasonal,
            aggregation_freq=request.aggregation_freq,
        )
        return TrainHoltWintersModelResponse(success=True, message="Модель Хольта-Винтерса успешно обучена")
    except Exception as e:
        return TrainHoltWintersModelResponse(success=False, message=str(e))


@router.post("/predict", response_model=PredictHoltWintersModelResponse)
def predict_holtwinter_model(
    request: PredictHoltWintersModelRequest = Body(
        ..., description="Параметры для выполнения предсказания модели Хольта-Винтерса"
    )
):
    """
    Выполняет предсказание на основе обученной модели Хольта-Винтерса.

    **Описание входных данных:**
    - **model_file_path**: Путь к файлу с обученной моделью для предсказания.
    - **start_date**: Дата начала предсказания в формате `'DD-MM-YYYY'` или `'DD-MM-YYYY HH:MM'`.
    - **end_date**: Дата окончания предсказания в формате `'DD-MM-YYYY'` или `'DD-MM-YYYY HH:MM'`.
    - **restore_freq**: Флаг, указывающий на необходимость восстановления исходной частоты данных. По умолчанию `True`.
    - **output_format**: Формат вывода данных. Допустимые значения: `'json'` или `'excel'`. По умолчанию `'json'`.

    **Возвращает:**
    - **success**: Булево значение, указывающее на успешность операции.
    - **message**: Сообщение об успешности операции или описание ошибки.
    - **data**: Результаты предсказания в указанном формате. В случае ошибки значение будет `null`.

    **Пример успешного ответа (JSON):**
    ```json
    {
      "success": true,
      "message": "Предсказание успешно выполнено",
      "data": {
        "07-06-2024 23:59": 123.45,
        "08-06-2024 00:00": 127.89
      }
    }
    ```

    **Пример ошибки:**
    ```json
    {
      "success": false,
      "message": "Описание ошибки",
      "data": null
    }
    ```
    """
    try:
        # Вызов функции предсказания
        prediction_result = predict_holtwinters_model(
            model_file_path=request.model_file_path,
            start=request.start,
            end=request.end,
            restore_freq=request.restore_freq,
            output_format=request.output_format,
        )

        return PredictHoltWintersModelResponse(
            success=True, message="Предсказание успешно выполнено", data=prediction_result
        )
    except Exception as e:
        return PredictHoltWintersModelResponse(success=False, message=str(e), data=None)


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

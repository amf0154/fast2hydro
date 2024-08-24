from fastapi.testclient import TestClient

from .main import app
import pandas as pd

client = TestClient(app)


def excel_to_json(file_path):
    # Загрузка данных из Excel
    df = pd.read_excel(file_path)

    # Предположим, что столбцы в Excel называются "date" и "value"
    df.columns = ["date", "value"]  # Переименуйте столбцы, если это необходимо

    # Преобразование данных в формат JSON
    json_data = []
    for index, row in df.iterrows():
        json_data.append({"date": row["date"].strftime("%Y-%m-%d %H:%M"), "value": row["value"]})

    return json_data


# def test_calculate_metrics_from_json():
#
#     # Пример данных для тестирования
#     test_data = {
#         "real_data": [{"date": "2024-08-11 00:00", "value": 100.5}, {"date": "2024-08-12 00:00", "value": 102.3}],
#         "predicted_data": [{"date": "2024-08-11 00:00", "value": 101}, {"date": "2024-08-12 00:00", "value": 103}],
#     }
#
#     # Ожидаемый результат
#     expected_response = {
#         "mean_absolute_error": 0.6,
#         "root_mean_squared_error": 0.61,
#         "mean_absolute_percentage_error": 0.01,
#     }
#
#     # Отправка POST запроса к эндпоинту
#     response = client.post("/calculate_metrics_from_json", json=test_data)
#
#     # Проверка успешного выполнения запроса
#     assert response.status_code == 200
#
#     # Проверка соответствия ответа ожидаемым метрикам
#     assert response.json() == expected_response


# def test_train_holtwinter_model():
#     # Путь к тестовому файлу
#     file_path = "e:/+Python/fast2hydro/data/test_train_data.xlsx"
#
#     # Параметры запроса
#     params = {
#         "file_path": file_path,
#         "model_name": "my_holtwinter_model",
#         # "seasonal_periods": None,
#         # "trend": None,
#         "seasonal": "add",
#         "aggregation_freq": "5min"
#     }
#
#     # Отправка GET-запроса к API
#     response = client.post("/holtwinter/train", params=params)
#
#     # Проверка ответа
#     assert response.status_code == 200, f"Expected 200, but got {response.status_code}.
#     Response content: {response.content}"
#     response_data = response.json()

#     # Проверка, что в ответе содержится сообщение об успешном обучении модели
#     assert "Модель Хольта-Винтерса успешно обучена и сохранена" in response_data["message"]


def test_predict_endpoint():
    # Задаем параметры для теста
    test_data = {
        "model_file_path": "e:/+Python/fast2hydro/stored_models/holtwinter/my_holtwinter_model.pkl",
        "start": "24-08-2023 00:00",
        "end": "25-08-2023 23:59",
        "output_format": "json",
    }

    # Выполняем POST запрос к эндпоинту /predict
    response = client.post("/holtwinter/predict", json=test_data)

    # Проверяем успешность выполнения запроса
    assert response.status_code == 200

    # Преобразуем JSON-ответ в словарь
    response_data = response.json()

    # Проверяем, что ответ содержит ключи 'success', 'message', и 'data'
    assert "success" in response_data
    assert "message" in response_data
    assert "data" in response_data

    # Проверяем, что 'success' равно True
    assert response_data["success"] is True

    # Проверяем, что сообщение не пустое
    assert len(response_data["message"]) > 0

    # Если 'data' присутствует, проверяем его структуру
    if response_data["data"]:
        assert isinstance(response_data["data"], dict)
        # Проверяем, что в каждом элементе данных есть корректные ключи (дата и значение)
        for date, value in response_data["data"].items():
            assert isinstance(date, str)  # Дата в формате строки
            assert isinstance(value, (int, float))  # Значение - число

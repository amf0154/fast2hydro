from fastapi.testclient import TestClient

from .main import app

client = TestClient(app)


def test_calculate_metrics_from_json():
    # Пример данных для тестирования
    test_data = {
        "real_data": [{"date": "2024-08-11 00:00", "value": 100.5}, {"date": "2024-08-12 00:00", "value": 102.3}],
        "predicted_data": [{"date": "2024-08-11 00:00", "value": 101}, {"date": "2024-08-12 00:00", "value": 103}],
    }

    # Ожидаемый результат
    expected_response = {
        "mean_absolute_error": 0.6,
        "root_mean_squared_error": 0.61,
        "mean_absolute_percentage_error": 0.01,
    }

    # Отправка POST запроса к эндпоинту
    response = client.post("/calculate_metrics_from_json", json=test_data)

    # Проверка успешного выполнения запроса
    assert response.status_code == 200

    # Проверка соответствия ответа ожидаемым метрикам
    assert response.json() == expected_response

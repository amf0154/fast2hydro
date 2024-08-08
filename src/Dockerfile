# Используем официальный образ Python 3.10 в качестве базового
FROM python:3.10

# Устанавливаем рабочую директорию в контейнере
WORKDIR /code

# Копируем файл зависимостей в рабочую директорию
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Копируем все файлы проекта в рабочую директорию
COPY . .

# Открываем порт 8000 для доступа к приложению
EXPOSE 3100

# Команда для запуска FastAPI приложения с использованием gunicorn и uvicorn
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "api.main:app"]

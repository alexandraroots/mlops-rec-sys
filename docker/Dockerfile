FROM python:3.13-slim AS base

WORKDIR /app

# Установить зависимости напрямую
# Копируем метаданные и код
COPY pyproject.toml ./
COPY src ./src
COPY models ./models

# Теперь ставим
RUN pip install --no-cache-dir .

EXPOSE 50051

CMD ["python", "src/mlops_rec_sys/server.py"]

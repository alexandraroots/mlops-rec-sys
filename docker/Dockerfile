FROM python:3.13-slim AS base

WORKDIR /app

RUN pip install poetry

COPY ../pyproject.toml poetry.lock ./
COPY ../src ./src
COPY ../models ./models

RUN poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi

EXPOSE 50051

CMD ["python", "src/mlops_rec_sys/server.py"]
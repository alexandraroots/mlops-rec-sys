FROM python:3.13-slim AS base

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY models ./models

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src:/app/src/mlops_rec_sys/proto:$PYTHONPATH

EXPOSE 50051

CMD ["python", "src/mlops_rec_sys/server.py"]

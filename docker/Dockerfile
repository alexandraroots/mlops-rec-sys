FROM python:3.13-slim AS base

WORKDIR /app

COPY requirements.txt ./
COPY src ./src
COPY models ./models

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 50051

CMD ["python", "src/mlops_rec_sys/server.py"]
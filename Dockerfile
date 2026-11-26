FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY models ./models
COPY data/train ./data/train
COPY data/test ./data/test

# writable dirs for the database, uploads and retrained models
# (cloud platforms such as Hugging Face Spaces run the container as a non-root user)
RUN mkdir -p data/uploads models/archive store && chmod -R 777 data models store

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]

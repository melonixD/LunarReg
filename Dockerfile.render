FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    SITE_DIR=/app/site \
    CRATER_MODEL_PATH=/app/models/crater_detector.pt

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY inference-service/requirements.render.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY inference-service/app.py ./app.py
COPY inference-service/models ./models

COPY index.html ./site/index.html
COPY assets ./site/assets
COPY docs ./site/docs
COPY downloads ./site/downloads
COPY samples ./site/samples

EXPOSE 10000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]

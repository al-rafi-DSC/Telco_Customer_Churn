FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 GRADIO_ANALYTICS_ENABLED=False \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PORT=8000
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY src/app ./src/app
COPY src/serving/inference.py ./src/serving/inference.py
COPY src/serving/bundle ./src/serving/bundle
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '8000') + '/', timeout=4)"
CMD ["python", "-m", "src.app.serve"]

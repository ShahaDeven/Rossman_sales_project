# Rossmann Sales Forecasting API — Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY api/          ./api/
COPY model/config.yaml ./model/config.yaml
COPY scaler.pkl    ./scaler.pkl
COPY features.json ./features.json
COPY mlflow.db     ./mlflow.db
COPY model_state_dict.pth ./model_state_dict.pth

# Create directory for prediction logs
RUN mkdir -p /app/logs

RUN find /app -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Health check — Docker will restart container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Run the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
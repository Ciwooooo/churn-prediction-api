# Stage 1: Builder stage
FROM python:3.11-slim AS builder

# set the workdir
WORKDIR /app

# install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# copy pyproject
COPY pyproject.toml .

# Install Python dependencies is venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Stage 2: runtime
FROM python:3.11-slim

# workidr setup
WORKDIR /app

# copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# set env variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \ 
    MODEL_PATH=/app/models/churn_model.pkl

# copy cource code
COPY src/ /app/src/
COPY models/ /app/models/

# create a non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# port expose
EXPOSE 5000

# health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')"

# run the app with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "60", "src.app:app"]
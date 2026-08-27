FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY services/navigation-engine/requirements.txt /app/services/navigation-engine/
COPY services/ml/requirements.txt /app/services/ml/
COPY services/api/requirements.txt /app/services/api/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/services/navigation-engine/requirements.txt && \
    pip install --no-cache-dir -r /app/services/ml/requirements.txt && \
    pip install --no-cache-dir -r /app/services/api/requirements.txt

# Copy source trees
COPY services /app/services
COPY simulation /app/simulation
COPY data /app/data

ENV PYTHONPATH="/app:/app/services/navigation-engine:/app/services/ml"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "services.api.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

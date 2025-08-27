# Dockerfile
FROM python:3.11-slim

# System deps (psycopg2-binary needs these at runtime)
RUN apt-get update && apt-get install -y build-essential libpq5 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Copy env at runtime via docker-compose; not baking secrets into image

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

# Use Uvicorn in production mode (no --reload), multiple workers
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*", "--workers", "3"]

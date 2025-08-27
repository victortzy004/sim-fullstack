# Fast, small image
FROM python:3.11-slim

# System deps (psycopg needs libpq)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev curl && \
    rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Copy only requirements first (better caching)
COPY requirements.txt /app/

# Install deps (psycopg3 recommended)
# requirements.txt should include:
# fastapi uvicorn gunicorn sqlalchemy "psycopg[binary]" alembic pydantic
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app
COPY . /app

# Gunicorn run (2-4 workers is fine to start)
ENV WEB_CONCURRENCY=2
EXPOSE 8000
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "--workers", "2", "backend.main:app"]

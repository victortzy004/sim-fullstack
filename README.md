## 42 DPM API (FastAPI + SQLAlchemy)

Production-ready FastAPI backend for 42:DPM.
Exposes REST endpoints (see `/docs`) and persists to Postgres (or SQLite for quick local dev).

---

# Backend Setup

## Requirements

- Python 3.11+
- `pip` (or `uv`)
- Postgres 14+ (local or hosted) — **or** SQLite for quick start
- (Optional) Docker & Docker Compose
- (Optional) gcloud CLI for Cloud Run

---

## Environment

Create `backend/.env`:

```env
# --- Database ---
# Quick local (SQLite):
# DATABASE_URL=sqlite:///./app.db

# Local/Cloud Postgres (psycopg v3 driver):
# DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME?sslmode=require
DATABASE_URL=sqlite:///./app.db

# Connection pool (optional)
DB_POOL_SIZE=5
DB_MAX_OVERFLOW=10
DB_POOL_RECYCLE_SEC=1800

# CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:8501,http://localhost:8505
```

> Make sure your code reads `DATABASE_URL` and `CORS_ORIGINS` (it already does in your `db.py` / `main.py`).

---

## Install deps (local)

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**Minimal `backend/requirements.txt`**

```
fastapi
uvicorn[standard]
gunicorn
SQLAlchemy>=2.0
alembic
pydantic
python-dotenv
psycopg2-binary>=2.9
typing-extensions
```

---

## Run locally (SQLite quick start)

### Backend Setup

From root folder:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

## Open: [http://localhost:8000/docs](http://localhost:8000/docs)

## Docker (local, with Postgres and API)

Top-level `docker-compose.yml` (you already have this). From repo root:

```bash
docker compose build --no-cache
docker compose up -d

# optional: force recreate even if configs look same
# docker compose up -d --force-recreate
```

- API on [http://localhost:8000](http://localhost:8000)
- DB internal name: `db` (connection string inside Docker: `postgresql+psycopg://app:supersecret@db:5432/appdb`)

> If you add a **frontend** container, keep `API_BASE_URL=http://api:8000` (service name).

---

# Frontend Setup

## 42 DPM Frontend (Streamlit)

Streamlit app that talks to the FastAPI backend via REST.
Key env var is `API_BASE_URL`.

---

## Requirements

- Python 3.11+
- `pip` (or `uv`)
- (Optional) Docker & Docker Compose
- A reachable API (local `http://127.0.0.1:8000` or your Cloud Run URL)

---

## Environment

Create `frontend/.env`:

```env
# Local machine (talk to locally running API)
API_BASE_URL=http://127.0.0.1:8000

# When running inside Docker Compose, the API is the service name on the network:
# API_BASE_URL=http://api:8000

# When using Cloud Run:
# API_BASE_URL=https://dpm-api-xxxxxxxx-uc.a.run.app
```

`api.py` already reads `API_BASE_URL` with a default fallback.

---

## Install & run locally

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Run on 8501 (default)
python -m streamlit run app.py --server.port 8501
```

Open: [http://localhost:8501](http://localhost:8501)

**Minimal `frontend/requirements.txt`**

```
streamlit==1.36.*
pandas==2.2.*
numpy==1.26.*
plotly==5.*
requests==2.*
python-dotenv==1.0.*
```

---

## Docker (local)

**Dockerfile** (in `frontend/`):

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit's default port is 8501
EXPOSE 8501
ENV PORT=8501

# Read API_BASE_URL at runtime
CMD ["sh", "-c", "python -m streamlit run app.py --server.port ${PORT} --server.address 0.0.0.0"]
```

**Compose** (top-level), example services:

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: supersecret
      POSTGRES_DB: appdb
    volumes: ["dbdata:/var/lib/postgresql/data"]
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U app -d appdb"]
      interval: 5s
      timeout: 5s
      retries: 10

  api:
    build: .
    env_file: backend/.env
    depends_on:
      db:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql+psycopg://app:supersecret@db:5432/appdb
      CORS_ORIGINS: "http://localhost:8505"
    ports:
      - "8000:8000"

  frontend:
    build:
      context: ./frontend
    environment:
      API_BASE_URL: "http://api:8000" # talk to the api service by name
    ports:
      - "8505:8501" # host:container
    depends_on:
      - api

volumes:
  dbdata:
```

Run:

```bash
docker compose up -d --build
```

Open:

- API: [http://localhost:8000/docs](http://localhost:8000/docs)
- Frontend: [http://localhost:8505](http://localhost:8505)

---

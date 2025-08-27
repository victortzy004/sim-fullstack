sweet — here are drop-in READMEs you can paste into your repo.

---

# backend/README.md

## 42 DPM API (FastAPI + SQLAlchemy)

Production-ready FastAPI backend for 42:DPM.
Exposes REST endpoints (see `/docs`) and persists to Postgres (or SQLite for quick local dev).

---

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
fastapi==0.112.*
uvicorn[standard]==0.30.*
gunicorn==22.*
SQLAlchemy==2.0.*
psycopg[binary]==3.2.*        # or psycopg2-binary if you use +psycopg2 URLs
python-dotenv==1.0.*
pydantic==2.*
starlette==0.38.*
```

---

## Run locally (SQLite quick start)

From root folder:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Open: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Run locally (Postgres)

Option A: your own Postgres instance
Option B (quick): run Postgres in Docker

```bash
docker run --name dpm-db -e POSTGRES_USER=app -e POSTGRES_PASSWORD=supersecret \
  -e POSTGRES_DB=appdb -p 5432:5432 -d postgres:16
```

Set in `backend/.env`:

```
DATABASE_URL=postgresql+psycopg://app:supersecret@localhost:5432/appdb
```

Start the API:

```bash
cd backend && uvicorn backend.main:app --reload --port 8000
```

---

## Docker (local, with Postgres and API)

Top-level `docker-compose.yml` (you already have this). From repo root:

```bash
docker compose up -d --build
docker compose ps
```

- API on [http://localhost:8000](http://localhost:8000)
- DB internal name: `db` (connection string inside Docker: `postgresql+psycopg://app:supersecret@db:5432/appdb`)

> If you add a **frontend** container, keep `API_BASE_URL=http://api:8000` (service name).

---

## Database init / migrations

You currently call `Base.metadata.create_all(bind=engine)` on startup, which is fine for dev.
For production, consider **Alembic** later. If you want a one-time init:

```bash
python -c "from backend.db import Base, engine; Base.metadata.create_all(bind=engine)"
```

**Gotcha:** If you saw `duplicate key value violates unique constraint ... _seq`, it means `create_all()` ran concurrently against Postgres. Prefer running it once (before scaling) or switch to Alembic.

---

## Production on Cloud Run (recommended)

### 1) Dockerfile (at repo root)

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "2", "-b", "0.0.0.0:8080", "backend.main:app"]
```

### 2) gcloud one-liner

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud config set run/region us-central1

gcloud run deploy dpm-api \
  --source . \
  --allow-unauthenticated \
  --set-env-vars DATABASE_URL="postgresql+psycopg://USER:PASSWORD@HOST/DB?sslmode=require" \
  --set-env-vars CORS_ORIGINS="https://your-frontend.vercel.app,https://*.yourdomain.com" \
  --min-instances=1 --concurrency=80
```

You’ll get a URL like `https://dpm-api-xxxx.a.run.app`. Open `.../docs`.

**Logs**

```bash
gcloud logs tail --service dpm-api
```

---

## Troubleshooting

- **`ModuleNotFoundError: psycopg2`**
  Use `psycopg[binary]` and a `postgresql+psycopg://...` connection string.

- **`password authentication failed / role does not exist`**
  Create the user/db in Postgres or update `DATABASE_URL` to match actual credentials.

- **CORS issues**
  Add your frontend origin(s) to `CORS_ORIGINS` env var (comma-separated).

- **Gunicorn binds to 8080**
  Cloud Run injects `$PORT` (8080). Locally you can still run with uvicorn on 8000.

---

# frontend/README.md

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

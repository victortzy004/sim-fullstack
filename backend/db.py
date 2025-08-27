# backend/db.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Read from env for prod; fall back to local SQLite for dev
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

# Heroku-style URLs sometimes come as 'postgres://'
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)

# If you use psycopg3 (recommended):
# pip install "psycopg[binary]"
# Example DSN:
# postgresql+psycopg://user:pass@host:5432/dbname?sslmode=require

is_sqlite = DATABASE_URL.startswith("sqlite")

engine_kwargs = {
    "pool_pre_ping": True,  # drops dead/stale connections safely
}

if is_sqlite:
    # SQLite needs this when used in FastAPI threadpool
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Reasonable defaults; adjust to your DB limits
    engine_kwargs.update(
        pool_size=int(os.getenv("DB_POOL_SIZE", "5")),
        max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "10")),
        pool_recycle=int(os.getenv("DB_POOL_RECYCLE_SEC", "1800")),  # 30 min
    )

engine = create_engine(DATABASE_URL, **engine_kwargs)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

def get_db():
    """FastAPI dependency that yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

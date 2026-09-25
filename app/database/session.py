# app/database/session.py
import os
from pathlib import Path
from typing import Generator
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Load environment variables (.env first, fallback to .env.example)
_root_dir = Path(__file__).resolve().parents[2]
_env_path = _root_dir / ".env"
if not _env_path.exists():
    _env_path = _root_dir / ".env.example"

if _env_path.exists():
    load_dotenv(_env_path)

# 2. Extract database parameters safely from environment
DB_DRIVER = os.getenv("DB_DRIVER", "postgresql+psycopg2")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "#PassLetter#")
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = str(os.getenv("DB_PORT", "5000")).strip()
DB_NAME = os.getenv("DB_NAME", "roadessy")

# Ensure port is valid
if not DB_PORT.isdigit():
    DB_PORT = "5000"

# Safely encode special characters in password (e.g., # becomes %23)
encoded_password = quote_plus(DB_PASSWORD)

# Construct sanitized connection string
DATABASE_URL: str = f"{DB_DRIVER}://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

IS_POSTGRES = DATABASE_URL.startswith("postgresql") or DATABASE_URL.startswith("postgres")

if not IS_POSTGRES and os.getenv("ROADESSY_ALLOW_SQLITE", "false").lower() != "true":
    raise RuntimeError(
        "PostgreSQL with PostGIS is required. Set DATABASE_URL to a PostgreSQL URL; "
        "set ROADESSY_ALLOW_SQLITE=true only for isolated local tests."
    )

if IS_POSTGRES:
    engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
else:
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def ensure_postgres_extensions(db):
    if not IS_POSTGRES:
        return
    try:
        db.execute(text('CREATE EXTENSION IF NOT EXISTS "postgis"'))
        db.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
        db.commit()
    except OperationalError as exc:
        db.rollback()
        raise RuntimeError("PostGIS extensions could not be enabled; check database privileges") from exc
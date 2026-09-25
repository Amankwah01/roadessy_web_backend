"""Simple helper to run Alembic upgrade head programmatically using the app's DATABASE_URL."""
import os
import sys

from alembic.config import Config
from alembic import command

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from app.database.session import DATABASE_URL, engine as app_engine
from sqlalchemy import inspect


def main():
    here = os.path.dirname(__file__)
    cfg = Config(os.path.join(here, "..", "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
    print("Applying alembic migrations using:", DATABASE_URL)

    try:
        command.upgrade(cfg, "head")
        print("Migrations applied successfully.")
    except Exception as exc:
        print(f"Migration apply failed: {exc}")
        print("Falling back to create_all for table creation.")
        from app.database.session import Base
        Base.metadata.create_all(bind=app_engine)


if __name__ == "__main__":
    main()
